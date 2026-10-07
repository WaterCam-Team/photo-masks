"""The inner loop: fine-tune SegFormer on a labeled set, score it on a fixed val set.

This is the step every RL batch pays for, so it comes in three speeds:

    full     from the ImageNet MiT weights, the training package's recipe
             (epochs over the labeled set). The reference; slow.
    warm     from the previous batch's model, a fixed number of steps over the
             whole labeled set (old and new masks together, so the new batch
             does not simply overwrite what the model knew).
    decoder  encoder frozen at a base model; its features are computed once per
             scene (a fixed set of augmented crops) and only the all-MLP decode
             head trains. Seconds instead of minutes, at the price that the
             encoder, including its thermal and NIR filters, stops adapting.

Every mode scores the *final* model, never the best of several evaluations: the
score is a reward, and taking the max over noisy evaluations on a small val set
biases it upward (that is how an epoch-0 checkpoint became `best.pt`).

Band statistics are fixed per campaign (`Campaign.stats`), not re-measured per
batch: a warm-started model must keep seeing inputs normalised the way it was
trained.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from training import models
from training import stats as ST
from training.data import Geometry, SceneDataset
from training.engine import evaluate
from training.losses import SegLoss
from training.stats import Normalizer

ARCH = "segformer-b0"
MODALITY = "fiveband"


@dataclass
class Campaign:
    """What stays fixed across every batch of one labeling campaign."""

    val_rows: list[dict]
    stats: ST.BandStats
    crop: int = 512
    batch: int = 2
    eval_long_side: int | None = None      # None = native frame, as training scores
    device: str = "cpu"
    amp: bool = True                       # fp16 autocast, CUDA only
    workers: int | None = None             # DataLoader workers; None = 0 on CPU, 4 on CUDA
    loss: str = "ce+lovasz"
    _val_ds: SceneDataset | None = field(default=None, repr=False)

    @classmethod
    def from_rows(cls, stats_rows: list[dict], val_rows: list[dict], **kw) -> "Campaign":
        return cls(val_rows=val_rows,
                   stats=ST.accumulate(stats_rows, MODALITY, method="meanstd"), **kw)

    @property
    def norm(self) -> Normalizer:
        return Normalizer(self.stats, "meanstd")

    def val_loader(self):
        from torch.utils.data import DataLoader
        if self._val_ds is None:             # cached: the val TIFFs are read once
            self._val_ds = SceneDataset(
                self.val_rows, MODALITY, self.norm,
                Geometry(eval_long_side=self.eval_long_side), train=False, cache=True)
        return DataLoader(self._val_ds, batch_size=1)

    @property
    def use_amp(self) -> bool:
        return self.amp and self.device.startswith("cuda")

    def loss_fn(self) -> SegLoss:
        return SegLoss(self.loss, self.stats.class_weights())


def new_model(init: str | None = None):
    """ImageNet MiT-B0 with a 5-band stem, or a saved checkpoint dir."""
    if init is None:
        return models.build(ARCH, in_channels=5, num_classes=2)
    return models.load(ARCH, init, in_channels=5)


def clone(model):
    import copy
    return copy.deepcopy(model)


def score(model, camp: Campaign) -> dict:
    met, _ = evaluate(model, camp.val_loader(), camp.device, 2, tta=False,
                      boundary_tol=None)
    return met.summary()


def _cosine(opt, total: int, warm_frac: float = 0.05):
    from torch.optim.lr_scheduler import LambdaLR
    warm = max(1, int(warm_frac * total))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, total - warm))))
    return LambdaLR(opt, f)


def _adamw(params, lr: float, wd: float = 0.01):
    import torch
    params = [p for p in params if p.requires_grad]
    return torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": wd},
                              {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}],
                             lr=lr)


# Decoded (image, mask) per scene, kept for the life of the process. A
# campaign retrains on the same scenes dozens of times, and each DataLoader
# worker would otherwise re-read every TIFF at startup (~3 s a run). Filled
# in the parent before the loader forks, so on Linux (fork) workers inherit
# it copy-on-write instead of re-reading.
_SCENES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _preload(ds: SceneDataset) -> None:
    for i, r in enumerate(ds.rows):
        key = r["tiff_path"] + "|" + r["mask_path"]
        if key not in _SCENES:
            _SCENES[key] = ds._read(i)
        ds._cache[i] = _SCENES[key]


class _StepStream:
    """Every sample one run will draw, as one flat indexable dataset.

    The epoch loop re-created DataLoader iterators every epoch, which with
    workers means re-forking them every ~7 steps, and without workers leaves
    the GPU idle while numpy crops. Flattening the whole run (epoch-seeded
    permutations, the same (seed, epoch, i) augmentation RNG SceneDataset
    uses) lets one worker pool stream it. Module-level so it pickles for
    forkserver (Python 3.14).
    """

    def __init__(self, ds: SceneDataset, batch: int, steps: int, seed: int):
        n = len(ds)
        drop_last = n > batch and n % batch == 1
        order: list[tuple[int, int]] = []
        epoch = 0
        while len(order) < steps * batch:
            perm = np.random.default_rng((seed, epoch)).permutation(n)
            if drop_last:
                perm = perm[: n - n % batch]
            order += [(epoch, int(i)) for i in perm]
            epoch += 1
        self.ds, self.order = ds, order[: steps * batch]

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, k: int):
        epoch, i = self.order[k]
        self.ds.epoch = epoch                 # worker-local copy; seeds the crop RNG
        return self.ds[i]


def train_steps(model, rows: list[dict], camp: Campaign, steps: int,
                lr: float = 6e-5, seed: int = 0) -> dict:
    """Full-network fine-tune for exactly `steps` optimizer steps."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    ds = SceneDataset(rows, MODALITY, camp.norm,
                      Geometry(crop=camp.crop), train=True, cache=True, seed=seed)
    _preload(ds)
    cuda = camp.device.startswith("cuda")
    workers = camp.workers if camp.workers is not None else (4 if cuda else 0)
    dl = DataLoader(_StepStream(ds, camp.batch, steps, seed), batch_size=camp.batch,
                    shuffle=False, num_workers=workers, pin_memory=cuda,
                    prefetch_factor=4 if workers else None)
    model.to(camp.device).train()
    opt = _adamw(model.module.parameters(), lr)
    sched = _cosine(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=camp.use_amp)
    loss_fn = camp.loss_fn()
    t0 = time.time()
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=camp.use_amp):
            loss = model.train_loss(x.to(camp.device, non_blocking=True),
                                    y.to(camp.device, non_blocking=True), loss_fn)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    return {"steps": steps, "train_s": round(time.time() - t0, 1),
            "last_loss": round(loss.item(), 4), "workers": workers}


def full_steps(n_rows: int, camp: Campaign, epochs: int = 40) -> int:
    """How many steps the training package's epoch recipe takes on n scenes."""
    per = n_rows // camp.batch
    if n_rows % camp.batch and not (n_rows > camp.batch and n_rows % camp.batch == 1):
        per += 1
    return epochs * max(1, per)


# ---------------------------------------------------------------------------
# decoder-only on cached encoder features
# ---------------------------------------------------------------------------

class FeatureCache:
    """Frozen-encoder features: `views` augmented crops per training scene,
    whole frames for val. Computed once per base model, reused by every batch.
    Stored fp16 (MiT-B0 at a 512 crop is ~1M values per view)."""

    def __init__(self, base, camp: Campaign, views: int = 8, seed: int = 0):
        self.base, self.camp, self.views, self.seed = base, camp, views, seed
        self.train: dict[str, list[tuple[list, "object"]]] = {}   # stem -> [(feats, y)]
        self.val: list[tuple[list, "object"]] | None = None
        self.encode_s = 0.0

    def _encode(self, x):
        import torch
        with torch.no_grad():
            hs = self.base.module.segformer(x, output_hidden_states=True).hidden_states
        return [h.half() for h in hs]

    def ensure_train(self, rows: list[dict]) -> None:
        todo = [r for r in rows if r["stem"] not in self.train]
        if not todo:
            return
        t0 = time.time()
        self.base.to(self.camp.device).eval()
        ds = SceneDataset(todo, MODALITY, self.camp.norm, Geometry(crop=self.camp.crop),
                          train=True, cache=False, seed=self.seed)
        for i, r in enumerate(todo):
            items = []
            for v in range(self.views):
                ds.set_epoch(v)                         # a different crop per view
                x, y = ds[i]
                items.append((self._encode(x[None].to(self.camp.device)), y))
            self.train[r["stem"]] = items
        self.encode_s += time.time() - t0

    def ensure_val(self) -> None:
        if self.val is not None:
            return
        t0 = time.time()
        self.base.to(self.camp.device).eval()
        self.val = [(self._encode(x.to(self.camp.device)), y[0])
                    for x, y in self.camp.val_loader()]
        self.encode_s += time.time() - t0


def _head_logits(head, feats, hw):
    import torch.nn.functional as F
    lo = head([f.float() for f in feats])
    return F.interpolate(lo, size=hw, mode="bilinear", align_corners=False)


def train_decoder(model, rows: list[dict], cache: FeatureCache, steps: int,
                  lr: float = 3e-4, seed: int = 0) -> dict:
    """Train only `model`'s decode head on cached features of `rows`."""
    import torch

    cache.ensure_train(rows)
    rng = np.random.default_rng(seed)
    pool = [(s, v) for s in (r["stem"] for r in rows) for v in range(cache.views)]
    head = model.module.decode_head.to(cache.camp.device).train()
    opt = _adamw(head.parameters(), lr)
    sched = _cosine(opt, steps)
    loss_fn = cache.camp.loss_fn()
    scaler = torch.amp.GradScaler("cuda", enabled=cache.camp.use_amp)
    b = cache.camp.batch
    t0 = time.time()
    for _ in range(steps):
        pick = [pool[i] for i in rng.choice(len(pool), size=b, replace=len(pool) < b)]
        feats = [torch.cat([cache.train[s][v][0][k] for s, v in pick]) for k in range(4)]
        y = torch.stack([cache.train[s][v][1] for s, v in pick]).to(cache.camp.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cache.camp.use_amp):
            loss = loss_fn(_head_logits(head, feats, y.shape[-2:]), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    return {"steps": steps, "train_s": round(time.time() - t0, 1),
            "last_loss": round(loss.item(), 4)}


def score_decoder(model, cache: FeatureCache) -> dict:
    """Val score of (frozen base encoder + model's head), from cached features."""
    import torch
    from training.metrics import Metrics

    cache.ensure_val()
    head = model.module.decode_head.eval()
    met = Metrics(num_classes=2)
    with torch.no_grad():
        for feats, y in cache.val:
            pred = _head_logits(head, feats, y.shape[-2:]).argmax(1).cpu().numpy()
            met.add(pred, y.numpy()[None])
    return met.summary()


# ---------------------------------------------------------------------------
# what the current model says about unlabeled candidates (the agent's state)
# ---------------------------------------------------------------------------

def candidate_stats(model, rows: list[dict], camp: Campaign,
                    auto_masks: list[np.ndarray | None] | None = None) -> list[dict]:
    """Per scene: the model's mean entropy, low-confidence fraction and
    predicted water fraction, plus IoU of its prediction against the auto mask
    when one is given. Uses only the image: the mask SceneDataset loads is read
    solely for where the /32 padding is (IGNORE), never for its labels."""
    import torch
    import torch.nn.functional as TF
    from torch.utils.data import DataLoader

    ds = SceneDataset(rows, MODALITY, camp.norm,
                      Geometry(eval_long_side=camp.eval_long_side), train=False, cache=False)
    model.to(camp.device).eval()
    out = []
    with torch.no_grad():
        for i, (x, _y) in enumerate(DataLoader(ds, batch_size=1)):
            p = TF.softmax(model.scores(x.to(camp.device)).float(), dim=1)[0]
            valid = (_y[0] != 255).to(camp.device)       # drop the /32 padding
            ent = -(p * torch.log(p + 1e-9)).sum(0)
            pred = p.argmax(0) == 1
            s = {"entropy": float(ent[valid].mean()),
                 "low_conf": float((p.max(0).values < 0.6)[valid].float().mean()),
                 "pred_water": float(pred[valid].float().mean())}
            a = auto_masks[i] if auto_masks else None
            if a is not None:
                h, w = a.shape
                pv = pred[:h, :w].cpu().numpy()
                av = a > 0
                u = np.logical_or(pv, av).sum()
                s["iou_model_auto"] = float(np.logical_and(pv, av).sum() / u) if u else 1.0
            out.append(s)
    return out
