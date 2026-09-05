"""Training and evaluation loop.

Emits the same newline-delimited JSON the annotator's Train panel already
parses, so the in-tool loop and a standalone research run report progress
identically:

    {"event":"start", "device":..., "n_train":..., "n_val":...}
    {"event":"step",  "step":N, "epoch":E, "loss":..., "lr":...}
    {"event":"eval",  "step":N, "miou":..., "iou_water":..., "best":bool}
    {"event":"done",  "best_miou":..., "ckpt":..., "hf":...}
    {"event":"error", "msg":...}

Run directory:

    ckpt.pt        model + optimizer + scheduler + epoch (resumable)
    best.pt        model weights at the best val mIoU
    best_hf/       HF checkpoint dir, plus norm.json  <- inference reads this
    norm.json      the frozen band statistics used for training
    config.json    everything needed to reproduce the run
    metrics.json   full per-eval history

`norm.json` living inside `best_hf/` is the mechanism that keeps training and
inference in lockstep: the model and the statistics it was normalised with are
one artifact, so a checkpoint cannot be served under different statistics than
it learned. Checkpoints without it are read as legacy per-image min-max.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import models
from .data import IGNORE, Geometry, SceneDataset
from .losses import SegLoss
from .metrics import Metrics
from .stats import BandStats, Normalizer


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def resolve_device(spec: str) -> str:
    """auto | cpu | cuda[:N] | mps | xpu, plus aliases amd/rocm/hip/nvidia/gpu
    -> cuda (AMD ROCm PyTorch reports as 'cuda' too), intel -> xpu."""
    import torch
    aliases = {"amd": "cuda", "rocm": "cuda", "hip": "cuda", "gpu": "cuda",
               "nvidia": "cuda", "intel": "xpu"}
    spec = aliases.get((spec or "auto").lower(), (spec or "auto").lower())
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str | None:
    import torch
    try:
        if device.startswith("cuda"):
            hip = getattr(torch.version, "hip", None)
            return torch.cuda.get_device_name(0) + (f" (ROCm {hip})" if hip else "")
        if device == "xpu":
            return torch.xpu.get_device_name(0)
    except Exception:                                # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _tta_scores(model, x, scales, flip: bool):
    """Average softmax over scales and an optional horizontal flip."""
    import torch
    import torch.nn.functional as F

    H, W = x.shape[-2:]
    acc = None
    for s in scales:
        xi = x
        if s != 1.0:
            h = int(round(H * s / 32)) * 32 or 32
            w = int(round(W * s / 32)) * 32 or 32
            xi = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        views = [xi, torch.flip(xi, dims=[-1])] if flip else [xi]
        for vi, v in enumerate(views):
            sc = model.scores(v)
            if vi == 1:
                sc = torch.flip(sc, dims=[-1])
            sc = F.softmax(sc, dim=1)
            if sc.shape[-2:] != (H, W):
                sc = F.interpolate(sc, size=(H, W), mode="bilinear", align_corners=False)
            acc = sc if acc is None else acc + sc
    return acc


def evaluate(model, loader, device: str, num_classes: int = 2,
             tta: bool = False, boundary_tol: int | None = 3,
             per_scene: bool = False) -> tuple[Metrics, list[dict]]:
    import torch

    model.eval()
    met = Metrics(num_classes=num_classes)
    scales = (0.75, 1.0, 1.25) if tta else (1.0,)
    rows: list[dict] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            sc = _tta_scores(model, x, scales, flip=tta)
            pred = sc.argmax(1).cpu().numpy()
            true = y.numpy()
            met.add(pred, true, boundary_tol=boundary_tol)
            if per_scene:
                one = Metrics(num_classes=num_classes)
                one.add(pred, true, boundary_tol=boundary_tol)
                rows.append({"index": i, **one.summary()})
    return met, rows


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def build_loaders(cfg, stats: BandStats):
    from torch.utils.data import DataLoader

    norm = Normalizer(stats, cfg.norm_method)
    geom = Geometry(crop=cfg.crop, scale_range=tuple(cfg.scale_range),
                    eval_long_side=cfg.eval_long_side)
    tr = SceneDataset(cfg.train_rows, cfg.modality, norm, geom, train=True,
                      photometric=cfg.photometric, cache=cfg.cache, seed=cfg.seed)
    va = SceneDataset(cfg.val_rows, cfg.modality, norm, geom, train=False,
                      cache=cfg.cache) if cfg.val_rows else None

    is_cuda = cfg.device.startswith("cuda")
    workers = cfg.workers if cfg.workers is not None else (2 if cfg.device != "cpu" else 0)
    # SegFormer's decode head normalises over the batch, so a trailing batch of
    # one would fail; drop it only when there is more than one batch to keep.
    n = len(tr)
    drop_last = n > cfg.batch and n % cfg.batch == 1
    dltr = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=workers,
                      pin_memory=is_cuda, drop_last=drop_last,
                      persistent_workers=workers > 0)
    # eval runs whole frames, whose sizes may differ between capture sessions
    dlva = DataLoader(va, batch_size=1, num_workers=workers, pin_memory=is_cuda) \
        if va is not None else None
    return tr, dltr, dlva, workers


def train(cfg) -> dict:
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.device = resolve_device(cfg.device)
    is_cuda = cfg.device.startswith("cuda")
    amp = is_cuda and not cfg.no_amp

    stats = cfg.stats
    if stats.water_frac <= 0.0:
        emit(event="error", msg="every training mask is empty (0% water). Masks are "
                                "expected as 0/255 or 0/1 PNGs on the TIFF grid — a "
                                "mismatched threshold or the wrong ann_dir reads as "
                                "all background.")
        raise SystemExit(1)
    if stats.water_frac >= 1.0:
        emit(event="error", msg="every training mask is fully water (100%) — check "
                                "the masks before training on them.")
        raise SystemExit(1)
    stats.to_json(out / "norm.json")
    in_ch = stats.channels

    tr, dltr, dlva, workers = build_loaders(cfg, stats)
    if len(tr) == 0:
        emit(event="error", msg="no training scenes — check the manifest/split")
        raise SystemExit(1)
    if len(tr) == 1:
        emit(event="error", msg="only 1 training scene: batch normalisation in the "
                                "decode head needs at least 2. Label more scenes.")
        raise SystemExit(1)

    model = models.build(cfg.arch, in_channels=in_ch, num_classes=cfg.num_classes,
                         init=cfg.init, emit=emit).to(cfg.device)
    weights = stats.class_weights() if cfg.class_weights else None
    loss_fn = SegLoss(cfg.loss, weights)
    if not model.supports_pixel_loss and cfg.loss != "ce":
        emit(event="warn", msg=f"{cfg.arch} trains with its own mask-matching loss; "
                               f"--loss {cfg.loss} does not apply to it")

    decay = [p for n, p in model.module.named_parameters()
             if p.requires_grad and p.ndim > 1]
    no_decay = [p for n, p in model.module.named_parameters()
                if p.requires_grad and p.ndim <= 1]      # norms and biases
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    steps_per_epoch = max(1, len(dltr) // max(1, cfg.accum))
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_frac * total))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        if cfg.schedule == "cosine":
            return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))
        return max(0.0, 1.0 - p)

    sched = LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, step, best, history = 0, 0, -1.0, []
    ck = out / "ckpt.pt"
    if cfg.resume and ck.exists():
        state = torch.load(ck, map_location=cfg.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        step = state.get("step", 0)
        best = state.get("best", -1.0)
        history = state.get("history", [])
        resumed_stale = int(state.get("stale", 0))
        emit(event="info", msg=f"resumed from {ck} at epoch {start_epoch} "
                               f"(best mIoU {best:.4f})")

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    emit(event="start", device=cfg.device, gpu=device_label(cfg.device), amp=amp,
         workers=workers, arch=cfg.arch, modality=cfg.modality, in_channels=in_ch,
         params_m=round(model.n_params / 1e6, 2), n_train=len(tr),
         n_val=(len(dlva.dataset) if dlva else 0), epochs=cfg.epochs, crop=cfg.crop,
         lr=cfg.lr, batch=cfg.batch, accum=cfg.accum, loss=cfg.loss,
         norm=cfg.norm_method, class_weights=weights, init=model.init,
         pixel_loss_applies=model.supports_pixel_loss,
         resumed_from_epoch=start_epoch or None)

    t0 = time.time()
    stale = locals().get("resumed_stale", 0)
    for ep in range(start_epoch, cfg.epochs):
        model.train()
        tr.set_epoch(ep)
        opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(dltr):
            x = x.to(cfg.device, non_blocking=is_cuda)
            y = y.to(cfg.device, non_blocking=is_cuda)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model.train_loss(x, y, loss_fn) / max(1, cfg.accum)
            scaler.scale(loss).backward()
            if (bi + 1) % max(1, cfg.accum) == 0:
                if cfg.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step == 1 or step % cfg.log_every == 0:
                    emit(event="step", step=step, epoch=ep,
                         loss=round(float(loss) * max(1, cfg.accum), 4),
                         lr=round(sched.get_last_lr()[0], 8),
                         secs=round(time.time() - t0, 1))

        stop = False
        if dlva and (ep + 1) % cfg.eval_every == 0:
            met, _ = evaluate(model, dlva, cfg.device, cfg.num_classes,
                              tta=cfg.tta_during_training, boundary_tol=cfg.boundary_tol)
            s = met.summary()
            is_best = s["miou"] > best
            history.append({"epoch": ep, "step": step, **s})
            emit(event="eval", step=step, epoch=ep, best=is_best, **s)
            if is_best:
                best = s["miou"]
                stale = 0
                torch.save(model.state_dict(), out / "best.pt")
                model.save(out / "best_hf")
                stats.to_json(out / "best_hf" / "norm.json")   # travels with the model
            else:
                stale += 1
                if cfg.patience and stale >= cfg.patience:
                    emit(event="info", msg=f"early stop: {stale} evals without "
                                           f"improvement (patience {cfg.patience})")
                    stop = True

        # written after the eval: a checkpoint saved before it would carry a
        # `best` one eval out of date, and resuming from that overwrites the
        # real best.pt with a worse model.
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "step": step, "best": best, "history": history,
                    "stale": stale, "arch": cfg.arch, "modality": cfg.modality,
                    "in_channels": in_ch, "stats": asdict(stats),
                    "norm_method": cfg.norm_method}, ck)
        if stop:
            break

    if not (out / "best.pt").exists():                 # no val set: last is best
        torch.save(model.state_dict(), out / "best.pt")
        model.save(out / "best_hf")
        stats.to_json(out / "best_hf" / "norm.json")

    final = {"best_miou": round(best, 4) if best >= 0 else None,
             "history": history, "config": cfg.to_dict(),
             "minutes": round((time.time() - t0) / 60, 2)}

    if cfg.test_rows:                                  # final held-out number, once
        from torch.utils.data import DataLoader
        te = SceneDataset(cfg.test_rows, cfg.modality, Normalizer(stats, cfg.norm_method),
                          Geometry(eval_long_side=cfg.eval_long_side), train=False,
                          cache=cfg.cache)
        if (out / "best.pt").exists():
            model.load_state_dict(torch.load(out / "best.pt", map_location=cfg.device))
        met, per = evaluate(model, DataLoader(te, batch_size=1), cfg.device,
                            cfg.num_classes, tta=cfg.tta, boundary_tol=cfg.boundary_tol,
                            per_scene=True)
        final["test"] = met.summary()
        final["test_per_scene"] = [{**r, "stem": cfg.test_rows[r["index"]]["stem"]}
                                   for r in per]
        emit(event="test", **met.summary())

    (out / "metrics.json").write_text(json.dumps(final, indent=2, default=str))
    emit(event="done", best_miou=final["best_miou"], ckpt=str(out / "best.pt"),
         hf=str(out / "best_hf"), minutes=final["minutes"])
    return final
