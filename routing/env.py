"""The labeling-route environment: one episode is one labeling campaign.

    reset()  -> obs          a fresh campaign: a random pool, a small free
                             starting set S0 (gold), SegFormer trained on S0
    step(a)  -> obs, r, done, info
                             a[i] in {AUTO, MANUAL, SKIP} for each of the K
                             candidates in the current batch

Each step routes one batch, the way the tool would: MANUAL adds the scene's
gold mask and charges the human time it took, AUTO adds the auto labeler's mask
(which may be wrong, and training on it can hurt) and charges a short review,
SKIP adds nothing and costs nothing. SegFormer is then **fully retrained** on
the labeled set (the benchmark showed only the full retrain ranks batches as
reliably as it agrees with itself) and scored on the fixed val set:

    r_t = alpha * (mIoU_t - mIoU_{t-1}) - lam * minutes_t

so the return telescopes to alpha * (final - initial mIoU) - lam * total
minutes. A batch is a step and the campaign is the episode, because a batch's value depends on what is
already labeled, and that sequential dependence is the whole reason for RL
rather than a bandit. With `steps=1` it reduces to a one-batch episode.

The episode is a simulation over already-labeled scenes: every scene has a gold
mask, and "MANUAL" reveals it. Episodes resample one small labeled set, so they
are not independent; report spread over seeds and pools, not one number.

Retrains are memoised on (labeled set + mask sources, seed, recipe): score in
retrain_cache.jsonl, fp16 weights in retrain_ckpt/ (~7.5 MB each for B0), so a
revisited state costs a load, not a retrain. Keyed by content, never by step.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import finetune as F

AUTO, MANUAL, SKIP = 0, 1, 2
ACTIONS = ("auto", "manual", "skip")

# label-free scene descriptors from rl_features.scene_features used as state
SCENE_FEATURES = ("ndwi_pos_frac", "ndwi_mean", "nir_dark_frac_25", "thermal_otsu_sep",
                  "edge_density", "gray_entropy", "nir_mean", "thermal_std")


@dataclass
class CostModel:
    """Seconds of annotator time per route.

    `manual` is the scene's own measured active time where the log has a
    timing-valid row for it (all current ones are interactive_clicks, which is
    how the gold masks were really made), else the median of those. `auto_s`
    is NOT measured yet: no timing-valid auto rows exist, so it is a
    placeholder until the planned labeling session calibrates it.
    Keep that in anything reported from this env.
    """

    manual_default_s: float
    auto_s: float = 5.0                   # glance and accept: placeholder, uncalibrated
    per_scene_manual: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_scenes(cls, scenes: dict, **kw) -> "CostModel":
        per = {k: v["manual"]["active_seconds"] for k, v in scenes.items() if v.get("manual")}
        return cls(manual_default_s=statistics.median(per.values()), per_scene_manual=per, **kw)

    def seconds(self, key: str, action: int) -> float:
        """What the route actually costs: charged by the environment."""
        if action == MANUAL:
            return self.per_scene_manual.get(key, self.manual_default_s)
        return self.auto_s if action == AUTO else 0.0

    def predicted(self, key: str, action: int) -> float:
        """What a campaign can know BEFORE labeling: the agent observes and
        decides on this, never on `seconds` (a scene's own labeling time is
        only known once someone has labeled it). For now the median; a
        regression on scene features would be the refinement."""
        if action == MANUAL:
            return self.manual_default_s
        return self.auto_s if action == AUTO else 0.0


@dataclass
class EnvConfig:
    data: Path                            # dir holding scenes.csv, env_scenes.json, scene dirs
    auto_backend: str = "sam2"
    s0_size: int = 4
    batch: int = 4                        # K candidates per step
    steps: int = 4                        # batches per campaign
    alpha: float = 100.0                  # reward per unit mIoU (100 => per mIoU point)
    lam: float = 1.0                      # reward cost per annotator minute
    epochs: int = 40                      # full-retrain recipe
    device: str = "cuda"
    amp: bool = True
    cache: Path | None = None             # retrain memo (jsonl); default <data>/retrain_cache.jsonl
    test_groups: tuple[str, ...] = ()     # val-split sessions held out of the reward, scored at the end
    exclude_groups: tuple[str, ...] = ()  # pool sessions never drawn (held-out-session training)
    ckpt_cap_gb: float = 5.0              # memo checkpoints beyond this are pruned, oldest first


class RoutingEnv:
    def __init__(self, cfg: EnvConfig, cost: CostModel | None = None):
        from training import manifest as mf
        from PIL import Image

        self.cfg = cfg
        d = Path(cfg.data)
        self.scenes = json.loads((d / "env_scenes.json").read_text())
        rows = {self._key(r): r for r in mf.load(d / "scenes.csv")}
        self.rows = rows
        self.pool_keys = sorted(k for k, v in self.scenes.items()
                                if v["split"] == "train" and v["group"] not in cfg.exclude_groups)
        all_pool = sorted(k for k, v in self.scenes.items() if v["split"] == "train")
        vals = [(k, v) for k, v in sorted(self.scenes.items()) if v["split"] == "val"]
        unknown = set(cfg.test_groups) - {v["group"] for _, v in vals}
        if unknown:
            raise ValueError(f"test_groups not in the val split: {sorted(unknown)}")
        val = [rows[k] for k, v in vals if v["group"] not in cfg.test_groups]
        test = [rows[k] for k, v in vals if v["group"] in cfg.test_groups]
        self.cost = cost or CostModel.from_scenes(self.scenes)
        self.auto = {}
        for k in all_pool:
            a = self.scenes[k]["auto"].get(cfg.auto_backend)
            if a is None:
                raise ValueError(f"{k}: no auto_{cfg.auto_backend}.png; run gen_auto_masks")
            self.auto[k] = d / k / a["file"]
        self._auto_arr = {k: np.array(Image.open(p).convert("L")) for k, p in self.auto.items()}
        self.val_keys = [self._key(r) for r in val]
        self.test_keys = [self._key(r) for r in test]
        # One set of band statistics for every campaign: from the whole pool's
        # images (label-free apart from the water fraction used for class
        # weights), so no retrain renormalises differently from another.
        # Statistics from the WHOLE pool, even when sessions are excluded, so a
        # held-out-session run normalises exactly like the runs it is compared to.
        self.camp = F.Campaign.from_rows([rows[k] for k in all_pool], val,
                                         device=cfg.device, amp=cfg.amp)
        self.test_camp = F.Campaign(val_rows=test, stats=self.camp.stats,
                                    device=cfg.device, amp=cfg.amp) if test else None
        self.cache_path = Path(cfg.cache or d / "retrain_cache.jsonl")
        self._memo = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                j = json.loads(line)
                self._memo[j["key"]] = j
        self.ckpt_dir = self.cache_path.parent / "retrain_ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    @staticmethod
    def _key(r: dict) -> str:
        return f'{r["group"]}/{r["stem"]}'.replace(" ", "_")

    # -- labeled set -> trained, scored model --------------------------------
    def _row(self, key: str, source: str) -> dict:
        r = dict(self.rows[key])
        if source == "auto":
            r["mask_path"] = str(self.auto[key])
        return r

    def _state_key(self, labeled) -> str:
        recipe = f"full{self.cfg.epochs}|amp{int(self.cfg.amp)}|{F.ARCH}"
        blob = json.dumps([sorted(labeled), self.seed, recipe, self.cfg.auto_backend]
                          + ([self.val_keys, self.test_keys] if self.test_keys else []))
        return hashlib.sha1(blob.encode()).hexdigest()

    def _retrain(self, labeled: list[tuple[str, str]]) -> dict:
        """Train on `labeled` and score, or restore both from the memo."""
        import torch
        key = self._state_key(labeled)
        ck = self.ckpt_dir / f"{key}.pt"
        if key in self._memo and ck.exists():
            try:
                sd = torch.load(ck)
            except (FileNotFoundError, RuntimeError, EOFError):   # pruned mid-read
                sd = None
            if sd is not None:
                m = F.new_model()
                m.module.load_state_dict({k: v.float() for k, v in sd.items()})
                self.model = m
                ck.touch()                        # recently used: prune it last
                return self._memo[key]
        rows = [self._row(k, s) for k, s in labeled]
        m = F.new_model()
        t = time.time()
        F.train_steps(m, rows, self.camp, F.full_steps(len(rows), self.camp, self.cfg.epochs),
                      seed=self.seed)
        s = F.score(m, self.camp)
        rec = {"key": key, "miou": s["miou"], "iou_water": s["iou_water"],
               "n": len(rows), "train_s": round(time.time() - t, 1)}
        if self.test_camp is not None:              # recorded, never part of the reward
            ts = F.score(m, self.test_camp)
            rec.update(test_miou=ts["miou"], test_iou_water=ts["iou_water"])
        tmp = ck.with_suffix(f".tmp{os.getpid()}")    # atomic: other runs share this dir
        torch.save({k: v.half() for k, v in m.module.state_dict().items()}, tmp)
        os.replace(tmp, ck)
        self._prune_ckpts()
        self._memo[key] = rec
        with self.cache_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.model = m
        return rec

    def _prune_ckpts(self) -> None:
        """Keep retrain_ckpt/ under the cap (the 3080 is shared). Scores stay in
        retrain_cache.jsonl; a pruned state is simply retrained if revisited."""
        cap = self.cfg.ckpt_cap_gb * 1e9
        files = []
        for f in self.ckpt_dir.glob("*.pt"):
            try:
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f))
            except FileNotFoundError:             # pruned by a concurrent run
                pass
        total = sum(sz for _, sz, _ in files)
        for _, sz, f in sorted(files):
            if total <= cap:
                break
            f.unlink(missing_ok=True)
            total -= sz

    # -- observation -----------------------------------------------------------
    def _obs(self) -> dict:
        cand = self.batch_keys
        stats = F.candidate_stats(self.model, [self.rows[k] for k in cand], self.camp,
                                  [self._auto_arr[k] for k in cand])
        # Session redundancy, label-free (the capture session is known without
        # labels): near-duplicate frames from an already-covered session were
        # measured to pull the model away from held-out sessions.
        sess = lambda k: self.scenes[k]["group"]
        lab = [sess(k) for k, _ in self.labeled]
        feats = []
        for k, s in zip(cand, stats):
            f = self.scenes[k]["features"]
            same = sum(g == sess(k) for g in lab)
            feats.append([f[n] for n in SCENE_FEATURES] +
                         [s["entropy"], s["low_conf"], s["pred_water"], s["iou_model_auto"],
                          float((self._auto_arr[k] > 0).mean()),
                          self.cost.predicted(k, MANUAL) / 60.0,
                          same / max(1, len(lab)), float(same),
                          float(sum(sess(o) == sess(k) for o in cand) - 1)])
        return {"candidates": np.array(feats, dtype=np.float32), "keys": list(cand),
                "global": np.array([self.t / self.cfg.steps, len(self.labeled),
                                    self.miou, self.minutes], dtype=np.float32)}

    # -- API -------------------------------------------------------------------
    def reset(self, seed: int = 0) -> dict:
        self.seed = seed
        rng = np.random.default_rng(seed)
        need = self.cfg.s0_size + self.cfg.batch * self.cfg.steps
        if need > len(self.pool_keys):
            raise ValueError(f"campaign needs {need} scenes, pool has {len(self.pool_keys)}")
        order = [self.pool_keys[i] for i in rng.permutation(len(self.pool_keys))[:need]]
        s0, rest = order[:self.cfg.s0_size], order[self.cfg.s0_size:]
        self.queue = [rest[i:i + self.cfg.batch] for i in range(0, len(rest), self.cfg.batch)]
        self.labeled = [(k, "gold") for k in s0]
        self.t, self.minutes, self.model = 0, 0.0, None
        rec0 = self._retrain(self.labeled)
        self.miou = self.miou0 = rec0["miou"]
        self.test_miou = self.test_miou0 = rec0.get("test_miou")
        self.batch_keys = self.queue[0]
        self.history = []
        return self._obs()

    def step(self, actions) -> tuple[dict | None, float, bool, dict]:
        actions = [int(a) for a in actions]
        if len(actions) != len(self.batch_keys):
            raise ValueError(f"need {len(self.batch_keys)} actions, got {len(actions)}")
        secs = 0.0
        for k, a in zip(self.batch_keys, actions):
            secs += self.cost.seconds(k, a)
            if a == MANUAL:
                self.labeled.append((k, "gold"))
            elif a == AUTO:
                self.labeled.append((k, "auto"))
        minutes = secs / 60.0
        rec = self._retrain(self.labeled)
        gain = rec["miou"] - self.miou
        r = self.cfg.alpha * gain - self.cfg.lam * minutes
        info = {"t": self.t, "keys": self.batch_keys, "actions": [ACTIONS[a] for a in actions],
                "minutes": round(minutes, 3), "miou": rec["miou"], "gain": gain,
                "reward": r, "auto_iou_vs_gold": [
                    self.scenes[k]["auto"][self.cfg.auto_backend]["iou_vs_gold"]
                    for k in self.batch_keys]}
        self.history.append(info)
        self.miou, self.minutes, self.t = rec["miou"], self.minutes + minutes, self.t + 1
        self.test_miou = rec.get("test_miou")
        if self.test_miou is not None:
            info["test_miou"] = self.test_miou
        done = self.t >= self.cfg.steps
        if done:
            return None, r, True, info
        self.batch_keys = self.queue[self.t]
        return self._obs(), r, False, info
