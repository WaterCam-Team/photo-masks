"""One training run: `python -m training.train`.

    # 5-band SegFormer-B0 on the gold manifest
    uv run --project annotator python -m training.train \
        --manifest annotator/work/scenes.csv --modality fiveband \
        --arch segformer-b0 --out annotator/work/runs/fiveband-b0

    # cross-validated: fold 2 of 5 is val, the rest train
    ... --fold 2

    # the legacy img_dir/ann_dir tree the annotator's Train panel exports
    ... --data-root annotator/work/dataset

Normalisation statistics are measured from the *training* rows of this run and
written into the run directory, so val and test never leak into them and the
served model always carries the statistics it learned.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import data as D
from . import manifest as mf
from . import models
from . import modalities as M
from . import stats as ST
from .losses import SegLoss


@dataclass
class RunConfig:
    """Everything a run needs. Serialised to `config.json` in the run dir."""

    out: Path = Path("annotator/work/runs/run")
    arch: str = "segformer-b0"
    modality: str = "fiveband"
    init: str | None = None
    num_classes: int = 2

    # data / geometry
    crop: int = 512
    scale_range: tuple[float, float] = (0.5, 2.0)
    eval_long_side: int | None = None
    photometric: bool = True
    cache: bool = True

    # optimisation
    epochs: int = 40
    lr: float = 6e-5
    batch: int = 2
    accum: int = 1
    weight_decay: float = 0.01
    schedule: str = "cosine"
    warmup_frac: float = 0.05
    clip_grad: float = 1.0
    loss: str = "ce+lovasz"
    class_weights: bool = True
    norm_method: str = "meanstd"

    # runtime
    device: str = "auto"
    workers: int | None = None
    no_amp: bool = False
    seed: int = 0
    resume: bool = False

    # evaluation
    eval_every: int = 1
    patience: int = 0
    tta: bool = True
    tta_during_training: bool = False
    boundary_tol: int = 3
    log_every: int = 5

    # populated by the CLI
    train_rows: list[dict] = field(default_factory=list)
    val_rows: list[dict] = field(default_factory=list)
    test_rows: list[dict] = field(default_factory=list)
    stats: ST.BandStats | None = None

    def to_dict(self) -> dict:
        """Reproducible record — scene lists collapse to counts plus stems."""
        skip = {"train_rows", "val_rows", "test_rows", "stats"}
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        out["out"] = str(self.out)
        out["n_train"] = len(self.train_rows)
        out["n_val"] = len(self.val_rows)
        out["n_test"] = len(self.test_rows)
        out["train_stems"] = [r["stem"] for r in self.train_rows]
        out["val_stems"] = [r["stem"] for r in self.val_rows]
        out["test_stems"] = [r["stem"] for r in self.test_rows]
        if self.stats is not None:
            from dataclasses import asdict
            out["stats"] = asdict(self.stats)
        return out


def add_args(ap: argparse.ArgumentParser) -> None:
    d = RunConfig()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", type=Path, default=Path("annotator/work/scenes.csv"),
                     help="scene manifest from training.manifest")
    src.add_argument("--data-root", type=Path, default=None,
                     help="legacy export_dataset.py img_dir/ann_dir tree")
    ap.add_argument("--fold", type=int, default=None,
                    help="k-fold CV: this fold is val, the others train")
    ap.add_argument("--out", type=Path, default=d.out)
    ap.add_argument("--arch", default=d.arch, choices=models.ARCHS)
    ap.add_argument("--modality", default=d.modality, choices=list(M.MODALITIES))
    ap.add_argument("--init", default=None, help="HF id or checkpoint dir (default: "
                                                 "the arch's pretrained weights)")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--accum", type=int, default=d.accum,
                    help="gradient accumulation steps (raises effective batch)")
    ap.add_argument("--crop", type=int, default=d.crop)
    ap.add_argument("--scale-min", type=float, default=d.scale_range[0])
    ap.add_argument("--scale-max", type=float, default=d.scale_range[1])
    ap.add_argument("--eval-long-side", type=int, default=None,
                    help="resize eval frames to this long side (default: native)")
    ap.add_argument("--loss", default=d.loss, choices=SegLoss.SPECS)
    ap.add_argument("--norm-method", default=d.norm_method, choices=ST.METHODS)
    ap.add_argument("--stats", type=Path, default=None,
                    help="reuse a stats JSON instead of measuring this run's train split")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--no-photometric", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="do not hold scenes in RAM")
    ap.add_argument("--schedule", default=d.schedule, choices=("cosine", "linear"))
    ap.add_argument("--warmup-frac", type=float, default=d.warmup_frac)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--clip-grad", type=float, default=d.clip_grad)
    ap.add_argument("--device", default=d.device, help="auto | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--resume", action="store_true", help="continue from ckpt.pt in --out")
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--patience", type=int, default=d.patience,
                    help="stop after N evals without improvement (0 = never)")
    ap.add_argument("--no-tta", action="store_true", help="disable test-time augmentation")
    ap.add_argument("--tta-during-training", action="store_true")
    ap.add_argument("--boundary-tol", type=int, default=d.boundary_tol,
                    help="waterline slack in px for boundary F1 (0 = skip)")
    ap.add_argument("--log-every", type=int, default=d.log_every)


def config_from_args(a) -> RunConfig:
    cfg = RunConfig(
        out=a.out, arch=a.arch, modality=a.modality, init=a.init,
        crop=a.crop, scale_range=(a.scale_min, a.scale_max),
        eval_long_side=a.eval_long_side, photometric=not a.no_photometric,
        cache=not a.no_cache, epochs=a.epochs, lr=a.lr, batch=a.batch, accum=a.accum,
        weight_decay=a.weight_decay, schedule=a.schedule, warmup_frac=a.warmup_frac,
        clip_grad=a.clip_grad, loss=a.loss, class_weights=not a.no_class_weights,
        norm_method=a.norm_method, device=a.device, workers=a.workers,
        no_amp=a.no_amp, seed=a.seed, resume=a.resume, eval_every=a.eval_every,
        patience=a.patience, tta=not a.no_tta,
        tta_during_training=a.tta_during_training,
        boundary_tol=a.boundary_tol or None, log_every=a.log_every)

    if a.data_root:
        splits = D.legacy_rows(a.data_root)
        cfg.train_rows = splits.get("train", [])
        cfg.val_rows = splits.get("val", [])
        cfg.test_rows = splits.get("test", [])
        m = M.get(a.modality)
        if m.extra_glob:
            raise SystemExit(
                f"modality {a.modality!r} reads {m.extra_glob} from the scene "
                f"directory, which --data-root does not copy. Use --manifest.")
    elif a.fold is not None:
        cfg.train_rows = mf.load(a.manifest, fold=a.fold, fold_role="train")
        cfg.val_rows = mf.load(a.manifest, fold=a.fold, fold_role="val")
    else:
        cfg.train_rows = mf.load(a.manifest, split="train")
        cfg.val_rows = mf.load(a.manifest, split="val")
        cfg.test_rows = mf.load(a.manifest, split="test")

    if a.stats:
        cfg.stats = ST.BandStats.from_json(a.stats)
        if cfg.stats.modality != a.modality:
            raise SystemExit(f"--stats is for modality {cfg.stats.modality!r}, "
                             f"not {a.modality!r}")
    else:
        cfg.stats = ST.accumulate(cfg.train_rows, a.modality, method=a.norm_method)
    cfg.stats.method = a.norm_method
    return cfg


def main():
    from .engine import train

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    train(cfg)


if __name__ == "__main__":
    main()
