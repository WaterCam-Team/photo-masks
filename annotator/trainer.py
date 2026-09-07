#!/usr/bin/env python
"""Fine-tune a water-segmentation model for the annotator's Train panel.

A thin adapter over the `training/` package, which holds the real pipeline
(modalities, band statistics, augmentation, losses, metrics, resume, TTA). This
file exists to keep one CLI and one JSONL protocol stable for `server.py`:

    {"event":"start", "device":..., "n_train":..., "n_val":...}
    {"event":"step",  "step":N, "epoch":E, "loss":..., "lr":...}
    {"event":"eval",  "step":N, "miou":..., "iou_water":..., "best":bool}
    {"event":"done",  "best_miou":..., "ckpt":".../best.pt", "hf":".../best_hf"}
    {"event":"error", "msg":...}

Outputs in <out>/: last-resumable ckpt.pt, best.pt, best_hf/ (with norm.json),
config.json, metrics.json.

    python trainer.py --data-root work/dataset --out work/runs/r1 --epochs 40
    python trainer.py --export-onnx work/runs/r1/best_hf --onnx-out weights/segformer_5band.onnx

Two behaviours changed when the pipeline moved into `training/`, both
deliberate — see `training/data.py` and `training/stats.py` for why:

  * `--img-size` is now the training **crop** taken from a randomly rescaled
    frame, not a squash of the whole 4:3 frame into a square. Validation runs
    on whole frames at native resolution unless `--eval-long-side` is given, so
    the reported mIoU is comparable with a paper's.
  * Inputs are normalised with band statistics measured over the training
    split and written to `best_hf/norm.json`, instead of per-image min-max.
    The `segformer` backend reads that file, so a served model always uses the
    statistics it was trained with.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:                 # training/ lives beside annotator/
    sys.path.insert(0, str(HERE.parent))

from training import data as D                       # noqa: E402
from training import modalities as M                 # noqa: E402
from training import models                          # noqa: E402
from training import stats as ST                     # noqa: E402
from training.engine import emit, train              # noqa: E402
from training.train import RunConfig                 # noqa: E402


def export_pi(hf_dir: str, out_dir: str, img_size: int = 512, int8: bool = True,
              calib: str | None = None, n_calib: int = 32) -> None:
    """Build the camera-node bundle: fp32 + INT8 + deploy.json, ready to copy.

    Calibration uses real captures — by default the gold dataset the Train
    panel exported, which is the same distribution the node will see.
    """
    from training import deploy

    hf = Path(hf_dir)
    norm_src = hf / "norm.json"
    if not norm_src.exists():
        emit(event="error", msg=f"no norm.json in {hf} — this checkpoint predates "
                                f"band statistics; retrain, or export with "
                                f"--export-onnx for the legacy contract")
        sys.exit(1)
    stats = ST.BandStats.from_json(norm_src)

    rows: list[dict] = []
    src = Path(calib) if calib else (HERE / "work" / "dataset")
    if src.is_file():                                   # a manifest
        from training import manifest as mf
        rows = mf.load(src, split="train")
    elif src.exists():                                  # an img_dir/ann_dir tree
        rows = D.legacy_rows(src).get("train", [])
    if not rows:
        emit(event="warn", msg=f"no calibration scenes under {src} — INT8 ranges "
                               f"will come from whatever the quantizer guesses")
    rows = rows[:n_calib]

    arch = "segformer-b0"
    info = deploy.build(hf, Path(out_dir), stats, rows, arch=arch,
                        size=img_size, int8=int8, emit=emit)
    emit(event="deploy", **info)


def export_onnx(hf_dir: str, onnx_out: str, img_size: int = 512,
                no_embed_norm: bool = False) -> None:
    """Export a trained HF checkpoint for the camera nodes.

    `norm.json` is copied next to the .onnx: the deployed runtime needs the
    same band statistics the model trained with, and an .onnx file cannot
    carry them itself.
    """
    import shutil

    hf = Path(hf_dir)
    norm_src = hf / "norm.json"
    channels = 5
    if norm_src.exists():
        channels = ST.BandStats.from_json(norm_src).channels
    else:
        emit(event="warn", msg=f"no norm.json in {hf} — assuming a 5-band model "
                               f"normalised per-image (legacy min-max)")
    arch = "segformer-b0"
    m = models.load(arch, str(hf), in_channels=channels)
    stats = ST.BandStats.from_json(norm_src) if norm_src.exists() else None
    try:
        out = m.export_onnx(Path(onnx_out), size=img_size, stats=stats,
                            embed_norm=not no_embed_norm)
    except Exception as e:                          # noqa: BLE001
        msg = str(e)
        low = msg.lower()
        if "onnx is not installed" in low or "no module named 'onnx'" in low \
                or "requires onnx" in low:
            emit(event="error", msg="ONNX export needs the 'onnx' package: "
                 "`uv sync --group export` (the fine-tuned model still works in "
                 "the annotator via its HF checkpoint dir).")
            sys.exit(2)
        raise
    if norm_src.exists():
        shutil.copy(norm_src, out.parent / (out.stem + ".norm.json"))
    emit(event="onnx", path=str(out), bytes=out.stat().st_size,
         norm=str(out.parent / (out.stem + ".norm.json")) if norm_src.exists() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="dir with img_dir/{train,val} and ann_dir/{train,val}")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="scene manifest from training.manifest (preferred: it can "
                         "serve every modality, not just band subsets of the TIFF)")
    ap.add_argument("--out", help="run output dir")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--batch", type=int, default=2,
                    help="raise this on a GPU (8-16); 2 is a safe CPU default")
    ap.add_argument("--img-size", type=int, default=512, help="training crop, in px")
    ap.add_argument("--eval-long-side", type=int, default=None,
                    help="downscale val frames to this long side (default: native)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default: 2 on GPU, 0 on CPU)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable fp16 mixed precision (CUDA only; on by default there)")
    ap.add_argument("--init", default=None,
                    help="HF model id or local dir to initialise from")
    ap.add_argument("--arch", default="segformer-b0", choices=models.ARCHS)
    ap.add_argument("--modality", default="fiveband", choices=list(M.MODALITIES))
    ap.add_argument("--loss", default="ce+lovasz")
    ap.add_argument("--norm-method", default="meanstd", choices=ST.METHODS)
    ap.add_argument("--resume", action="store_true", help="continue from ckpt.pt in --out")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1, help="epochs between val")
    ap.add_argument("--export-onnx", metavar="HF_DIR",
                    help="skip training: export this HF model dir to ONNX")
    ap.add_argument("--export-pi", metavar="HF_DIR",
                    help="skip training: build the camera-node bundle (fp32 + "
                         "INT8 + deploy.json) from this HF model dir")
    ap.add_argument("--out-dir", default=None, help="where --export-pi writes")
    ap.add_argument("--no-int8", action="store_true",
                    help="--export-pi: fp32 only, skip quantization")
    ap.add_argument("--calib", default=None,
                    help="--export-pi: calibration scenes — a training.manifest CSV "
                         "or an img_dir/ann_dir tree (default: work/dataset)")
    ap.add_argument("--n-calib", type=int, default=32,
                    help="--export-pi: how many scenes to calibrate on")
    ap.add_argument("--onnx-out", default=None)
    ap.add_argument("--no-embed-norm", action="store_true",
                    help="export a graph that expects pre-normalised input "
                         "(the legacy contract) instead of normalising inside it")
    args = ap.parse_args()

    try:
        if args.export_pi:
            export_pi(args.export_pi, args.out_dir or str(HERE / "weights"),
                      args.img_size, not args.no_int8, args.calib, args.n_calib)
            return
        if args.export_onnx:
            export_onnx(args.export_onnx, args.onnx_out or "segformer_5band.onnx",
                        args.img_size, args.no_embed_norm)
            return
        if not (args.data_root or args.manifest) or not args.out:
            ap.error("--out plus one of --data-root / --manifest is required")

        cfg = RunConfig(
            out=Path(args.out), arch=args.arch, modality=args.modality,
            init=args.init, crop=args.img_size, eval_long_side=args.eval_long_side,
            epochs=args.epochs, lr=args.lr, batch=args.batch, loss=args.loss,
            norm_method=args.norm_method, device=args.device, workers=args.workers,
            no_amp=args.no_amp, resume=args.resume, patience=args.patience,
            log_every=args.log_every, eval_every=args.eval_every)

        if args.manifest:
            from training import manifest as mf
            cfg.train_rows = mf.load(args.manifest, split="train")
            cfg.val_rows = mf.load(args.manifest, split="val")
            cfg.test_rows = mf.load(args.manifest, split="test")
        else:
            splits = D.legacy_rows(Path(args.data_root))
            cfg.train_rows = splits.get("train", [])
            cfg.val_rows = splits.get("val", [])
            cfg.test_rows = splits.get("test", [])
            if M.get(args.modality).extra_glob:
                emit(event="error", msg=f"modality {args.modality!r} needs files the "
                                        f"img_dir/ann_dir export does not copy; "
                                        f"use --manifest")
                sys.exit(1)
        if not cfg.train_rows:
            emit(event="error", msg=f"no training images under "
                                    f"{args.data_root or args.manifest}")
            sys.exit(1)

        cfg.stats = ST.accumulate(cfg.train_rows, args.modality, method=args.norm_method)
        train(cfg)
    except SystemExit:
        raise
    except Exception as e:                          # noqa: BLE001
        import traceback
        emit(event="error", msg=str(e), trace=traceback.format_exc()[-1500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
