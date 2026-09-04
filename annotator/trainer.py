#!/usr/bin/env python
"""Fine-tune a 5-band SegFormer-B0 on an exported img_dir/ann_dir dataset.

Runs in a Python with transformers + torch (`uv sync --group train`). Streams
JSONL progress on stdout so the annotator's Train panel can render it:

    {"event":"start", "device":..., "n_train":..., "n_val":...}
    {"event":"step",  "step":N, "epoch":E, "loss":..., "lr":...}
    {"event":"eval",  "step":N, "miou":..., "iou_water":..., "best":bool}
    {"event":"done",  "best_miou":..., "ckpt":".../best.pt", "hf":".../best_hf"}
    {"event":"error", "msg":...}

Outputs in <out>/:
    last.pt / best.pt     state_dicts
    best_hf/              HF model dir (config.json + safetensors) for ONNX export
    metrics.json          full history

    python trainer.py --data-root work/dataset --out work/runs/r1 --epochs 40
    python trainer.py --export-onnx work/runs/r1/best_hf --onnx-out weights/segformer_5band.onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BANDS = 5


def emit(**kw):
    print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_tiff(path: Path, size: int) -> np.ndarray:
    import cv2
    import rasterio
    with rasterio.open(path) as s:
        a = s.read().astype(np.float32)[:BANDS]
    if a.shape[0] < BANDS:                       # pad missing bands with zeros
        a = np.concatenate([a, np.zeros((BANDS - a.shape[0], *a.shape[1:]), np.float32)])
    a = np.stack([cv2.resize(a[i], (size, size), interpolation=cv2.INTER_AREA)
                  for i in range(BANDS)])
    for i in range(BANDS):                       # per-band min-max -> [0,1]
        lo, hi = float(a[i].min()), float(a[i].max())
        a[i] = (a[i] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(a, 0.0, 1.0)


def load_mask(path: Path, size: int) -> np.ndarray:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.int64)


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:                                   # noqa: BLE001 - torch imported later
    _TorchDataset = object


class _SegDS(_TorchDataset):
    """Module-level so DataLoader workers can pickle it (py3.14 forkserver)."""

    def __init__(self, imgs, ann_dir: Path, size: int, aug: bool):
        self.imgs = imgs
        self.ann_dir = ann_dir
        self.size = size
        self.aug = aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        import torch
        ip = self.imgs[i]
        x = load_tiff(ip, self.size)
        y = load_mask(self.ann_dir / f"{ip.stem}.png", self.size)
        if self.aug and np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def make_dataset(root: str, split: str, size: int, aug: bool):
    img_dir = Path(root) / "img_dir" / split
    ann_dir = Path(root) / "ann_dir" / split
    imgs = sorted(p for p in img_dir.glob("*.tif*"))
    return _SegDS(imgs, ann_dir, size, aug), len(imgs)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(init: str, num_labels: int = 2):
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            init, num_labels=num_labels, num_channels=BANDS,
            ignore_mismatched_sizes=True)
        emit(event="info", msg=f"initialised encoder from {init} (5-band patch-embed re-init)")
        # warm-start the new 5-ch stem conv: RGB->0,1,2 ; mean(RGB)->3,4
        try:
            src = SegformerForSemanticSegmentation.from_pretrained(init)
            with torch.no_grad():
                new_w = model.segformer.encoder.patch_embeddings[0].proj.weight  # (C,5,7,7)
                old_w = src.segformer.encoder.patch_embeddings[0].proj.weight    # (C,3,7,7)
                new_w[:, :3] = old_w
                new_w[:, 3:] = old_w.mean(dim=1, keepdim=True)
            del src
            emit(event="info", msg="warm-started 5-band stem from RGB weights")
        except Exception as e:                   # noqa: BLE001
            emit(event="warn", msg=f"stem warm-start skipped: {e}")
    except Exception as e:                       # noqa: BLE001
        emit(event="warn", msg=f"pretrained init failed ({e}); random init")
        cfg = SegformerConfig(num_labels=num_labels, num_channels=BANDS)
        model = SegformerForSemanticSegmentation(cfg)
    return model


def evaluate_miou(model, dl, device, num_labels=2):
    import torch
    import torch.nn.functional as F
    model.eval()
    cm = np.zeros((num_labels, num_labels), np.int64)
    amp = device.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for x, y in dl:
            logits = model(pixel_values=x.to(device)).logits
            logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
            p = logits.argmax(1).cpu().numpy().ravel()
            t = y.numpy().ravel()
            k = (t >= 0) & (t < num_labels)
            cm += np.bincount(num_labels * t[k] + p[k],
                              minlength=num_labels ** 2).reshape(num_labels, num_labels)
    denom = cm.sum(1) + cm.sum(0) - np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom == 0, np.nan, np.diag(cm) / denom)   # absent class -> nan, not 0
    miou = float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0
    return miou, [None if np.isnan(v) else float(v) for v in iou]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def resolve_device(spec: str) -> str:
    """auto | cpu | cuda[:N] | mps | xpu, plus aliases amd/rocm/hip/nvidia/gpu
    -> cuda (AMD ROCm PyTorch uses the 'cuda' string too), intel -> xpu."""
    import torch
    aliases = {"amd": "cuda", "rocm": "cuda", "hip": "cuda", "gpu": "cuda",
               "nvidia": "cuda", "intel": "xpu"}
    spec = aliases.get((spec or "auto").lower(), (spec or "auto").lower())
    if spec != "auto":
        return spec
    if torch.cuda.is_available():                    # NVIDIA CUDA or AMD ROCm
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(0)
    is_cuda = device.startswith("cuda")              # True on AMD ROCm too
    amp = is_cuda and not args.no_amp                 # fp16 autocast: CUDA / ROCm
    workers = args.workers if args.workers is not None else (2 if device != "cpu" else 0)
    gpu_name = None
    try:
        if is_cuda:
            hip = getattr(torch.version, "hip", None)
            gpu_name = torch.cuda.get_device_name(0) + (f" (ROCm {hip})" if hip else "")
        elif device == "xpu":
            gpu_name = torch.xpu.get_device_name(0)
    except Exception:                                # noqa: BLE001
        pass

    tr, n_tr = make_dataset(args.data_root, "train", args.img_size, aug=True)
    va, n_va = make_dataset(args.data_root, "val", args.img_size, aug=False)
    if n_tr == 0:
        emit(event="error", msg=f"no training images under {args.data_root}/img_dir/train")
        sys.exit(1)

    emit(event="start", device=device, gpu=gpu_name, amp=amp, workers=workers,
         n_train=n_tr, n_val=n_va, epochs=args.epochs, img_size=args.img_size,
         lr=args.lr, batch=args.batch, init=args.init)

    # SegFormer's decode-head BatchNorm can't take a single sample, so drop a
    # trailing batch of size 1 — but only when there's more than one batch to
    # begin with. With a single training image there is nothing to drop and
    # nothing to normalise over, which is a config error, not a runtime one.
    if n_tr == 1:
        emit(event="error", msg="only 1 training image — SegFormer's decode-head "
                                "BatchNorm needs at least 2. Label more scenes "
                                "(or lower --val-split) and export again.")
        sys.exit(1)
    drop_last = n_tr > args.batch and n_tr % args.batch == 1
    dltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=workers,
                      pin_memory=is_cuda, drop_last=drop_last,
                      persistent_workers=workers > 0)
    dlva = (DataLoader(va, batch_size=args.batch, num_workers=workers, pin_memory=is_cuda)
            if n_va else None)

    model = build_model(args.init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * max(1, len(dltr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step = 0
    best = -1.0
    history = []
    for ep in range(args.epochs):
        model.train()
        for x, y in dltr:
            x = x.to(device, non_blocking=is_cuda)
            y = y.to(device, non_blocking=is_cuda)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(pixel_values=x).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                       align_corners=False)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                emit(event="step", step=step, epoch=ep, loss=round(loss.item(), 4),
                     lr=round(sched.get_last_lr()[0], 8))

        torch.save(model.state_dict(), out / "last.pt")
        if dlva and (ep + 1) % args.eval_every == 0:
            m, iou = evaluate_miou(model, dlva, device)
            is_best = m > best
            best = max(best, m)
            history.append({"step": step, "epoch": ep, "miou": m, "iou": iou})
            rnd = lambda v: None if v is None else round(v, 4)   # noqa: E731
            emit(event="eval", step=step, epoch=ep, miou=round(m, 4),
                 iou_bg=rnd(iou[0]), iou_water=rnd(iou[1]), best=is_best)
            if is_best:
                torch.save(model.state_dict(), out / "best.pt")
                model.save_pretrained(out / "best_hf")

    torch.save(model.state_dict(), out / "last.pt")
    if not (out / "best.pt").exists():              # no val set -> last is best
        shutil.copy(out / "last.pt", out / "best.pt")
        model.save_pretrained(out / "best_hf")
    (out / "metrics.json").write_text(json.dumps(
        {"best_miou": best, "history": history, "args": vars(args)}, indent=2))
    emit(event="done", best_miou=round(best, 4), ckpt=str(out / "best.pt"),
         hf=str(out / "best_hf"))


# ---------------------------------------------------------------------------
# onnx export
# ---------------------------------------------------------------------------

def export_onnx(hf_dir: str, onnx_out: str, img_size: int = 512):
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(hf_dir)
    model.eval()
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            lo = self.m(pixel_values=x).logits
            return torch.nn.functional.interpolate(
                lo, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)

    dummy = torch.zeros(1, BANDS, img_size, img_size)
    try:
        torch.onnx.export(Wrap(model), dummy, str(onnx_out),
                          input_names=["input"], output_names=["logits"],
                          opset_version=13, dynamo=False,
                          dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}})
    except Exception as e:                           # noqa: BLE001
        if "onnx" in str(e).lower():
            emit(event="error", msg="ONNX export needs the 'onnx' package: "
                 "pip install onnx  (the fine-tuned model still works in the "
                 "annotator via its HF checkpoint dir).")
            sys.exit(2)
        raise
    emit(event="onnx", path=str(onnx_out), bytes=onnx_out.stat().st_size)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="dir with img_dir/{train,val} and ann_dir/{train,val}")
    ap.add_argument("--out", help="run output dir")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--batch", type=int, default=2,
                    help="raise this on a GPU (8-16); 2 is a safe CPU default")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default: 2 on GPU, 0 on CPU)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable fp16 mixed precision (CUDA only; on by default there)")
    ap.add_argument("--init", default="nvidia/mit-b0",
                    help="HF model id or local dir to initialise from")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1, help="epochs between val")
    ap.add_argument("--export-onnx", metavar="HF_DIR",
                    help="skip training: export this HF model dir to ONNX")
    ap.add_argument("--onnx-out", default=None)
    args = ap.parse_args()

    try:
        if args.export_onnx:
            export_onnx(args.export_onnx, args.onnx_out or "segformer_5band.onnx", args.img_size)
        else:
            if not args.data_root or not args.out:
                ap.error("--data-root and --out are required for training")
            train(args)
    except Exception as e:                          # noqa: BLE001
        import traceback
        emit(event="error", msg=str(e), trace=traceback.format_exc()[-1500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
