"""Scene dataset, geometry and augmentation.

Three things here differ deliberately from `annotator/trainer.py`, which
resized every frame to a 512x512 square:

1. **Geometry is preserved.** The rig's frames are 1296x972 (4:3); squashing
   them to 1:1 stretches every shoreline by 33% in one axis, and the model then
   sees a geometry that never occurs at inference. Training instead samples a
   random scale and takes a fixed-size crop, so pixels keep their aspect and
   the model sees full sensor detail rather than a 2.5x downscale.
2. **Padding is ignored, not labelled.** Where a crop or an eval pad extends
   past the frame, the mask is filled with `IGNORE` (255) rather than 0, so the
   loss never learns "the edge of the image is background".
3. **Photometric augmentation touches only visible-light channels.** Jittering
   brightness on a thermal or NIR-difference band would fabricate radiometry
   the sensor cannot produce, and those absolute values are exactly the water
   signature the model is supposed to learn (see `stats.py`).

Crops are rejected and resampled when one class exceeds `cat_max_ratio` of the
crop, which keeps the batch from filling up with all-water or all-shore tiles —
the same guard mmseg's RandomCrop applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import modalities as M
from .stats import Normalizer

IGNORE = 255                     # label value the loss must skip

try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:                                   # noqa: BLE001 - torch imported later
    _TorchDataset = object


def read_mask(path: Path) -> np.ndarray:
    """Binary water mask -> (H,W) uint8 in {0,1}.

    `> 0`, not `> 127`, because two mask encodings are in circulation and both
    must read correctly: the annotator writes `water_mask.png` as 0/255, while
    `export_dataset.py` converts to class indices 0/1 for its img_dir/ann_dir
    tree. Thresholding at 127 reads every exported mask as entirely
    background — empty labels, no error, a model that predicts nothing.
    """
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)


def _resize(x: np.ndarray, hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    import cv2
    h, w = hw
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    if x.ndim == 2:
        return cv2.resize(x, (w, h), interpolation=interp)
    return np.stack([cv2.resize(x[i], (w, h), interpolation=interp)
                     for i in range(x.shape[0])])


def _pad_to(x: np.ndarray, y: np.ndarray, mult: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad right/bottom so H and W are multiples of `mult` (image 0, mask IGNORE)."""
    h, w = y.shape
    ph, pw = (-h) % mult, (-w) % mult
    if not (ph or pw):
        return x, y
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
    y = np.pad(y, ((0, ph), (0, pw)), constant_values=IGNORE)
    return x, y


def _photometric(x: np.ndarray, channels: tuple[int, ...], rng: np.random.Generator
                 ) -> np.ndarray:
    """Brightness / contrast / per-channel gain on visible-light channels only."""
    if not channels:
        return x
    out = x.astype(np.float32)
    sel = list(channels)
    if rng.random() < 0.5:                                    # brightness
        out[sel] += rng.uniform(-24, 24)
    if rng.random() < 0.5:                                    # contrast
        mu = out[sel].mean()
        out[sel] = (out[sel] - mu) * rng.uniform(0.75, 1.25) + mu
    if rng.random() < 0.3:                                    # per-channel gain
        for c in sel:
            out[c] *= rng.uniform(0.92, 1.08)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Geometry:
    """How a scene becomes a training or eval tensor."""

    crop: int = 512                       # training crop side, in sensor pixels
    scale_range: tuple[float, float] = (0.5, 2.0)
    cat_max_ratio: float = 0.85           # reject a crop this dominated by one class
    #: eval: None keeps the native frame (padded to /32); an int resizes the
    #: long side to it, preserving aspect
    eval_long_side: int | None = None
    pad_mult: int = 32


class SceneDataset(_TorchDataset):
    """Module-level and picklable — DataLoader workers use forkserver on 3.14."""

    def __init__(self, rows: list[dict], modality: str, norm: Normalizer,
                 geom: Geometry | None = None, train: bool = True,
                 photometric: bool = True, cache: bool = True, seed: int = 0):
        self.rows = list(rows)
        self.modality = modality
        self.m = M.get(modality)
        self.norm = norm
        self.geom = geom or Geometry()
        self.train = train
        self.photometric = photometric and train
        self.cache_on = cache
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, e: int) -> None:
        self.epoch = e                    # keeps worker RNGs decorrelated per epoch

    def _read(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i in self._cache:
            return self._cache[i]
        r = self.rows[i]
        x = M.read_modality(self.m, Path(r["tiff_path"]), Path(r["scene_dir"]))
        y = read_mask(Path(r["mask_path"]))
        if x.shape[1:] != y.shape:
            raise ValueError(f"{r['stem']}: mask {y.shape} does not match image "
                             f"{x.shape[1:]} — mask must be on the TIFF grid")
        if self.cache_on:
            self._cache[i] = (x, y)
        return x, y

    def _train_sample(self, x, y, rng):
        c = self.geom.crop
        s = rng.uniform(*self.geom.scale_range)
        h, w = y.shape
        # never scale below the crop in both axes: it would be all padding
        s = max(s, c / min(h, w) * 0.6)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        x, y = _resize(x, (nh, nw)), _resize(y, (nh, nw), nearest=True)
        if nh < c or nw < c:
            ph, pw = max(0, c - nh), max(0, c - nw)
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
            y = np.pad(y, ((0, ph), (0, pw)), constant_values=IGNORE)
            nh, nw = y.shape
        best = None
        for attempt in range(10):
            top = rng.integers(0, nh - c + 1)
            left = rng.integers(0, nw - c + 1)
            yc = y[top:top + c, left:left + c]
            valid = yc[yc != IGNORE]
            if valid.size == 0:
                continue
            frac = float(np.bincount(valid, minlength=2).max() / valid.size)
            if best is None or frac < best[0]:
                best = (frac, top, left)
            if frac <= self.geom.cat_max_ratio:
                break
        if best is None:                                  # all-ignore, take a corner
            best = (1.0, 0, 0)
        _, top, left = best
        x = x[:, top:top + c, left:left + c]
        y = y[top:top + c, left:left + c]
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        if self.photometric:
            x = _photometric(x, self.m.optical, rng)
        return x, y

    def _eval_sample(self, x, y):
        long = self.geom.eval_long_side
        if long:
            h, w = y.shape
            s = long / max(h, w)
            if s != 1.0:
                nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
                x, y = _resize(x, (nh, nw)), _resize(y, (nh, nw), nearest=True)
        return _pad_to(x, y, self.geom.pad_mult)

    def __getitem__(self, i: int):
        import torch
        x, y = self._read(i)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            x, y = self._train_sample(x, y, rng)
        else:
            x, y = self._eval_sample(x, y)
        xt = torch.from_numpy(np.ascontiguousarray(self.norm(x)))
        yt = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64)))
        return xt, yt


def legacy_rows(data_root: Path) -> dict[str, list[dict]]:
    """Read an `export_dataset.py` img_dir/ann_dir tree as manifest-style rows.

    Keeps the annotator's Train panel working unchanged. Only band-subset
    modalities are available this way: the export copies the TIFF but not the
    NIR-ON frame, so `rgb_nofilt` needs a real manifest.
    """
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        img = Path(data_root) / "img_dir" / split
        ann = Path(data_root) / "ann_dir" / split
        if not img.exists():
            continue
        rows = []
        for p in sorted(img.glob("*.tif*")):
            mask = ann / f"{p.stem}.png"
            if mask.exists():
                rows.append({"stem": p.stem, "scene_dir": str(img),
                             "tiff_path": str(p), "mask_path": str(mask),
                             "split": split, "fold": "-1", "route": "", "n_annotators": "1"})
        if rows:
            out[split] = rows
    return out
