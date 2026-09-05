"""Per-band normalisation statistics over a training split.

Why this exists: the audit in `segformer_5band/PERFORMANCE.md` identified
per-image per-band min-max normalisation as the main accuracy bug of the
original 5-band config —

    "Destroys absolute radiometric values. Water has a characteristic low
     reflectance in NIR [...]. Min-max normalization within each image erases
     that absolute signature — the model cannot learn it."

and it is also vulnerable to a single specular or dead pixel setting the range.
`annotator/trainer.py` and the `segformer` backend both did exactly that, so
this module replaces it with statistics measured once over the training split
and then *frozen into the checkpoint*, which is what keeps training and
inference in lockstep.

Every band the rig produces is uint8, so the pass accumulates an exact 256-bin
histogram per channel. Mean, std and any percentile then come out exactly, in a
single streaming pass with no per-scene arrays retained — the same cost for 12
scenes as for 12,000.

    uv run --project annotator python -m training.stats --modality fiveband
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import modalities as M

#: how a Normalizer maps raw uint8 counts to model input
METHODS = ("meanstd", "percentile", "minmax")


@dataclass
class BandStats:
    """Frozen normalisation for one modality. Travels with the checkpoint."""

    modality: str
    channels: int
    method: str
    mean: list[float]
    std: list[float]
    p_lo: list[float]
    p_hi: list[float]
    lo_q: float
    hi_q: float
    n_scenes: int
    n_pixels: int
    water_frac: float
    band_names: list[str]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: Path) -> "BandStats":
        return BandStats(**json.loads(Path(path).read_text()))

    @staticmethod
    def identity(modality: str, channels: int, method: str = "minmax") -> "BandStats":
        """Stand-in for checkpoints that carry no stats (legacy min-max)."""
        return BandStats(modality, channels, method, [0.0] * channels, [1.0] * channels,
                         [0.0] * channels, [255.0] * channels, 0.02, 0.98,
                         0, 0, 0.5, list(M.BAND_NAMES[:channels]))

    def class_weights(self, cap: float = 5.0) -> list[float]:
        """Inverse-frequency weights [background, water], clipped.

        Water covers roughly 40-60% of these frames, so this is usually near
        [1,1] — it matters for the single-band modalities and for sessions
        with only a puddle in view.
        """
        w = max(min(self.water_frac, 0.999), 1e-3)
        raw = [0.5 / (1.0 - w), 0.5 / w]
        return [float(min(v, cap)) for v in raw]


class Normalizer:
    """Applies frozen stats to a (C,H,W) uint8 array -> float32.

    `minmax` reproduces the pre-existing per-image behaviour exactly, so an old
    checkpoint keeps predicting what it always did.
    """

    def __init__(self, stats: BandStats, method: str | None = None):
        self.stats = stats
        self.method = method or stats.method
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        self.mean = np.asarray(stats.mean, np.float32).reshape(-1, 1, 1)
        self.std = np.maximum(np.asarray(stats.std, np.float32), 1e-6).reshape(-1, 1, 1)
        self.lo = np.asarray(stats.p_lo, np.float32).reshape(-1, 1, 1)
        self.hi = np.asarray(stats.p_hi, np.float32).reshape(-1, 1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, np.float32)
        if self.method == "meanstd":
            return (a - self.mean) / self.std
        if self.method == "percentile":
            rng = np.maximum(self.hi - self.lo, 1e-6)
            return np.clip((a - self.lo) / rng, 0.0, 1.0)
        out = np.empty_like(a)                      # minmax: per-image, per-band
        for i in range(a.shape[0]):
            lo, hi = float(a[i].min()), float(a[i].max())
            out[i] = (a[i] - lo) / (hi - lo) if hi > lo else 0.0
        return np.clip(out, 0.0, 1.0)


def accumulate(rows: list[dict], modality: str, lo_q: float = 0.02,
               hi_q: float = 0.98, method: str = "meanstd") -> BandStats:
    """One streaming pass over the scenes in `rows` (manifest dicts)."""
    import cv2

    m = M.get(modality)
    hist = np.zeros((m.channels, 256), np.int64)
    water = total = 0
    for r in rows:
        x = M.read_modality(m, Path(r["tiff_path"]), Path(r["scene_dir"]))
        for c in range(m.channels):
            hist[c] += np.bincount(x[c].ravel(), minlength=256)
        msk = cv2.imread(str(r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if msk is not None:
            water += int((msk > 0).sum())      # 0/255 and 0/1 masks both, see data.read_mask
            total += int(msk.size)

    n = hist[0].sum()
    if n == 0:
        raise ValueError(f"no pixels read for modality {modality!r} — empty scene list?")
    v = np.arange(256, dtype=np.float64)
    mean = (hist * v).sum(1) / n
    var = (hist * v * v).sum(1) / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    def q(frac):                                    # exact percentile from the histogram
        cum = np.cumsum(hist, axis=1)
        want = frac * n
        return np.array([float(np.searchsorted(cum[c], want)) for c in range(m.channels)])

    return BandStats(
        modality=modality, channels=m.channels, method=method,
        mean=[round(float(x), 4) for x in mean],
        std=[round(float(x), 4) for x in std],
        p_lo=[round(float(x), 4) for x in q(lo_q)],
        p_hi=[round(float(x), 4) for x in q(hi_q)],
        lo_q=lo_q, hi_q=hi_q, n_scenes=len(rows), n_pixels=int(n),
        water_frac=round(water / total, 6) if total else 0.5,
        band_names=[M.BAND_NAMES[b] for b in m.bands],
    )


def main():
    from . import manifest as mf

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("annotator/work/scenes.csv"))
    ap.add_argument("--modality", default="fiveband",
                    help=f"one of: {', '.join(M.MODALITIES)}, or 'all'")
    ap.add_argument("--split", default="train", help="which split to measure (never val/test)")
    ap.add_argument("--method", default="meanstd", choices=METHODS)
    ap.add_argument("--out-dir", type=Path, default=Path("annotator/work/norm"))
    a = ap.parse_args()

    rows = mf.load(a.manifest, split=a.split)
    names = list(M.MODALITIES) if a.modality == "all" else [a.modality]
    for name in names:
        st = accumulate(rows, name, method=a.method)
        out = a.out_dir / f"{name}.json"
        st.to_json(out)
        print(f"{name:12s} n={st.n_scenes} water={st.water_frac:.3f} "
              f"weights={[round(w, 2) for w in st.class_weights()]} -> {out}")
        for i, bn in enumerate(st.band_names):
            print(f"    {bn:8s} mean={st.mean[i]:7.2f} std={st.std[i]:6.2f} "
                  f"p{int(st.lo_q*100)}={st.p_lo[i]:5.0f} p{int(st.hi_q*100)}={st.p_hi[i]:5.0f}")


if __name__ == "__main__":
    main()
