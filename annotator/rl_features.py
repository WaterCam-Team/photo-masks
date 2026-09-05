"""Per-scene features + decision logging for the labeling-route RL dataset.

``label_log.csv`` is the training set for a policy that decides, per scene,
whether to *accept* an auto mask, *edit* it, or *hand-label from scratch*.
Each row pairs the decision and its realised cost / quality with scene
features:

    route              the action taken
    active_seconds     human cost   (h_i in the reward formula)
    auto_vs_final_iou  auto-label quality against the accepted mask (Q_i)
    edited_pixel_frac  how much of the auto mask the human had to change
    features_json      scene state the policy conditions on

See the project notes on the routing reward for how these map to r_i.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LOG_HEADER = [
    "timestamp", "annotator", "scene_id", "scene_dir",
    "route",                 # auto_accepted | auto_edited | interactive_clicks
                             # | manual_from_scratch | rejected | empty
    "seed_backend", "backend_params",
    "auto_water_pct", "final_water_pct",
    "auto_vs_final_iou", "edited_pixel_frac",
    "active_seconds", "n_strokes", "n_undos",
    "seg_mean_entropy", "seg_low_conf_frac",
    "ensemble_agreement", "ensemble_disagreement_frac", "n_backends_run",
    "features_json",
    # appended (invariant 5: new columns go at the end, never reordered)
    "n_clicks",        # SAM2 click-to-segment prompt points the human placed
    "mask_path",       # this annotator's own copy of the mask, kept for
                       # inter-annotator agreement — water_mask.png in the scene
                       # dir is overwritten by whoever saves last
]


# ---------------------------------------------------------------------------
# mask comparison
# ---------------------------------------------------------------------------

def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def edited_fraction(seed: np.ndarray | None, final: np.ndarray | None) -> float | None:
    if seed is None or final is None:
        return None
    return float(((seed > 0) != (final > 0)).mean())


# ---------------------------------------------------------------------------
# scene features
# ---------------------------------------------------------------------------

def _stats(name: str, b: np.ndarray) -> dict:
    p5, p50, p95 = np.percentile(b, [5, 50, 95])
    return {
        f"{name}_mean": float(b.mean()), f"{name}_std": float(b.std()),
        f"{name}_p5": float(p5), f"{name}_p50": float(p50), f"{name}_p95": float(p95),
    }


def scene_features(tiff_path: Path) -> dict:
    """Cheap band / index / texture descriptors for one scene."""
    import cv2
    import rasterio

    with rasterio.open(tiff_path) as src:
        arr = src.read().astype(np.float32)
    r, g, b, th, nir = arr[0], arr[1], arr[2], arr[3], arr[4]
    eps = 1e-6
    f: dict = {}

    for nm, band in (("r", r), ("g", g), ("b", b), ("thermal", th), ("nir", nir)):
        f.update(_stats(nm, band))

    ndwi = (g - nir) / (g + nir + eps)
    f.update(_stats("ndwi", ndwi))
    f["ndwi_pos_frac"] = float((ndwi > 0).mean())

    nn = (nir - nir.min()) / (nir.max() - nir.min() + eps)
    for t in (0.15, 0.25, 0.35):
        f[f"nir_dark_frac_{int(t * 100)}"] = float((nn < t).mean())

    # thermal bimodality: Otsu between-class variance ratio (0 = unimodal)
    th8 = np.clip(th, 0, 255).astype(np.uint8)
    tval, _ = cv2.threshold(th8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = th[th <= tval], th[th > tval]
    if lo.size and hi.size:
        w0, w1 = lo.size / th.size, hi.size / th.size
        f["thermal_otsu_sep"] = float(
            w0 * w1 * (lo.mean() - hi.mean()) ** 2 / (th.var() + eps))
    else:
        f["thermal_otsu_sep"] = 0.0

    # optical texture / edges
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    f["edge_density"] = float((mag > mag.mean() + mag.std()).mean())

    hist = np.histogram(gray, bins=32, range=(0, 255))[0].astype(np.float64)
    p = hist / (hist.sum() + eps)
    p = p[p > 0]
    f["gray_entropy"] = float(-(p * np.log2(p)).sum())
    return f


# ---------------------------------------------------------------------------
# log writer
# ---------------------------------------------------------------------------

def append_log(log_path: Path, row: dict) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_HEADER, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def classify_route(seed_present: bool, n_strokes: int,
                   iou: float | None, final_has_water: bool,
                   interactive: bool = False) -> str:
    """Which labeling route produced this mask — the RL policy's action.

    `interactive_clicks` is its own action rather than a flavour of
    `auto_edited`: steering SAM2 by hand and accepting a batch backend's
    proposal have very different costs, and collapsing them would teach the
    policy that they are interchangeable. How *much* correction followed is
    already carried quantitatively by n_strokes, auto_vs_final_iou and
    edited_pixel_frac, so the accepted/edited split stays recoverable without
    fragmenting a small dataset into more route values.
    """
    if not seed_present:
        return "manual_from_scratch" if final_has_water else "empty"
    if interactive:
        return "interactive_clicks"
    if n_strokes == 0 and iou is not None and iou >= 0.999:
        return "auto_accepted"
    return "auto_edited"
