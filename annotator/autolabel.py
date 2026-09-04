"""Spectral water auto-labeling for UFONet 5-band scenes (no torch).

`spectral_water()` fuses several cheap physics/appearance cues into a
water-probability map, then snaps the boundary to RGB edges:

    NIR darkness     water absorbs NIR  -> low values (primary cue)
    NDWI             (G - NIR)/(G + NIR), high over water (secondary)
    thermal smooth   water is spatially smooth in LWIR
    low texture      water surface has little optical texture
    specular/sky     bright, blue-ish, smooth  -> reflected sky ON water
    below-horizon    sky region is hard non-water

`derive_prompts()` turns the probability map into positive/negative click
points + a box, for prompting a SAM-style segmenter.

Band order (0-based): 0 R  1 G  2 B  3 Thermal(LWIR)  4 NIR-difference
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6
B_R, B_G, B_B, B_TH, B_NIR = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _norm(x: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _otsu01(x: np.ndarray) -> float:
    """Otsu threshold of a float array scaled to [0,1]."""
    import cv2
    u8 = np.clip(x * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def _local_std(gray: np.ndarray, k: int = 7) -> np.ndarray:
    import cv2
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sq = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def _grad_mag(x: np.ndarray) -> np.ndarray:
    import cv2
    gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _largest_components(mask: np.ndarray, min_area_frac: float = 0.002) -> np.ndarray:
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    keep = np.zeros_like(mask)
    thr = min_area_frac * mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            keep[lab == i] = 1
    return keep


def horizon_row(bands: np.ndarray) -> int | None:
    """Rough sky/ground split: lowest row that is still mostly 'sky-like'.

    Sky-like = bright and blue >= red. Returns a row index, or None if the
    top of the frame isn't sky (camera pointed down / indoors).
    """
    r, g, b = bands[B_R], bands[B_G], bands[B_B]
    v = np.maximum(np.maximum(r, g), b) / 255.0
    sky = (v > 0.55) & (b >= r - 5)
    frac = sky.mean(axis=1)
    if frac[:10].mean() < 0.3:                       # frame doesn't start on sky
        return None
    below = np.where(frac < 0.15)[0]
    return int(below[0]) if below.size else None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def spectral_water(bands: np.ndarray, params: dict | None = None):
    """Return (mask uint8 {0,255}, meta dict, prob float32 HxW).

    Water must be relatively DARK in NIR (snow / bright pavement reflect NIR
    and are excluded by that gate); the appearance cues only modulate the
    confidence within NIR-dark regions.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    p = {
        "nir_gate": 0.45,          # nir_n below this -> water-plausible
        "nir_gate_soft": 0.12,     # gate softness
        "w_ndwi": 0.40, "w_texture": 0.35, "w_thermal": 0.25,   # support blend
        "base": 0.35,              # min prob inside the gate before support
        "specular": False,         # reflected-sky-on-water; off by default (snow FPs)
        "snow_guard": True,        # suppress bright achromatic regions (snow, blown sky, white walls)
        "horizon": True, "refine": "guided",
        "min_area_frac": 0.003, "thresh": 0.5,
    }
    p.update(params or {})

    bands = bands.astype(np.float32)
    r, g, b, th, nir = (bands[B_R], bands[B_G], bands[B_B], bands[B_TH], bands[B_NIR])
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    nir_range = float(nir.max() - nir.min())
    nir_reliable = nir_range > 12
    nir_n = _norm(nir)

    ndwi = (g - nir) / (g + nir + EPS)
    ndwi_n = np.clip((ndwi + 1) / 2, 0, 1).astype(np.float32)
    ndwi_t = _otsu01(ndwi_n)
    s_ndwi = np.clip((ndwi_n - ndwi_t) / 0.15 + 0.5, 0, 1).astype(np.float32)

    th_grad = _norm(cv2.GaussianBlur(_grad_mag(th), (0, 0), 2))
    s_thermal = cv2.GaussianBlur(1.0 - th_grad, (0, 0), 3).astype(np.float32)

    s_tex = (1.0 - _norm(cv2.GaussianBlur(_local_std(gray, 7), (0, 0), 2))).astype(np.float32)

    wsum = p["w_ndwi"] + p["w_texture"] + p["w_thermal"] + EPS
    support = (p["w_ndwi"] * s_ndwi + p["w_texture"] * s_tex
               + p["w_thermal"] * s_thermal) / wsum

    if nir_reliable:
        nir_gate = _sigmoid((p["nir_gate"] - nir_n) / p["nir_gate_soft"]).astype(np.float32)
        prob = nir_gate * (p["base"] + (1.0 - p["base"]) * support)
    else:                                            # NIR diff unusable: cues only, stricter
        prob = (0.55 * np.clip((s_ndwi - 0.55) / 0.25, 0, 1)
                + 0.30 * s_tex + 0.15 * s_thermal).astype(np.float32)
    prob = prob.astype(np.float32)

    # bright achromatic = snow / blown-out sky / white siding; the NIR-difference
    # band does NOT separate these from water, so knock them down explicitly.
    snow_frac = 0.0
    if p["snow_guard"]:
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = (mx - mn) / (mx + EPS)
        snow = (mx / 255.0 > 0.78) & (sat < 0.10)
        prob = (prob * np.where(snow, 0.12, 1.0)).astype(np.float32)
        snow_frac = float(snow.mean())

    spec_frac = 0.0
    if p["specular"] and nir_reliable:               # only bright+smooth+NIR-dark near a water prior
        v = np.maximum(np.maximum(r, g), b) / 255.0
        near = cv2.dilate((prob > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        spec = (v > 0.75) & (b >= r) & (nir_n < 0.5) & near
        prob = np.maximum(prob, np.where(spec, 0.85, 0.0)).astype(np.float32)
        spec_frac = float(spec.mean())

    hr = horizon_row(bands) if p["horizon"] else None
    if hr is not None and hr > 5:
        prob[:hr] = 0.0

    thr = float(p["thresh"])
    mask = (prob >= thr).astype(np.uint8)
    mask = _largest_components(mask, p["min_area_frac"])
    mask = binary_fill_holes(mask).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    rgb = np.dstack([r, g, b]).astype(np.uint8)
    mask, refine_used = refine_boundary(rgb, mask, prob, method=p["refine"])

    meta = {
        "ndwi_otsu": round(ndwi_t, 3),
        "nir_dyn_range": round(nir_range, 1),
        "nir_reliable": nir_reliable,
        "snow_guard_frac": round(snow_frac, 4),
        "specular_frac": round(spec_frac, 4),
        "horizon_row": hr,
        "prob_thresh": round(thr, 3),
        "water_pct": round(100 * float((mask > 0).mean()), 2),
        "refine": refine_used,
    }
    return (mask.astype(np.uint8) * 255), meta, prob


# ---------------------------------------------------------------------------
# boundary refinement
# ---------------------------------------------------------------------------

def guided_available() -> bool:
    """cv2.ximgproc.guidedFilter ships in opencv-contrib, not plain opencv.

    Install the wrong wheel and the `guided` refine mode becomes a silent
    no-op, so callers check this and report which method actually ran rather
    than claiming a refinement that never happened.
    """
    import cv2
    return hasattr(cv2, "ximgproc")


def refine_boundary(rgb: np.ndarray, coarse: np.ndarray, prob: np.ndarray,
                    method: str = "guided") -> tuple[np.ndarray, str]:
    """Snap `coarse` to RGB edges. Returns (mask, method_actually_used).

    The returned method is what ran, which is not always what was asked for:
    `guided` needs opencv-contrib and `rw` needs scikit-image, and either can
    fall back to `none`.
    """
    import cv2

    if method == "guided" and guided_available():
        guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        src = np.clip(np.maximum(prob, coarse.astype(np.float32)), 0, 1).astype(np.float32)
        filt = cv2.ximgproc.guidedFilter(guide, src, radius=8, eps=1e-4)
        out = (filt >= max(0.4, float(np.percentile(filt[coarse > 0], 30))
                           if coarse.any() else 0.5)).astype(np.uint8)
        return (out, "guided") if out.any() else (coarse, "none")

    if method == "guided":
        return coarse, "none (guided needs opencv-contrib-python-headless)"

    if method == "rw":
        try:
            from skimage.segmentation import random_walker
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fg = cv2.erode(coarse, k, iterations=2)
            bg = cv2.erode(1 - coarse, k, iterations=2)
            if fg.any() and bg.any():
                markers = np.zeros_like(coarse, dtype=np.int32)
                markers[bg > 0] = 1
                markers[fg > 0] = 2
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                lab = random_walker(gray, markers, beta=130, mode="bf")
                return (lab == 2).astype(np.uint8), "rw"
        except ImportError:
            return coarse, "none (rw needs scikit-image)"
        except Exception:                             # noqa: BLE001
            return coarse, "none (rw failed)"

    return coarse, "none"


# ---------------------------------------------------------------------------
# prompt derivation for SAM-style models
# ---------------------------------------------------------------------------

def _spread_sample(coords: np.ndarray, weights: np.ndarray, n: int,
                   grid: int = 6) -> np.ndarray:
    """Pick up to n points, spatially spread by bucketing into a grid."""
    if len(coords) == 0:
        return np.empty((0, 2), int)
    ys, xs = coords[:, 0], coords[:, 1]
    H = ys.max() + 1
    W = xs.max() + 1
    bucket = (ys * grid // H) * grid + (xs * grid // W)
    picks = []
    for bk in np.unique(bucket):
        idx = np.where(bucket == bk)[0]
        picks.append(idx[np.argmax(weights[idx])])
    picks = np.array(picks)
    if len(picks) > n:
        picks = picks[np.argsort(weights[picks])[::-1][:n]]
    return np.stack([xs[picks], ys[picks]], axis=1)    # (x, y)


def derive_prompts(bands: np.ndarray, prob: np.ndarray,
                   n_pos: int = 8, n_neg: int = 8,
                   pos_thr: float = 0.8, neg_thr: float = 0.15):
    """Return dict: points (Nx2 x,y int), labels (N,), box ([x0,y0,x1,y1] or None)."""
    import cv2

    nir_n = _norm(bands[B_NIR])
    gray = (0.299 * bands[B_R] + 0.587 * bands[B_G] + 0.114 * bands[B_B])
    tex = _norm(_local_std(gray, 7))

    pc = np.argwhere(prob >= pos_thr)
    pos = _spread_sample(pc, prob[prob >= pos_thr], n_pos)

    neg_score = np.where(prob <= neg_thr, 0.5 + 0.25 * nir_n + 0.25 * tex, 0.0)
    nc = np.argwhere(neg_score > 0)
    neg = _spread_sample(nc, neg_score[neg_score > 0], n_neg)

    pts = np.concatenate([pos, neg], axis=0) if len(pos) or len(neg) else np.empty((0, 2), int)
    labels = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])

    box = None
    strong = (prob >= 0.6).astype(np.uint8)
    if strong.any():
        strong = cv2.dilate(strong, np.ones((15, 15), np.uint8))
        ys, xs = np.where(strong > 0)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return {"points": pts.astype(int), "labels": labels.astype(int), "box": box}
