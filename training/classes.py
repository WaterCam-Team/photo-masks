"""The label taxonomy, in one place.

Decided 2026-09-27. Four classes, replacing the binary water/background mask:

    0  background      everything not covered below
    1  water           standing or flowing liquid water — the flood signal
    2  snow_ice        snow, ice, frozen surface
    3  wet_ground      wet pavement or soil, dark sheen, no visible depth

Why these three negatives are separate rather than one "background": they are
the confusions that matter. Snow and wet asphalt are what a water detector gets
wrong, and NIR separates them physically — water absorbs, snow reflects — so a
5-band model *can* learn the distinction, but only if the labels draw it. Folded
into one undifferentiated background alongside sky, trees and buildings, the
signal that justifies carrying the NIR band is thrown away.

**The deployed node stays binary.** `DEPLOY_BINARY` collapses everything except
water back to background, so the mask a node transmits is unchanged and the LoRa
bitmap keeps its one bit per pixel. The extra classes are training signal, not
output — see `SU-WaterCam/BITMAP_COMPRESSION_ANALYSIS.md` for the 228-byte
budget that makes this the sensible split.
"""
from __future__ import annotations

#: Canonical order. Index is the label value stored in a mask.
CLASS_NAMES: tuple[str, ...] = ("background", "water", "snow_ice", "wet_ground")

BACKGROUND, WATER, SNOW_ICE, WET_GROUND = 0, 1, 2, 3

#: Label value the loss must skip: pixels outside the frame after a warp, and
#: pixels a legacy binary mask cannot speak for. Must stay 255 — it is what
#: `training/data.py` pads with.
IGNORE = 255

#: Display colours (R, G, B), for the annotator overlay and QA previews.
#: Chosen to stay distinguishable under the common colour-vision deficiencies.
PALETTE: dict[str, tuple[int, int, int]] = {
    "background": (40, 40, 40),
    "water": (0, 114, 178),        # blue
    "snow_ice": (240, 228, 66),    # yellow
    "wet_ground": (213, 94, 0),    # vermillion
}

#: Which classes count as water when a multi-class model is deployed binary.
WATER_CLASSES: frozenset[int] = frozenset({WATER})


def name(index: int) -> str:
    return CLASS_NAMES[index]


def index(name_: str) -> int:
    try:
        return CLASS_NAMES.index(name_)
    except ValueError:
        raise KeyError(f"unknown class {name_!r}; "
                       f"choose from {', '.join(CLASS_NAMES)}") from None


def collapse(mask, num_classes: int):
    """Map a mask's labels into `range(num_classes)`, preserving IGNORE.

    Labelling is four-class; training and deployment may be binary. A 2-class
    model fed a raw four-class mask gets targets 2 and 3 against 2 logits,
    which torch rejects outright — and a model that *didn't* reject it would be
    learning from labels it cannot represent.

    Collapsing keeps water and folds every other class into background, which
    is the same rule `deploy_binary` applies at inference, so the binary model
    trains on exactly what it will be asked to predict. Identity when the mask
    already fits.
    """
    import numpy as np
    m = np.asarray(mask)
    if num_classes >= len(CLASS_NAMES):
        return m
    out = np.zeros_like(m)
    out[m == IGNORE] = IGNORE
    keep = [c for c in range(num_classes) if c != BACKGROUND]
    for c in keep:
        out[m == c] = c
    return out


def collapse_fracs(fracs: list, num_classes: int) -> list:
    """The same collapse applied to per-class pixel fractions."""
    if not fracs or num_classes >= len(fracs):
        return list(fracs)
    out = [0.0] * num_classes
    for i, f in enumerate(fracs):
        out[i if i < num_classes else BACKGROUND] += float(f)
    return out


def deploy_binary(mask):
    """Collapse a multi-class mask to the binary one a node transmits."""
    import numpy as np
    m = np.asarray(mask)
    out = np.zeros_like(m, dtype="uint8")
    for c in WATER_CLASSES:
        out[m == c] = 1
    return out
