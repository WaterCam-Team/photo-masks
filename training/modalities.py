"""Named input modalities, all living in one pixel space.

Every modality here is sampled on the **co-registered grid of
`color_preserved_5_band.tiff`** (typically 972x1296), because that is the grid
the annotator's gold masks are drawn on. A modality that needed a different
geometry would need the mask warped to match, and the co-registration transform
is not stored in the scene directory — co-registration happens upstream on the
capture rig and is outside this repo.

Provenance of the 5 TIFF bands, measured across all 13 labelled scenes (5
capture sessions) rather than assumed — see `verify_provenance()` to re-check
on your own data:

    band 0,1,2  R,G,B    == `*-NIR-OFF.jpg` downscaled 2x
                         MAE 0.00 on every scene: exact, in every session.
    band 3      thermal  FLIR Lepton, co-registration-warped.
                         vs. a plain resize of `IMG_*.pgm`: MAE 12 to 1465,
                         never close. The raw .pgm is in the Lepton's own
                         160x120 geometry, is NOT in mask space, and must not
                         be used as a thermal modality.
    band 4      NIR      == `NIR_band.png` exactly (MAE 0.00 on every scene).
                         Nominally red(NIR-ON) - red(NIR-OFF), but that
                         identity is only loose in practice (band0 + band4 vs.
                         red(NIR-ON) ranges from MAE 0.25 to 28.9 across
                         sessions — the difference clips at 0/255 and exposure
                         moves between the two captures). Nothing here relies
                         on it: `rgb_nofilt` reads the NIR-ON frame itself
                         rather than reconstructing it.

The claim `rgb_nofilt` actually rests on is the first one. Because bands 0-2
are an *exact* 2x downscale of the NIR-OFF frame, the optical image was
resized and not warped during co-registration, so the NIR-ON frame can be
downscaled the same way and still line up with the mask. That has held for
every session captured so far; re-check it before trusting a new one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# TIFF band indices, by meaning
R, G, B, THERMAL, NIR = 0, 1, 2, 3, 4

BAND_NAMES = ("red", "green", "blue", "thermal", "nir")


@dataclass(frozen=True)
class Modality:
    """A named channel set plus how to read it onto the mask grid."""

    name: str
    bands: tuple[int, ...]          # TIFF band indices, in output channel order
    description: str
    #: glob for a sibling file replacing the optical bands (rgb_nofilt only)
    extra_glob: str | None = None
    #: which output channels the extra file supplies, if any
    extra_channels: tuple[int, ...] = field(default_factory=tuple)
    #: channels holding visible-light data — the only ones a photometric
    #: augmentation may touch (brightness/contrast jitter on a thermal or NIR
    #: difference band would fake radiometry the sensor cannot produce)
    optical: tuple[int, ...] = (0, 1, 2)

    @property
    def channels(self) -> int:
        return len(self.bands)


MODALITIES: dict[str, Modality] = {
    "fiveband": Modality(
        "fiveband", (R, G, B, THERMAL, NIR),
        "RGB + thermal + NIR difference. The full sensor fusion the rig captures."),
    "rgb_std": Modality(
        "rgb_std", (R, G, B),
        "Standard filtered RGB, i.e. an ordinary colour photo (the NIR-OFF frame)."),
    "rgb_nofilt": Modality(
        "rgb_nofilt", (R, G, B),
        "RGB with the NIR-cut filter off (the NIR-ON frame): visible + leaked NIR.",
        extra_glob="*-NIR-ON.jpg", extra_channels=(0, 1, 2)),
    "lwir": Modality(
        "lwir", (THERMAL,),
        "FLIR Lepton thermal alone. Lepton native is 160x120, upsampled upstream.",
        optical=()),
    "nir": Modality(
        "nir", (NIR,),
        "NIR difference alone — the band water absorbs most strongly.",
        optical=()),
    "rgb_nir": Modality(
        "rgb_nir", (R, G, B, NIR),
        "Fusion ablation: drop thermal, keep the NIR difference."),
    "rgb_thermal": Modality(
        "rgb_thermal", (R, G, B, THERMAL),
        "Fusion ablation: drop NIR, keep thermal."),
}

#: the four modalities of the sensor-value comparison (segformer_5band/MODALITY_COMPARISON.md)
COMPARISON_SET = ("rgb_std", "rgb_nofilt", "lwir", "fiveband")


def get(name: str) -> Modality:
    try:
        return MODALITIES[name]
    except KeyError:
        raise KeyError(f"unknown modality {name!r}; "
                       f"choose from {', '.join(MODALITIES)}") from None


def read_tiff(path: Path) -> np.ndarray:
    """Read all 5 bands as (5,H,W) uint8, zero-padding any missing band."""
    import rasterio
    with rasterio.open(path) as s:
        a = s.read()
    if a.shape[0] < 5:
        a = np.concatenate([a, np.zeros((5 - a.shape[0], *a.shape[1:]), a.dtype)])
    return a[:5]


def read_modality(m: Modality, tiff_path: Path, scene_dir: Path | None = None) -> np.ndarray:
    """Load one scene as (C,H,W) uint8 on the TIFF/mask grid."""
    import cv2
    a = read_tiff(tiff_path)
    out = a[list(m.bands)].copy()
    if m.extra_glob:
        d = scene_dir or tiff_path.parent
        hits = sorted(d.glob(m.extra_glob))
        if not hits:
            raise FileNotFoundError(
                f"modality {m.name!r} needs a file matching {m.extra_glob} in {d}")
        im = cv2.imread(str(hits[0]), cv2.IMREAD_COLOR)      # BGR
        if im is None:
            raise OSError(f"could not read {hits[0]}")
        H, W = a.shape[1], a.shape[2]
        if im.shape[:2] != (H, W):
            # same downscale the co-registration applied to the optical frame
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        rgb = im[:, :, ::-1]                                  # -> RGB
        for ch, src in zip(m.extra_channels, range(3)):
            out[ch] = rgb[:, :, src]
    return out


def verify_provenance(scene_dir: Path) -> dict[str, float]:
    """Re-measure the band-provenance claims in this module's docstring.

    Returns mean-absolute-error per claim; ~0 means the claim holds. Run this
    on new capture sessions before trusting `rgb_nofilt`: it is only valid
    while the optical frame is resized rather than warped.
    """
    import cv2
    tiff = next(iter(sorted(scene_dir.glob("*_5_band.tiff"))), None)
    if tiff is None:
        raise FileNotFoundError(f"no *_5_band.tiff in {scene_dir}")
    a = read_tiff(tiff)
    H, W = a.shape[1], a.shape[2]
    out: dict[str, float] = {}

    def mae(x, y):
        return float(np.abs(x.astype(np.int32) - y.astype(np.int32)).mean())

    def load(glob, flag=cv2.IMREAD_UNCHANGED):
        hits = sorted(scene_dir.glob(glob))
        return cv2.imread(str(hits[0]), flag) if hits else None

    off = load("*-NIR-OFF.jpg", cv2.IMREAD_COLOR)
    if off is not None:
        d = cv2.resize(off, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        out["rgb_bands_vs_nir_off"] = mae(np.transpose(a[:3], (1, 2, 0)), d)
    nirp = load("NIR_band.png")
    if nirp is not None:
        out["band4_vs_nir_band_png"] = mae(a[NIR], nirp)
    on = load("*-NIR-ON.jpg", cv2.IMREAD_COLOR)
    if on is not None:
        d = cv2.resize(on, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        est = np.clip(a[R].astype(np.int32) + a[NIR].astype(np.int32), 0, 255)
        out["r_off_plus_nir_vs_r_on"] = mae(est, d[:, :, 0])
    pgm = load("*.pgm")
    if pgm is not None:
        d = cv2.resize(pgm, (W, H), interpolation=cv2.INTER_LINEAR)
        out["band3_vs_raw_pgm"] = mae(a[THERMAL], d)      # expected LARGE: warped
    return out
