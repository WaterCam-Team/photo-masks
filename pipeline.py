#!/usr/bin/env python
"""Batch auto-labeling: co-registered 5-band TIFFs → water masks → manifest.csv.

**Co-registration is not part of this repo.** These tools only ever read scenes
that a capture rig has already co-registered into a 5-band TIFF. This script
scans a captures root for those TIFFs, runs one of the annotator's own
auto-labeling backends on each, writes `water_mask_auto.png` next to the TIFF,
and writes/updates `manifest.csv` for `review.py` or the annotator to pick up.

Band order in the 5-band TIFF:
    1 = Red   2 = Green   3 = Blue   4 = Thermal (LWIR)   5 = NIR difference

Usage:
    python pipeline.py <captures_dir> [options]

    python pipeline.py example_data/
    python pipeline.py /data/captures/ --backend nir --threshold 0.20 --preview
    python pipeline.py /data/captures/ --force        # re-label scenes that have a mask
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "annotator"))     # backends.py / autolabel.py live there

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TIFF_NAMES = ("color_preserved_5_band.tiff", "final_5_band.tiff")
EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "site-packages",
                "__pycache__", "work"}
MAX_DEPTH = 6

MASK_NAME = "water_mask_auto.png"
PREVIEW_NAME = "water_mask_preview.jpg"

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = [
    "scene_dir",
    "tiff_path",
    "autolabel_status",  # ok | error
    "mask_path",
    "water_pct",
    "review_status",     # pending | approved | rejected | manual
    "updated",
]

# CPU-only backends worth running in a batch. The model-based ones (sam, sam2,
# segformer) live in the annotator, where a human is there to check the result.
BACKENDS = ("spectral", "nir", "thermal")

# each backend's primary cut-off, so one --threshold flag can drive any of them
THRESHOLD_PARAM = {"nir": "threshold", "spectral": "thresh"}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    """Return dict keyed by scene_dir -> row dict."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row["scene_dir"]: row for row in reader}


def save_manifest(path: Path, rows: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def find_scenes(captures_root: Path) -> list[tuple[Path, Path]]:
    """Walk captures_root for directories holding a 5-band TIFF.

    Returns [(scene_dir, tiff_path)], one per directory, first matching name in
    TIFF_NAMES wins. Symlinked directories are followed, with a realpath guard
    against loops.
    """
    base_depth = len(captures_root.parts)
    found: list[tuple[Path, Path]] = []
    seen_real: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(captures_root, followlinks=True):
        d = Path(dirpath)
        real = os.path.realpath(dirpath)
        if real in seen_real:
            dirnames[:] = []
            continue
        seen_real.add(real)
        dirnames[:] = sorted(x for x in dirnames if x not in EXCLUDE_DIRS)
        if len(d.parts) - base_depth >= MAX_DEPTH:
            dirnames[:] = []
        tiff = next((d / n for n in TIFF_NAMES if n in filenames), None)
        if tiff is not None:
            found.append((d, tiff))
    return sorted(found)


# ---------------------------------------------------------------------------
# Auto-labeling step
# ---------------------------------------------------------------------------

def make_backend(name: str, threshold: float | None):
    """Instantiate one of the annotator's CPU backends + its params."""
    import backends

    cls = {"spectral": backends.SpectralBackend,
           "nir": backends.NirThresholdBackend,
           "thermal": backends.ThermalOtsuBackend}[name]
    params: dict = {}
    if threshold is not None and name in THRESHOLD_PARAM:
        params[THRESHOLD_PARAM[name]] = threshold
    return cls(), params


def write_preview(tiff_path: Path, mask, out_path: Path) -> None:
    """Side-by-side JPEG: true-colour photo | mask overlaid in magenta."""
    import cv2
    import numpy as np
    import rasterio

    with rasterio.open(tiff_path) as src:
        arr = src.read().astype(np.float32)
    bgr = np.dstack([arr[2], arr[1], arr[0]]).clip(0, 255).astype(np.uint8)
    overlay = bgr.copy()
    overlay[mask > 127] = (204, 0, 102)
    combined = np.hstack([bgr, cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)])
    cv2.imwrite(str(out_path), combined)


def run_autolabel(backend, params: dict, scene_dir: Path, tiff_path: Path,
                  force: bool, preview: bool) -> tuple[str, Path | None, float]:
    """Auto-label one scene. Returns (status, mask_path, water_pct)."""
    import cv2
    import numpy as np

    mask_path = scene_dir / MASK_NAME
    if mask_path.exists() and not force:
        cached = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if cached is not None:
            log.info(f"  using cached {mask_path.name}")
            return "ok", mask_path, float(np.mean(cached > 127)) * 100

    res = backend.run(scene_dir, tiff_path, params)
    if res.error or res.mask is None:
        log.error(f"  autolabel failed: {res.error or 'backend returned no mask'}")
        return "error", None, 0.0

    cv2.imwrite(str(mask_path), res.mask)
    water_pct = res.meta.get("water_pct")
    if water_pct is None:
        water_pct = 100 * float((res.mask > 127).mean())

    if preview:
        try:
            write_preview(tiff_path, res.mask, scene_dir / PREVIEW_NAME)
        except Exception as e:                       # noqa: BLE001 - preview is optional
            log.warning(f"  preview failed: {e}")

    return "ok", mask_path, float(water_pct)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures_dir",
                        help="Root directory to scan for scene dirs holding a 5-band TIFF")
    parser.add_argument("--backend", choices=BACKENDS, default="spectral",
                        help="Auto-labeling backend. Default: spectral")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Backend cut-off: NIR level for --backend nir (0–1, "
                             "default 0.25), water probability for --backend spectral "
                             "(0–1, default 0.5). Ignored by thermal.")
    parser.add_argument("--preview", action="store_true",
                        help=f"Also write {PREVIEW_NAME} (photo | mask overlay) per scene")
    parser.add_argument("--force", action="store_true",
                        help="Re-label scenes that already have a mask")
    args = parser.parse_args()

    captures_root = Path(args.captures_dir).resolve()
    if not captures_root.is_dir():
        log.error(f"Not a directory: {captures_root}")
        sys.exit(1)

    manifest_path = captures_root / MANIFEST_NAME
    manifest = load_manifest(manifest_path)

    scenes = find_scenes(captures_root)
    if not scenes:
        log.warning(f"No scene dirs with a 5-band TIFF ({' or '.join(TIFF_NAMES)}) "
                    f"under {captures_root}")
        log.warning("Co-registration happens upstream — this tool only reads "
                    "already-co-registered scenes.")
        sys.exit(0)

    backend, params = make_backend(args.backend, args.threshold)
    log.info(f"Found {len(scenes)} scene(s) under {captures_root}")
    log.info(f"Backend: {backend.name}" + (f"  params={params}" if params else ""))

    n_ok = n_err = 0
    for scene_dir, tiff_path in scenes:
        key = str(scene_dir)
        row = manifest.get(key, {
            "scene_dir": key,
            "tiff_path": "",
            "autolabel_status": "",
            "mask_path": "",
            "water_pct": "",
            "review_status": "pending",
            "updated": "",
        })

        log.info(f"Processing: {scene_dir.name}")
        row["tiff_path"] = str(tiff_path)

        status, mask_path, water_pct = run_autolabel(
            backend, params, scene_dir, tiff_path, args.force, args.preview)
        row["autolabel_status"] = status
        row["mask_path"] = str(mask_path) if mask_path else ""
        row["water_pct"] = f"{water_pct:.1f}" if status == "ok" else ""
        row["updated"] = _now()
        manifest[key] = row

        if status == "ok":
            n_ok += 1
        else:
            n_err += 1

        save_manifest(manifest_path, manifest)

    log.info(f"\nDone. {n_ok} succeeded, {n_err} failed.")
    log.info(f"Manifest: {manifest_path}")
    log.info("Next step: python review.py <captures_dir>   (or label in the annotator)")


if __name__ == "__main__":
    main()
