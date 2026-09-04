#!/usr/bin/env python
"""Interactive CLI review tool for auto-generated water segmentation masks.

Iterates pending scenes in manifest.csv and prompts for approval, rejection,
or flagging for manual correction. Opens preview images in the system viewer
when available.

Usage:
    python review.py <captures_dir> [options]

    python review.py /data/captures/
    python review.py /data/captures/ --no-viewer    # terminal-only, no image viewer
    python review.py /data/captures/ --status pending rejected  # re-review by status
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = [
    "scene_dir", "tiff_path", "autolabel_status",
    "mask_path", "water_pct", "review_status", "updated",
]

VALID_ACTIONS = {
    "a": "approved",
    "r": "rejected",
    "m": "manual",   # flagged for manual correction in LabelMe/CVAT
    "s": "skip",
    "q": "quit",
}

ACTION_HELP = "[a]pprove  [r]eject  [m]anual correction needed  [s]kip  [q]uit"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["scene_dir"]: row for row in csv.DictReader(f)}


def save_manifest(path: Path, rows: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())


def open_image(path: Path) -> None:
    """Try to open an image in the system viewer; print path as fallback."""
    print(f"  Preview: {path}")
    for viewer in ["feh", "eog", "gwenview", "xdg-open", "display"]:
        try:
            proc = subprocess.Popen([viewer, str(path)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Give it a moment to fail fast (e.g. missing display)
            time.sleep(0.3)
            if proc.poll() is None:
                return  # still running → opened successfully
        except FileNotFoundError:
            continue
    print("  (could not open viewer — open the path above manually)")


def make_preview(scene_dir: Path, mask_path: Path) -> Path | None:
    """Generate a side-by-side preview from registered.jpg + mask overlay.

    Returns the path to the written preview, or None if inputs are missing.
    """
    preview_path = scene_dir / "water_mask_preview.jpg"
    if preview_path.exists():
        return preview_path

    # Find best background: prefer false_color_composite, fall back to registered.jpg
    bg_path = None
    for candidate in ("false_color_composite.jpg", "registered.jpg"):
        p = scene_dir / candidate
        if p.exists():
            bg_path = p
            break
    if bg_path is None or mask_path is None or not mask_path.exists():
        return bg_path  # open whatever we have, even without overlay

    try:
        import cv2
        import numpy as np
        bgr = cv2.imread(str(bg_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            return bg_path
        mask_resized = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        overlay = bgr.copy()
        overlay[mask_resized > 127] = [204, 0, 102]  # magenta = water
        combined = np.hstack([bgr, cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)])
        cv2.imwrite(str(preview_path), combined)
        return preview_path
    except Exception:
        return bg_path


def print_scene_info(row: dict, index: int, total: int) -> None:
    scene = Path(row["scene_dir"]).name
    water = row.get("water_pct", "?")
    mask = Path(row["mask_path"]).name if row.get("mask_path") else "—"
    print(f"\n─── Scene {index}/{total}: {scene} ───")
    print(f"  Water coverage : {water}%")
    print(f"  Mask file      : {mask}")
    print(f"  Review status  : {row.get('review_status', '?')}")


def prompt_action() -> str:
    while True:
        print(f"  {ACTION_HELP}")
        raw = input("  > ").strip().lower()
        if raw in VALID_ACTIONS:
            return raw
        print("  Invalid input. Choose one of: a r m s q")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures_dir", help="Root directory containing manifest.csv")
    parser.add_argument("--no-viewer", action="store_true", dest="no_viewer",
                        help="Don't open preview images in system viewer")
    parser.add_argument("--status", nargs="+", default=["pending"],
                        metavar="STATUS",
                        help="Review scenes with these statuses. Default: pending. "
                             "Options: pending approved rejected manual")
    args = parser.parse_args()

    captures_root = Path(args.captures_dir).resolve()
    manifest_path = captures_root / MANIFEST_NAME

    if not manifest_path.exists():
        log.error(f"manifest.csv not found at {manifest_path}")
        log.error("Run pipeline.py first to generate masks.")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    target_statuses = set(args.status)

    queue = [
        row for row in manifest.values()
        if row.get("review_status", "pending") in target_statuses
        and row.get("autolabel_status") == "ok"
    ]

    if not queue:
        print(f"No scenes with status {target_statuses} and successful auto-labeling.")
        sys.exit(0)

    print(f"\nReviewing {len(queue)} scene(s) with status: {target_statuses}")
    print("For each scene the preview image will open (if --no-viewer is not set).")
    print("The preview shows: false-color composite | water mask overlay (magenta = water)\n")

    approved = rejected = manual = skipped = 0

    for i, row in enumerate(queue, 1):
        print_scene_info(row, i, len(queue))

        if not args.no_viewer:
            scene_dir = Path(row["scene_dir"])
            mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
            img = make_preview(scene_dir, mask_path)
            if img:
                open_image(img)

        action = prompt_action()

        if action == "q":
            print("\nReview paused. Re-run to continue.")
            break
        elif action == "s":
            skipped += 1
            continue

        new_status = VALID_ACTIONS[action]
        row["review_status"] = new_status
        manifest[row["scene_dir"]] = row
        save_manifest(manifest_path, manifest)

        if action == "a":
            approved += 1
        elif action == "r":
            rejected += 1
        elif action == "m":
            manual += 1
            mask_path = Path(row["mask_path"])
            print("  → Flag for manual correction. Edit mask at:")
            print(f"    {mask_path}")
            print("    Save corrected mask as 'water_mask.png' in the same directory.")
            print("    Then set review_status to 'approved' in manifest.csv.")

    print("\nReview session summary:")
    print(f"  Approved : {approved}")
    print(f"  Rejected : {rejected}")
    print(f"  Manual   : {manual}")
    print(f"  Skipped  : {skipped}")
    print("\nNext step: python export_dataset.py <captures_dir> <output_dataset_dir>")


if __name__ == "__main__":
    main()
