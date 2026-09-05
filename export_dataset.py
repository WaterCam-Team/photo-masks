#!/usr/bin/env python
"""Export reviewed scenes to MMSeg / SegFormer training format.

Output: dataset_dir/
            img_dir/train/   ← color_preserved_5_band.tiff (copied)
            img_dir/val/     ← color_preserved_5_band.tiff (copied)
            ann_dir/train/   ← single-channel PNG, 0=bg 1=water
            ann_dir/val/     ← single-channel PNG, 0=bg 1=water

Two selection modes:

  default (manifest-driven)
      python export_dataset.py <captures_dir> <dataset_dir>
      Exports manifest.csv rows with review_status 'approved' (or 'manual'
      with a water_mask.png present). Mask priority: water_mask.png, then the
      row's auto mask_path.  May include unverified auto masks.

  --gold-only  (Tier 1: human-verified labels only)
      python export_dataset.py --gold-only <dataset_dir> [--label-log PATH]
      Reads the annotator's work/label_log.csv and exports ONLY scenes whose
      latest decision was made by a human:
          route in {auto_accepted, auto_edited, interactive_clicks,
                    manual_from_scratch}
      i.e. an accepted/edited SAM (or other) seed, a SAM2 click session the
      human steered, or a hand-drawn mask.
      Never falls back to water_mask_auto.png. 'rejected' / unlabelled scenes
      are excluded. Writes dataset_provenance.csv (route, seed_backend,
      auto_vs_final_iou, edited_pixel_frac, active_seconds, annotator, time)
      so the training set has an audit trail. Train/val split is by a stable
      hash of the scene name, so it doesn't reshuffle as labels accumulate.
      Scenes labeled by more than one person go to val instead: they are the
      only ones whose human-human agreement is measurable, which makes them
      the honest validation set. See agreement.py; --replicates-in-train opts
      out.

Usage examples:
    python export_dataset.py /data/captures/ /path/to/5band_data/
    python export_dataset.py --gold-only ./gold_dataset/ --val-split 0.15 --dry-run
"""

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.csv"
TIFF_NAMES = ("color_preserved_5_band.tiff", "final_5_band.tiff")

# decisions that mean "a human looked at this mask and kept it"
GOLD_ROUTES = {"auto_accepted", "auto_edited", "interactive_clicks",
               "manual_from_scratch"}

HERE = Path(__file__).resolve().parent
PROVENANCE_FIELDS = [
    "split", "stem", "scene_dir", "route", "seed_backend", "n_annotators",
    "auto_vs_final_iou", "edited_pixel_frac", "active_seconds",
    "ensemble_agreement", "ensemble_disagreement_frac",
    "annotator", "timestamp",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def find_tiff(scene_dir: Path) -> Path | None:
    for n in TIFF_NAMES:
        p = scene_dir / n
        if p.exists():
            return p
    return None


def convert_mask(src: Path, dst: Path) -> None:
    """Convert a binary mask (0/255) to a class-index mask (0/1) for MMSeg."""
    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {src}")
    cv2.imwrite(str(dst), (mask > 127).astype(np.uint8))


def scene_stem(item: dict) -> str:
    return Path(item["scene_dir"]).name


def hash_split(stem: str, val_split: float) -> str:
    """Deterministic per-scene split; stable as the dataset grows.

    usedforsecurity=False so this keeps working under a FIPS-enabled kernel,
    where the plain md5 constructor raises.
    """
    h = int(hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest(), 16) % 10000
    return "val" if h < val_split * 10000 else "train"


# ---------------------------------------------------------------------------
# Default mode: manifest-driven
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        log.error(f"manifest.csv not found: {path}")
        sys.exit(1)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_approved(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        status = row.get("review_status", "")
        if status == "approved":
            out.append(row)
        elif status == "manual" and (Path(row["scene_dir"]) / "water_mask.png").exists():
            out.append(row)
    return out


def resolve_mask_default(item: dict) -> Path | None:
    scene_dir = Path(item["scene_dir"])
    manual = scene_dir / "water_mask.png"
    if manual.exists():
        return manual
    auto = item.get("mask_path", "")
    if auto and Path(auto).exists():
        return Path(auto)
    return None


# ---------------------------------------------------------------------------
# --gold-only mode: label-log-driven, human-verified masks only
# ---------------------------------------------------------------------------

def find_label_log(captures_root: Path | None) -> Path | None:
    cands = [
        HERE / "annotator" / "work" / "label_log.csv",
        HERE.parent / "annotator" / "work" / "label_log.csv",
    ]
    if captures_root:
        cands = [
            captures_root / "label_log.csv",
            captures_root / "work" / "label_log.csv",
            *cands,
        ]
    return next((c for c in cands if c.exists()), None)


def load_label_log(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_gold(log_rows: list[dict]) -> list[dict]:
    """Latest decision per scene; keep only human-verified routes with a mask."""
    latest: dict[str, dict] = {}
    annotators: dict[str, set] = {}
    for r in log_rows:
        raw = (r.get("scene_dir") or "").strip()
        if not raw:                       # check before resolve(): Path("") is the cwd
            continue
        sd = str(Path(raw).resolve())
        who = (r.get("annotator") or "").strip()
        if who:
            annotators.setdefault(sd, set()).add(who)
        prev = latest.get(sd)
        if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[sd] = r

    items: list[dict] = []
    skipped_route = skipped_missing = 0
    for sd, r in latest.items():
        if r.get("route", "") not in GOLD_ROUTES:
            skipped_route += 1
            continue
        scene_dir = Path(sd)
        tiff = find_tiff(scene_dir)
        mask = scene_dir / "water_mask.png"
        if tiff is None or not mask.exists():
            skipped_missing += 1
            log.warning(f"  gold scene missing {'TIFF' if tiff is None else 'water_mask.png'}: "
                        f"{scene_dir.name}")
            continue
        items.append({
            "scene_dir": str(scene_dir),
            "tiff_path": str(tiff),
            "mask_path": "",                    # force water_mask.png, never the auto mask
            "route": r.get("route", ""),
            "seed_backend": r.get("seed_backend", ""),
            "auto_vs_final_iou": r.get("auto_vs_final_iou", ""),
            "edited_pixel_frac": r.get("edited_pixel_frac", ""),
            "active_seconds": r.get("active_seconds", ""),
            "ensemble_agreement": r.get("ensemble_agreement", ""),
            "ensemble_disagreement_frac": r.get("ensemble_disagreement_frac", ""),
            "annotator": r.get("annotator", ""),
            "timestamp": r.get("timestamp", ""),
            "n_annotators": len(annotators.get(sd, set())) or 1,
        })
    if skipped_route:
        log.info(f"  excluded {skipped_route} scene(s): not a human-verified route "
                 f"(rejected / empty / unlabelled)")
    if skipped_missing:
        log.info(f"  excluded {skipped_missing} scene(s): missing TIFF or water_mask.png")
    return sorted(items, key=lambda x: Path(x["scene_dir"]).name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("captures_dir", nargs="?", default=None,
                   help="Directory with manifest.csv (default mode). "
                        "In --gold-only mode: optional, only used to locate label_log.csv.")
    p.add_argument("dataset_dir", help="Output dataset directory (created/updated)")
    p.add_argument("--gold-only", action="store_true", dest="gold_only",
                   help="Export only human-verified masks, driven by the annotator's label_log.csv")
    p.add_argument("--label-log", default=None, dest="label_log",
                   help="Path to the annotator's work/label_log.csv (--gold-only)")
    p.add_argument("--val-split", type=float, default=0.2, dest="val_split",
                   help="Fraction of scenes for validation. Default: 0.2")
    p.add_argument("--replicates-in-train", action="store_true", dest="replicates_in_train",
                   help="Let double-labeled scenes be split normally. By default they "
                        "all go to val: they are the only scenes whose human-human "
                        "agreement is known, which makes them the honest val set (and "
                        "the ceiling your mIoU should be read against — see agreement.py).")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print what would be done without writing files")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    captures_root = Path(args.captures_dir).resolve() if args.captures_dir else None

    # -- select scenes -------------------------------------------------
    if args.gold_only:
        log_path = Path(args.label_log).resolve() if args.label_log else find_label_log(captures_root)
        if not log_path or not log_path.exists():
            log.error("label_log.csv not found. Pass --label-log /path/to/annotator/work/label_log.csv")
            sys.exit(1)
        log.info(f"Gold-only export — label log: {log_path}")
        items = collect_gold(load_label_log(log_path))
        if not items:
            log.error("No gold (human-verified) scenes in the label log.")
            sys.exit(1)
        n_rep = 0
        for it in items:
            replicated = int(it.get("n_annotators", 1) or 1) > 1
            if replicated and not args.replicates_in_train:
                it["_split"] = "val"
                n_rep += 1
            else:
                it["_split"] = hash_split(scene_stem(it), args.val_split)
        if n_rep:
            log.info(f"  {n_rep} double-labeled scene(s) held out as val "
                     f"(their mask is the most recent annotator's — adjudicate "
                     f"disagreements first; see agreement.py)")
        resolve_mask = lambda it: Path(it["scene_dir"]) / "water_mask.png"  # noqa: E731
        route_counts = Counter(it["route"] for it in items)
        log.info(f"Gold scenes: {len(items)}  routes={dict(route_counts)}")
    else:
        if captures_root is None:
            log.error("captures_dir is required (or use --gold-only).")
            sys.exit(1)
        items = collect_approved(load_manifest(captures_root / MANIFEST_NAME))
        if not items:
            log.error("No approved scenes in manifest.csv. Run review/annotator first.")
            sys.exit(1)
        n_val = max(1, round(len(items) * args.val_split)) if len(items) > 1 else 0
        for i, it in enumerate(items):
            it["_split"] = "train" if i < len(items) - n_val else "val"
        resolve_mask = resolve_mask_default

    n_train = sum(1 for it in items if it["_split"] == "train")
    n_val = sum(1 for it in items if it["_split"] == "val")
    log.info(f"Total {len(items)}  (train={n_train}, val={n_val})")

    # -- write --------------------------------------------------------
    exported: list[dict] = []
    for split in ("train", "val"):
        img_out = dataset_dir / "img_dir" / split
        ann_out = dataset_dir / "ann_dir" / split
        if not args.dry_run:
            img_out.mkdir(parents=True, exist_ok=True)
            ann_out.mkdir(parents=True, exist_ok=True)

        for it in (x for x in items if x["_split"] == split):
            stem = scene_stem(it)
            tiff_src = Path(it.get("tiff_path", "") or "")
            mask_src = resolve_mask(it)

            if not tiff_src.exists():
                log.warning(f"  [{split}] {stem}: missing TIFF, skipping")
                continue
            if mask_src is None or not mask_src.exists():
                log.warning(f"  [{split}] {stem}: no mask, skipping")
                continue

            mask_kind = "manual" if mask_src.name == "water_mask.png" else "auto"
            extra = f"  route={it['route']}" if "route" in it else ""
            log.info(f"  [{split}] {stem}  mask={mask_kind}{extra}")

            if not args.dry_run:
                shutil.copy2(tiff_src, img_out / f"{stem}.tiff")
                convert_mask(mask_src, ann_out / f"{stem}.png")

            row = {"split": split, "stem": stem}
            row.update({k: it.get(k, "") for k in PROVENANCE_FIELDS if k not in row})
            row["scene_dir"] = it["scene_dir"]
            exported.append(row)

    # -- provenance (gold mode) -------------------------------------
    if args.gold_only and not args.dry_run and exported:
        prov = dataset_dir / "dataset_provenance.csv"
        with prov.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(exported)
        log.info(f"Provenance: {prov}")

    # -- summary --------------------------------------------------
    if args.dry_run:
        log.info("\nDry run — no files written.")
        return
    ex_train = sum(1 for r in exported if r["split"] == "train")
    ex_val = sum(1 for r in exported if r["split"] == "val")
    log.info(f"\nDataset written to: {dataset_dir}")
    log.info(f"  img_dir/train  {ex_train} TIFFs   ann_dir/train  {ex_train} masks")
    log.info(f"  img_dir/val    {ex_val} TIFFs   ann_dir/val    {ex_val} masks")
    log.info(f"\nSet data_root = '{dataset_dir}/' in your SegFormer config.")
    if args.gold_only:
        log.info("These are Tier-1 gold labels (human-verified). Fine-tune from the "
                 "ADE20K B0 checkpoint; keep a hand-labelled val set for real mIoU.")


if __name__ == "__main__":
    main()
