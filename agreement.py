#!/usr/bin/env python
"""Inter-annotator agreement over the annotator's per-labeler masks.

Reads the annotator's append-only ``work/label_log.csv`` plus the per-annotator
mask copies it points at (``work/masks/<scene-id>/<annotator>.png``) and reports
how much two people labeling the same scene actually agree.

Why not raw pixel accuracy: water is a small fraction of most frames (~7% in
the bundled examples), so "predict background everywhere" already scores ~93%.
Every metric here is chosen to survive that imbalance:

    IoU              Jaccard on the water class — comparable to model mIoU
    Dice             more forgiving on small water bodies
    Cohen's kappa    chance-corrected, the honest headline under imbalance
    boundary F1      agreement on the waterline within a tolerance, which
                     separates "we disagree where the edge is" (expected, and
                     harmless — the guide says not to pixel-polish) from
                     "we disagree whether that puddle is water" (a real
                     guideline failure worth adjudicating)
    water-% bias     per annotator, to catch someone systematically over-calling

Human agreement is also the **noise ceiling** for the model: a SegFormer mIoU
above the humans' own IoU on the same scenes is measuring label noise, not
skill. Use the replicate set as a hand-labelled val set and report both.

Usage:
    python agreement.py [--label-log PATH] [--csv OUT.csv] [--tolerance 3]
    python agreement.py --label-log annotator/work/label_log.csv --top 10
"""

import argparse
import csv
import itertools
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# a decision that produced a mask; 'rejected' and 'empty' have none to compare
MASK_ROUTES = {"auto_accepted", "auto_edited", "manual_from_scratch"}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    tot = a.sum() + b.sum()
    return 1.0 if tot == 0 else float(2 * np.logical_and(a, b).sum() / tot)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Chance-corrected agreement on a binary mask.

    1 = perfect, 0 = no better than two people guessing with the same base
    rates, negative = worse than chance.
    """
    n = a.size
    tp = float(np.logical_and(a, b).sum())
    tn = float(np.logical_and(~a, ~b).sum())
    po = (tp + tn) / n
    pa, pb = a.sum() / n, b.sum() / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1.0 else float((po - pe) / (1 - pe))


def boundary_f1(a: np.ndarray, b: np.ndarray, tol: int = 3) -> float | None:
    """F1 between the two mask boundaries, allowing `tol` pixels of slack.

    None when neither mask has a boundary (both empty or both full), where the
    measure is undefined rather than perfect.
    """
    ea = _edges(a)
    eb = _edges(b)
    if not ea.any() and not eb.any():
        return None
    k = 2 * tol + 1
    kern = np.ones((k, k), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), kern) > 0
    db = cv2.dilate(eb.astype(np.uint8), kern) > 0
    prec = float((ea & db).sum() / ea.sum()) if ea.any() else 0.0
    rec = float((eb & da).sum() / eb.sum()) if eb.any() else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _edges(m: np.ndarray) -> np.ndarray:
    u = m.astype(np.uint8)
    return (cv2.morphologyEx(u, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def find_label_log(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    for c in (HERE / "annotator" / "work" / "label_log.csv",
              HERE / "work" / "label_log.csv"):
        if c.exists():
            return c
    return None


def latest_per_annotator(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """scene_id -> annotator -> that annotator's most recent decision."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        sid = (r.get("scene_id") or "").strip()
        who = (r.get("annotator") or "").strip()
        if not sid or not who:
            continue
        prev = out[sid].get(who)
        if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
            out[sid][who] = r
    return out


def load_mask(row: dict, shape: tuple[int, int] | None) -> np.ndarray | None:
    path = (row.get("mask_path") or "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if shape and m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-log", default=None,
                    help="Path to the annotator's work/label_log.csv")
    ap.add_argument("--tolerance", type=int, default=3,
                    help="Boundary-F1 slack in pixels. Default: 3")
    ap.add_argument("--csv", default=None, help="Write the per-pair table here")
    ap.add_argument("--top", type=int, default=10,
                    help="How many worst-disagreement scenes to list. Default: 10")
    args = ap.parse_args()

    log_path = find_label_log(args.label_log)
    if not log_path:
        log.error("label_log.csv not found. Pass --label-log "
                  "annotator/work/label_log.csv")
        sys.exit(1)
    with log_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        log.error(f"{log_path} is empty — nothing labeled yet.")
        sys.exit(1)
    log.info(f"Label log: {log_path}  ({len(rows)} decisions)")

    if "mask_path" not in (rows[0].keys()):
        log.error("This log predates per-annotator mask copies (no mask_path column), "
                  "so only the last mask per scene still exists on disk and agreement "
                  "cannot be computed retroactively. New labels will record it.")
        sys.exit(1)

    by_scene = latest_per_annotator(rows)
    multi = {sid: d for sid, d in by_scene.items() if len(d) > 1}
    log.info(f"Scenes labeled by >1 annotator: {len(multi)} of {len(by_scene)}")
    if not multi:
        log.warning("No double-labeled scenes yet. Turn on the agreement sample: set "
                    "\"replicate_pct\": 12 in annotator_config.json and have a second "
                    "person work through the queue.")
        sys.exit(0)

    pairs: list[dict] = []
    disagreed_reject: list[str] = []
    per_annotator: dict[str, list[float]] = defaultdict(list)

    for sid, per in sorted(multi.items()):
        # a rejection has no mask; disagreeing about usability is its own signal
        routes = {who: (r.get("route") or "") for who, r in per.items()}
        if len({rt in MASK_ROUTES for rt in routes.values()}) > 1:
            disagreed_reject.append(f"{sid}: " + ", ".join(f"{w}={r or '?'}"
                                                           for w, r in routes.items()))
        usable = {w: r for w, r in per.items() if routes[w] in MASK_ROUTES}
        if len(usable) < 2:
            continue

        masks: dict[str, np.ndarray] = {}
        shape = None
        for who, row in usable.items():
            m = load_mask(row, shape)
            if m is None:
                log.warning(f"  {sid}: no mask on disk for {who} — skipped")
                continue
            shape = shape or m.shape
            masks[who] = m
        if len(masks) < 2:
            continue

        for a, b in itertools.combinations(sorted(masks), 2):
            ma, mb = masks[a], masks[b]
            rec = {
                "scene_id": sid,
                "a": a, "b": b,
                "iou": round(iou(ma, mb), 4),
                "dice": round(dice(ma, mb), 4),
                "kappa": round(cohen_kappa(ma, mb), 4),
                "boundary_f1": (lambda v: None if v is None else round(v, 4))(
                    boundary_f1(ma, mb, args.tolerance)),
                "water_pct_a": round(100 * float(ma.mean()), 3),
                "water_pct_b": round(100 * float(mb.mean()), 3),
            }
            pairs.append(rec)
            per_annotator[a].append(rec["water_pct_a"])
            per_annotator[b].append(rec["water_pct_b"])

    if not pairs:
        log.warning("No comparable mask pairs (the overlapping decisions were "
                    "rejections, or the mask files are gone).")
        sys.exit(0)

    ious = [p["iou"] for p in pairs]
    kappas = [p["kappa"] for p in pairs]
    bf = [p["boundary_f1"] for p in pairs if p["boundary_f1"] is not None]

    print(f"\n=== Inter-annotator agreement — {len(pairs)} pair(s) "
          f"over {len({p['scene_id'] for p in pairs})} scene(s) ===")
    print(f"  IoU          mean {np.mean(ious):.3f}   median {np.median(ious):.3f}   "
          f"min {min(ious):.3f}")
    print(f"  Cohen kappa  mean {np.mean(kappas):.3f}   median {np.median(kappas):.3f}")
    if bf:
        print(f"  boundary F1  mean {np.mean(bf):.3f}  (±{args.tolerance} px slack)")
    print(f"\n  Read together: high boundary F1 with lower IoU means you disagree about")
    print(f"  the waterline, which is expected. Low boundary F1 means you disagree about")
    print(f"  what counts as water — that is a LABELING_GUIDE.md problem, not sloppiness.")

    print("\n  Per-annotator mean water %, on shared scenes (systematic bias):")
    for who, vals in sorted(per_annotator.items()):
        print(f"    {who:<20s} {np.mean(vals):6.2f} %   (n={len(vals)})")

    worst = sorted(pairs, key=lambda p: p["iou"])[:args.top]
    print(f"\n  Worst {len(worst)} disagreement(s) — adjudicate these first:")
    for p in worst:
        bfs = "n/a" if p["boundary_f1"] is None else f"{p['boundary_f1']:.3f}"
        print(f"    IoU {p['iou']:.3f}  kappa {p['kappa']:6.3f}  bF1 {bfs}  "
              f"{p['scene_id']}  ({p['a']} {p['water_pct_a']}% vs {p['b']} {p['water_pct_b']}%)")

    if disagreed_reject:
        print(f"\n  Disagreed on usability itself ({len(disagreed_reject)}):")
        for line in disagreed_reject[:args.top]:
            print(f"    {line}")

    print(f"\n  Model ceiling: a SegFormer mIoU above {np.mean(ious):.3f} on scenes like")
    print("  these is measuring label noise, not skill. Hold the double-labeled")
    print("  scenes out of training and use them as the val set.")

    if args.csv:
        out = Path(args.csv).resolve()
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
        print(f"\n  Per-pair table: {out}")


if __name__ == "__main__":
    main()
