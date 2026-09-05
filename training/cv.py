"""Leave-one-session-out cross-validation, run and aggregated.

With a couple of dozen labelled scenes spread over a handful of capture
sessions, a single held-out split is mostly noise — and if it is not grouped by
session it is worse than noise, because the rig's repeated captures put
near-duplicate frames on both sides (see `manifest.py`). This runs every fold
and reports the spread, which is the number worth quoting.

    uv run --project annotator python -m training.cv --modality fiveband --epochs 40

Unrecognised arguments forward to `training.train`, so anything that works
there works here.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import models
from . import modalities as M

ROOT = Path(__file__).resolve().parent.parent


def folds_in(manifest: Path) -> dict[str, set[str]]:
    """{fold index: held-out group names} straight from the manifest."""
    out: dict[str, set[str]] = {}
    with Path(manifest).open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("fold", "-1") != "-1":
                out.setdefault(r["fold"], set()).add(r.get("group", r["stem"]))
    return out


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("miou") is not None]
    if not ok:
        return {"folds": results, "n_folds": 0}
    v = np.array([r["miou"] for r in ok], float)
    n = np.array([r["n_val"] for r in ok], float)
    agg = {
        "folds": results,
        "n_folds": len(ok),
        "n_scenes": int(n.sum()),
        "mean_miou": round(float(v.mean()), 4),
        "std_miou": round(float(v.std()), 4),
        "min_miou": round(float(v.min()), 4),
        "max_miou": round(float(v.max()), 4),
        # a fold holding 8 scenes should not count the same as one holding 1
        "scene_weighted_miou": round(float((v * n).sum() / n.sum()), 4),
    }
    for key in ("iou_water", "boundary_f1"):
        vals = np.array([r.get(key) or 0.0 for r in ok], float)
        agg[f"mean_{key}"] = round(float(vals.mean()), 4)
        agg[f"std_{key}"] = round(float(vals.std()), 4)
        agg[f"scene_weighted_{key}"] = round(float((vals * n).sum() / n.sum()), 4)
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("annotator/work/scenes.csv"))
    ap.add_argument("--out", type=Path, default=Path("annotator/work/runs"))
    ap.add_argument("--modality", default="fiveband", choices=list(M.MODALITIES))
    ap.add_argument("--arch", default="segformer-b0", choices=models.ARCHS)
    ap.add_argument("--tag", default=None, help="run dir prefix (default: modality-arch)")
    a, extra = ap.parse_known_args()

    groups = folds_in(a.manifest)
    if not groups:
        raise SystemExit(f"{a.manifest} has no fold assignments — rebuild it with "
                         f"`python -m training.manifest --folds 5`")
    tag = a.tag or f"{a.modality}-{a.arch}"
    results = []
    for k in sorted(groups, key=int):
        run = a.out / f"cv-{tag}-{k}"
        print(f"[fold {k}] holding out: {', '.join(sorted(groups[k]))}", flush=True)
        cmd = [sys.executable, "-m", "training.train", "--manifest", str(a.manifest),
               "--modality", a.modality, "--arch", a.arch, "--fold", str(k),
               "--out", str(run), *extra]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        rec = {"fold": k, "held_out": sorted(groups[k])}
        mj = run / "metrics.json"
        if p.returncode != 0 or not mj.exists():
            rec["error"] = (p.stderr or p.stdout or "").strip().splitlines()[-2:]
            print(f"    FAILED: {rec['error']}", flush=True)
        else:
            m = json.loads(mj.read_text())
            best = max(m["history"], key=lambda h: h["miou"], default={})
            rec |= {"miou": best.get("miou"), "iou_water": best.get("iou_water"),
                    "boundary_f1": best.get("boundary_f1"),
                    "n_val": m["config"]["n_val"], "n_train": m["config"]["n_train"],
                    "minutes": m.get("minutes")}
            print(f"    mIoU {rec['miou']}  water IoU {rec['iou_water']}  "
                  f"(train {rec['n_train']}, val {rec['n_val']})", flush=True)
        results.append(rec)

    agg = aggregate(results)
    agg |= {"modality": a.modality, "arch": a.arch}
    out = a.out / f"cv-{tag}.json"
    out.write_text(json.dumps(agg, indent=2))

    print(f"\n{'fold':>4}  {'held-out':34s} {'val':>4} {'mIoU':>7} {'water':>7} {'bF1':>6}")
    for r in results:
        if r.get("miou") is None:
            print(f"{r['fold']:>4}  {', '.join(r['held_out'])[:34]:34s}   —  FAILED")
            continue
        print(f"{r['fold']:>4}  {', '.join(r['held_out'])[:34]:34s} {r['n_val']:>4} "
              f"{r['miou']:>7.4f} {r['iou_water']:>7.4f} {(r['boundary_f1'] or 0):>6.3f}")
    if agg["n_folds"]:
        print(f"\n{a.modality} / {a.arch}: mean mIoU {agg['mean_miou']} "
              f"± {agg['std_miou']} (min {agg['min_miou']}, max {agg['max_miou']}), "
              f"scene-weighted {agg['scene_weighted_miou']}, "
              f"over {agg['n_scenes']} scenes in {agg['n_folds']} folds")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
