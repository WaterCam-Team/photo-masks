"""Run a modality x architecture matrix and tabulate it.

This is the sensor-value question the capture rig exists to answer — does
5-band fusion actually beat plain RGB, or thermal, for finding water? — run on
the annotator's human-verified masks rather than on `water_autolabel.py`
thresholds.

Every modality is sampled from the *same* co-registered pixels (see
`modalities.py`), so a difference in the table is a difference in sensor
information, not in resampling, alignment or mask geometry. Each cell trains
with identical hyperparameters; only the input channels change.

    uv run --project annotator python -m training.compare \
        --manifest annotator/work/scenes.csv \
        --modalities rgb_std rgb_nofilt lwir fiveband \
        --archs segformer-b0 --epochs 60 --out annotator/work/compare

Each cell runs as its own subprocess, so one architecture failing to converge
(or running out of memory) leaves the rest of the matrix intact and recorded as
an error rather than taking the whole sweep down.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import modalities as M
from . import models

METRIC_COLUMNS = ("miou", "iou_water", "f1_water", "precision_water",
                  "recall_water", "boundary_f1")

#: what a --cv cell reports instead: a mean over folds, not one split
CV_COLUMNS = ("mean_miou", "std_miou", "scene_weighted_miou", "mean_iou_water",
              "mean_boundary_f1", "min_miou", "max_miou")


def cell_dir(out: Path, modality: str, arch: str) -> Path:
    return Path(out) / f"{modality}__{arch}"


def run_cell(manifest: Path, modality: str, arch: str, out: Path,
             extra: list[str], dry: bool = False) -> dict:
    d = cell_dir(out, modality, arch)
    cmd = [sys.executable, "-m", "training.train", "--manifest", str(manifest),
           "--modality", modality, "--arch", arch, "--out", str(d), *extra]
    if dry:
        return {"modality": modality, "arch": arch, "cmd": " ".join(cmd)}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent),
                       capture_output=True, text=True)
    rec: dict = {"modality": modality, "arch": arch,
                 "minutes": round((time.time() - t0) / 60, 2)}
    mj = d / "metrics.json"
    if p.returncode != 0 or not mj.exists():
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
        rec["error"] = " / ".join(tail)[-400:] or f"exit {p.returncode}"
        return rec
    m = json.loads(mj.read_text())
    rec["best_val_miou"] = m.get("best_miou")
    hist = m.get("history") or []
    best = max(hist, key=lambda h: h.get("miou", -1), default={})
    for k in METRIC_COLUMNS:
        rec[f"val_{k}"] = best.get(k)
    for k in METRIC_COLUMNS:
        rec[f"test_{k}"] = (m.get("test") or {}).get(k)
    cfg = m.get("config", {})
    rec["n_train"] = cfg.get("n_train")
    rec["n_val"] = cfg.get("n_val")
    rec["n_test"] = cfg.get("n_test")
    rec["in_channels"] = (cfg.get("stats") or {}).get("channels")
    return rec


def run_cell_cv(manifest: Path, modality: str, arch: str, out: Path,
                extra: list[str], dry: bool = False) -> dict:
    """One cell = a full leave-one-session-out sweep for that modality."""
    tag = f"{modality}__{arch}"
    cmd = [sys.executable, "-m", "training.cv", "--manifest", str(manifest),
           "--modality", modality, "--arch", arch, "--out", str(out),
           "--tag", tag, *extra]
    if dry:
        return {"modality": modality, "arch": arch, "cmd": " ".join(cmd)}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent),
                       capture_output=True, text=True)
    rec: dict = {"modality": modality, "arch": arch,
                 "minutes": round((time.time() - t0) / 60, 2)}
    agg_path = cv_agg_path(out, modality, arch)
    if p.returncode != 0 or not agg_path.exists():
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
        rec["error"] = " / ".join(tail)[-400:] or f"exit {p.returncode}"
        return rec
    return rec | cv_rec_from(agg_path)


def cv_agg_path(out: Path, modality: str, arch: str) -> Path:
    return Path(out) / f"cv-{modality}__{arch}.json"


def cv_rec_from(agg_path: Path) -> dict:
    """Read a finished CV sweep's aggregate — also how --skip-existing reuses one."""
    agg = json.loads(Path(agg_path).read_text())
    rec = {k: agg.get(k) for k in CV_COLUMNS}
    rec["n_folds"] = agg.get("n_folds")
    rec["n_scenes"] = agg.get("n_scenes")
    rec["per_fold"] = [{"held_out": f.get("held_out"), "miou": f.get("miou")}
                       for f in agg.get("folds", [])]
    return rec


def cv_to_markdown(rows: list[dict]) -> str:
    head = ["modality", "arch", "mean mIoU", "±", "scene-wtd", "water IoU",
            "bF1", "min", "max", "min"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    ok = [r for r in rows if "error" not in r and r.get("mean_miou") is not None]
    for r in sorted(ok, key=lambda r: -(r["mean_miou"])):
        lines.append("| " + " | ".join([
            r["modality"], r["arch"],
            f"{r['mean_miou']:.4f}", f"{r['std_miou']:.4f}",
            f"{r['scene_weighted_miou']:.4f}", f"{r['mean_iou_water']:.4f}",
            f"{r['mean_boundary_f1']:.3f}", f"{r['min_miou']:.3f}",
            f"{r['max_miou']:.3f}", str(r.get("minutes", ""))]) + " |")
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['modality']} | {r['arch']} | FAILED |"
                         + " |" * (len(head) - 3))
    return "\n".join(lines)


def paired_markdown(rows: list[dict], baseline: str | None = None) -> str:
    """Per-fold paired comparison against a baseline modality.

    With five capture sessions the between-session spread (~0.22 mIoU) is far
    larger than the gaps between modalities, so comparing means alone says
    little. Every cell ran on the *same* folds, though, so differencing per
    fold cancels "this session is hard" and leaves the effect of the input
    channels. Read the win count and the size of the losses, not just the mean.
    """
    ok = [r for r in rows if r.get("per_fold")]
    if len(ok) < 2:
        return ""
    base = baseline or max(ok, key=lambda r: r.get("mean_miou") or -1)["modality"]
    ref = {tuple(f["held_out"]): f["miou"] for f in
           next(r for r in ok if r["modality"] == base)["per_fold"]}
    folds = sorted(ref)

    lines = [f"Paired against **{base}**, fold by fold "
             f"(identical splits and hyperparameters):", "",
             "| vs | mean Δ mIoU | " + " | ".join(f[0][:22] for f in folds) + " | wins |",
             "|" + "|".join(["---"] * (len(folds) + 3)) + "|"]
    for r in ok:
        if r["modality"] == base:
            continue
        got = {tuple(f["held_out"]): f["miou"] for f in r["per_fold"]}
        d = [ref[f] - got[f] for f in folds if got.get(f) is not None]
        if not d:
            continue
        cells = " | ".join(f"{x:+.3f}" for x in d)
        lines.append(f"| {r['modality']} | {sum(d)/len(d):+.3f} | {cells} | "
                     f"{sum(1 for x in d if x > 0)}/{len(d)} |")
    return "\n".join(lines)


def to_markdown(rows: list[dict], prefix: str = "val") -> str:
    """A table of the `prefix` metrics, best mIoU first."""
    cols = [f"{prefix}_{k}" for k in METRIC_COLUMNS]
    head = ["modality", "arch", "ch"] + [c.replace(f"{prefix}_", "") for c in cols] + ["min"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    for r in sorted(ok, key=lambda r: -(r.get(f"{prefix}_miou") or -1)):
        cells = [r["modality"], r["arch"], str(r.get("in_channels") or "")]
        for c in cols:
            v = r.get(c)
            cells.append("—" if v is None else f"{v:.4f}")
        cells.append(str(r.get("minutes", "")))
        lines.append("| " + " | ".join(cells) + " |")
    for r in bad:
        lines.append(f"| {r['modality']} | {r['arch']} | | "
                     + " | ".join(["FAILED"] * len(cols)) + " | |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("annotator/work/scenes.csv"))
    ap.add_argument("--out", type=Path, default=Path("annotator/work/compare"))
    ap.add_argument("--modalities", nargs="+", default=list(M.COMPARISON_SET),
                    choices=list(M.MODALITIES))
    ap.add_argument("--archs", nargs="+", default=["segformer-b0"], choices=models.ARCHS)
    ap.add_argument("--cv", action="store_true",
                    help="run leave-one-session-out CV per cell and compare the "
                         "means. Slower (folds x cells) but the only defensible "
                         "form while a single val split is one or two scenes.")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep cells that already have metrics.json (resume a sweep)")
    a, extra = ap.parse_known_args()      # everything else forwards to training.train

    a.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total = len(a.modalities) * len(a.archs)
    n = 0
    for arch in a.archs:
        for modality in a.modalities:
            n += 1
            done = cv_agg_path(a.out, modality, arch) if a.cv else \
                cell_dir(a.out, modality, arch) / "metrics.json"
            if a.skip_existing and done.exists():
                print(f"[{n}/{total}] {modality} x {arch}: keeping existing result",
                      flush=True)
                base = {"modality": modality, "arch": arch}
                rows.append(base | (cv_rec_from(done) if a.cv
                                    else _from_existing(done)))
                continue
            print(f"[{n}/{total}] {modality} x {arch} ...", flush=True)
            runner = run_cell_cv if a.cv else run_cell
            rec = runner(a.manifest, modality, arch, a.out, extra, a.dry_run)
            rows.append(rec)
            if a.dry_run:
                print("    " + rec["cmd"])
            elif "error" in rec:
                print(f"    FAILED: {rec['error']}", flush=True)
            elif a.cv:
                print(f"    mean mIoU {rec['mean_miou']} ± {rec['std_miou']}  "
                      f"water IoU {rec['mean_iou_water']}  "
                      f"({rec['n_folds']} folds, {rec['minutes']} min)", flush=True)
            else:
                print(f"    val mIoU {rec.get('val_miou')}  water IoU "
                      f"{rec.get('val_iou_water')}  ({rec['minutes']} min)", flush=True)

    if a.dry_run:
        return
    import csv
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (a.out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if a.cv:
        md = ["# Modality x architecture comparison "
              "(leave-one-session-out CV)", "",
              f"Manifest: `{a.manifest}`  ", f"Cells: {len(rows)}", "",
              cv_to_markdown(rows), "",
              paired_markdown(rows), "",
              "Per-fold detail:"]
        for r in rows:
            if r.get("per_fold"):
                md.append(f"- **{r['modality']}**: " + ", ".join(
                    f"{', '.join(f['held_out'])}={f['miou']}" for f in r["per_fold"]))
    else:
        md = ["# Modality x architecture comparison", "",
              f"Manifest: `{a.manifest}`  ", f"Cells: {len(rows)}", "",
              "## Validation", "", to_markdown(rows, "val")]
        if any(r.get("test_miou") is not None for r in rows):
            md += ["", "## Held-out test", "", to_markdown(rows, "test")]
    md += ["", "Every modality is read from the same co-registered pixels, so a",
           "difference here is a difference in sensor information rather than in",
           "alignment or mask geometry. Hyperparameters are identical across cells."]
    (a.out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n" + (cv_to_markdown(rows) if a.cv else to_markdown(rows, "val")))
    if a.cv:
        print("\n" + paired_markdown(rows))
    print(f"\nwrote {a.out / 'comparison.csv'} and {a.out / 'comparison.md'}")


def _from_existing(mj: Path) -> dict:
    m = json.loads(mj.read_text())
    hist = m.get("history") or []
    best = max(hist, key=lambda h: h.get("miou", -1), default={})
    rec = {"best_val_miou": m.get("best_miou"), "minutes": m.get("minutes")}
    for k in METRIC_COLUMNS:
        rec[f"val_{k}"] = best.get(k)
        rec[f"test_{k}"] = (m.get("test") or {}).get(k)
    cfg = m.get("config", {})
    rec["in_channels"] = (cfg.get("stats") or {}).get("channels")
    rec.pop("cmd", None)
    return rec


if __name__ == "__main__":
    main()
