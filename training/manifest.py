"""Build a scene manifest for training — pointers, not copies.

`export_dataset.py` copies TIFFs into an `img_dir/ann_dir` tree, which suits a
single-modality run but cannot serve `rgb_nofilt` (the NIR-ON frame is not
copied) and duplicates ~6.6 MB per scene per export. This module instead writes
a CSV of *scene pointers* with a split assignment, so every modality reads from
the original scene directory and no pixel data is duplicated.

Selection is delegated to `export_dataset.collect_gold`, so "gold" means the
same thing in both paths: the latest decision for a scene was made by a human
(`auto_accepted`, `auto_edited`, `manual_from_scratch`) and `water_mask.png`
exists. Auto masks never enter (invariant 4).

Split policy, stable as labels accumulate — a scene keeps its split when the
dataset grows:
  * splits are assigned per **capture session** (the scene's parent directory),
    not per scene. The rig fires repeatedly within a session: `20251229-1427`,
    `-14270`, `-1428`, `-1429` are the same view seconds apart, so splitting
    them individually puts near-duplicate frames on both sides and the val
    score measures memorisation. `--group-by scene` restores the old
    per-scene behaviour when you genuinely have independent scenes.
  * a session containing a scene labelled by more than one person goes to
    **val**: those are the only scenes whose human-human agreement is
    measurable, which is what makes them an honest val set

    uv run --project annotator python -m training.manifest --out work/scenes.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:                 # import the sibling script
    sys.path.insert(0, str(HERE.parent))

import export_dataset as xd                          # noqa: E402

FIELDS = [
    "split", "fold", "group", "stem", "scene_dir", "tiff_path", "mask_path",
    "route", "seed_backend", "n_annotators", "auto_vs_final_iou",
    "edited_pixel_frac", "active_seconds", "annotator", "timestamp",
]


def group_of(scene_dir: Path | str, group_by: str = "session") -> str:
    """The key a split is assigned to: the capture session, or the scene."""
    p = Path(scene_dir)
    return p.name if group_by == "scene" else p.parent.name


def index_labelled_scenes(roots: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """`{(parent name, dir name): [paths]}` for every dir holding a water_mask.png.

    `os.walk(followlinks=True)` with a realpath guard, not `rglob`: a capture
    archive is normally reached through a symlink (`example_data/data`), and
    rglob does not follow those — it would silently find nothing.

    Keyed by `(parent, name)` because that is already this project's notion of
    scene identity (scene IDs are `<parent>__<name>-<hash>`).
    """
    idx: dict[tuple[str, str], list[Path]] = {}
    seen: set[str] = set()
    for root in roots:
        if not Path(root).exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in seen:                    # symlink loop / repeated mount
                dirnames[:] = []
                continue
            seen.add(real)
            if "water_mask.png" in filenames:
                p = Path(dirpath)
                idx.setdefault((p.parent.name, p.name), []).append(p)
    return idx


def relocate(scene_dir: Path, idx: dict[tuple[str, str], list[Path]]) -> Path | None:
    """Find a scene that moved since it was labelled.

    An archive gets re-mounted, or a symlink re-pointed a level up or down, and
    every `scene_dir` in the append-only label log goes stale while the labels
    themselves are perfectly fine on disk. Matching on `(parent, name)` finds
    them again. An ambiguous match returns None rather than guessing — two
    sessions can legitimately hold same-named scene directories.
    """
    hits = idx.get((scene_dir.parent.name, scene_dir.name), [])
    uniq = {os.path.realpath(p): p for p in hits}
    return next(iter(uniq.values())) if len(uniq) == 1 else None


def assign_split(stem: str, n_annotators: int, val_split: float,
                 test_split: float = 0.0) -> str:
    """Deterministic 2- or 3-way split of one group. Replicates -> val.

    `stem` is the group key (a capture session by default), not necessarily a
    scene name — see `group_of`.
    """
    if n_annotators > 1:
        return "val"
    if test_split <= 0:
        return xd.hash_split(stem, val_split)
    h = int(hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest(), 16) % 10000
    if h < test_split * 10000:
        return "test"
    if h < (test_split + val_split) * 10000:
        return "val"
    return "train"


def assign_fold(stem: str, folds: int) -> int:
    """Stable fold index for k-fold cross-validation (-1 when disabled).

    With a couple of dozen labelled scenes a single held-out split is mostly
    noise, so k-fold is the honest way to report a number.
    """
    if folds < 2:
        return -1
    h = int(hashlib.md5(("fold:" + stem).encode(), usedforsecurity=False).hexdigest(), 16)
    return h % folds


def build(label_log: Path | None = None, val_split: float = 0.2,
          test_split: float = 0.0, folds: int = 0,
          search_roots: list[Path] | None = None,
          group_by: str = "session") -> tuple[list[dict], dict]:
    """Gold scenes -> manifest rows, plus a report of what was relocated."""
    log_path = label_log or xd.find_label_log(None)
    if log_path is None or not Path(log_path).exists():
        raise FileNotFoundError(
            "no label_log.csv found — label some scenes in the annotator first "
            "(expected annotator/work/label_log.csv)")
    log_rows = xd.load_label_log(Path(log_path))

    # Repoint rows whose scene_dir has gone stale, before collect_gold drops
    # them for a missing TIFF. The label log is append-only (invariant 5), so
    # it keeps the path a scene had when it was labelled; re-mounting the
    # archive invalidates that path without touching the labels themselves.
    report = {"relocated": [], "ambiguous": [], "missing": []}
    stale = [r for r in log_rows
             if (r.get("scene_dir") or "").strip() and not Path(r["scene_dir"]).exists()]
    if stale:
        idx = index_labelled_scenes(search_roots or [Path("example_data")])
        for r in stale:
            found = relocate(Path(r["scene_dir"]), idx)
            if found is not None:
                report["relocated"].append((r["scene_dir"], str(found)))
                r["scene_dir"] = str(found.resolve())
            elif idx.get((Path(r["scene_dir"]).parent.name, Path(r["scene_dir"]).name)):
                report["ambiguous"].append(r["scene_dir"])
            else:
                report["missing"].append(r["scene_dir"])

    items = xd.collect_gold(log_rows)

    # Assign once per group, then hand every scene its group's verdict, so a
    # session can never straddle the train/val boundary.
    groups: dict[str, int] = {}
    for it in items:
        g = group_of(it["scene_dir"], group_by)
        groups[g] = max(groups.get(g, 1), int(it.get("n_annotators") or 1))
    verdict = {g: (assign_split(g, n, val_split, test_split), assign_fold(g, folds))
               for g, n in groups.items()}

    rows = []
    for it in items:
        stem = xd.scene_stem(it)
        g = group_of(it["scene_dir"], group_by)
        n = int(it.get("n_annotators") or 1)
        rows.append({
            "split": verdict[g][0],
            "fold": verdict[g][1],
            "group": g,
            "stem": stem,
            "scene_dir": it["scene_dir"],
            "tiff_path": it["tiff_path"],
            "mask_path": str(Path(it["scene_dir"]) / "water_mask.png"),
            "route": it.get("route", ""),
            "seed_backend": it.get("seed_backend", ""),
            "n_annotators": n,
            "auto_vs_final_iou": it.get("auto_vs_final_iou", ""),
            "edited_pixel_frac": it.get("edited_pixel_frac", ""),
            "active_seconds": it.get("active_seconds", ""),
            "annotator": it.get("annotator", ""),
            "timestamp": it.get("timestamp", ""),
        })
    return rows, report


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load(path: Path, split: str | None = None, fold: int | None = None,
         fold_role: str = "train") -> list[dict]:
    """Read a manifest, optionally selecting one split or one CV fold.

    With `fold` set, the split column is ignored: fold `fold` is val and every
    other fold is train, so k runs cover the whole labelled set.
    """
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if fold is not None:
        want = (lambda r: int(r["fold"]) == fold) if fold_role == "val" \
            else (lambda r: int(r["fold"]) != fold)
        rows = [r for r in rows if r.get("fold", "-1") != "-1" and want(r)]
    elif split:
        rows = [r for r in rows if r["split"] == split]
    missing = [r["stem"] for r in rows
               if not Path(r["tiff_path"]).exists() or not Path(r["mask_path"]).exists()]
    if missing:
        # a moved or unmounted archive: say so loudly, don't train on a subset
        # silently. Scene dirs are often reached through a symlink (see §7).
        raise FileNotFoundError(
            f"{len(missing)} manifest scene(s) are unreachable on disk "
            f"(e.g. {', '.join(missing[:3])}). Re-run training.manifest, or "
            f"restore the capture archive / symlink.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-log", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("annotator/work/scenes.csv"))
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0,
                    help="hold out a test split too (0 = train/val only)")
    ap.add_argument("--folds", type=int, default=0,
                    help="also assign k-fold CV indices (0 = off, 5 is typical)")
    ap.add_argument("--group-by", default="session", choices=("session", "scene"),
                    help="assign splits per capture session (default) or per scene. "
                         "Per-scene puts near-duplicate frames from one session on "
                         "both sides of the split.")
    ap.add_argument("--search-root", type=Path, action="append", default=None,
                    help="where to look for scenes whose recorded path went stale "
                         "(repeatable; default: example_data)")
    a = ap.parse_args()

    rows, report = build(a.label_log, a.val_split, a.test_split, a.folds,
                         a.search_root, a.group_by)
    if report["relocated"]:
        print(f"  relocated {len(report['relocated'])} scene(s) whose recorded path "
              f"went stale (the archive moved; the labels did not):")
        for old_p, new_p in report["relocated"][:3]:
            print(f"    {Path(old_p).name}: ...{old_p[-52:]}\n      -> {new_p}")
        if len(report["relocated"]) > 3:
            print(f"    ... and {len(report['relocated']) - 3} more")
    for kind, msg in (("ambiguous", "matched more than one directory — not guessing"),
                      ("missing", "not found under the search roots")):
        if report[kind]:
            print(f"  WARNING {len(report[kind])} scene(s) {msg}: "
                  f"{', '.join(Path(p).name for p in report[kind][:4])}")
    write(rows, a.out)
    counts: dict[str, int] = {}
    routes: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    ngroups = len({r["group"] for r in rows})
    print(f"{len(rows)} gold scene(s) in {ngroups} {a.group_by}(s) -> {a.out}")
    print("  splits:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    bysplit: dict[str, set] = {}
    for r in rows:
        bysplit.setdefault(r["split"], set()).add(r["group"])
    for k in sorted(bysplit):
        print(f"    {k:5s} {', '.join(sorted(bysplit[k]))}")
    print("  routes:", ", ".join(f"{k}={v}" for k, v in sorted(routes.items())) or "none")
    if a.folds >= 2:
        fc: dict[str, int] = {}
        for r in rows:
            fc[r["fold"]] = fc.get(r["fold"], 0) + 1
        print("  folds: ", ", ".join(f"{k}={v}" for k, v in sorted(fc.items())))
    unreachable = [r["stem"] for r in rows if not Path(r["tiff_path"]).exists()]
    if unreachable:
        print(f"  WARNING {len(unreachable)} scene(s) not on disk right now: "
              f"{', '.join(unreachable[:5])}")


if __name__ == "__main__":
    main()
