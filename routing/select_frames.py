"""Propose frames for the labeling session (see the local labeling plan).

    cd photo_processing
    annotator/.venv/bin/python -m routing.select_frames            # the plan's defaults
    annotator/.venv/bin/python -m routing.select_frames --gap-min 20 --extra 1.5

Writes two proposals and never the live list:

    annotator/work/eval_sets.proposed.csv    reward and test scenes  (review, then
                                             save as eval_sets.csv to switch them on)
    annotator/work/pool_queue.proposed.csv   pool additions for the AUTO timing protocol

Per session: daytime captures with a TIFF that nobody has labeled, a minimum
time gap from every labeled frame and from each other (the rig fires seconds
apart, so neighbours are near-duplicates), then farthest-point sampling on
the label-free scene features, so the picks differ in appearance (water, ice,
light) rather than being whatever came first. Node time series are spread
over their months before sampling. `--extra` proposes more than needed so you
can drop frames that are unusable (lens fog, blur) and keep the counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent            # photo_processing/
SESSIONS = HERE / "routing/sessions.local.json"

# The session plan lives in a local, gitignored JSON file (see
# sessions.example.json): data root, (set, session, count) entries, which
# sessions are a node's time series, and each node's deployment windows.
# Sessions are "<top>/<dir>" for Collected Data and UFO2, and the node name
# for a node's time series.
#
# Every session sits wholly in ONE of reward, test or pool. Field sessions are
# single visits of 6 to 36 minutes shot in bursts seconds apart, so a reward
# frame from a visit that also has pool frames is a near-duplicate of training
# data (the memorisation trap of annotator invariant 10). Deployment windows
# come from daily contact sheets; frames outside every window (lab, bench,
# indoors, maintenance visits) are never proposed.
UFO: Path = Path()
PLAN: list[tuple[str, str, int]] = []
NODE_SERIES: set[str] = set()
DEPLOY: dict[str, list[tuple[str, str, str]]] = {}


def load_sessions(path: Path) -> None:
    """Fill UFO, PLAN, NODE_SERIES and DEPLOY from the local session file."""
    global UFO, PLAN, NODE_SERIES, DEPLOY
    if not path.exists():
        sys.exit(f"{path} not found: copy routing/sessions.example.json there "
                 "and fill in your sessions")
    with open(path) as f:
        cfg = json.load(f)
    UFO = Path(cfg["data_root"])
    PLAN = [tuple(e) for e in cfg["plan"]]
    NODE_SERIES = set(cfg["node_series"])
    DEPLOY = {k: [tuple(w) for w in v] for k, v in cfg["deploy"].items()}


def placement(sess: str, date: str) -> str | None:
    """The deployment a frame belongs to; None = not deployed (lab, indoors)."""
    if sess not in DEPLOY:
        return ""
    return next((name for name, a, b in DEPLOY[sess] if a <= date <= b), None)
FEATS = ("ndwi_pos_frac", "ndwi_mean", "nir_dark_frac_25", "thermal_otsu_sep",
         "edge_density", "gray_entropy", "r_mean", "b_mean", "nir_mean", "thermal_std")


def session_of(rel: str) -> str | None:
    p = rel.split("/")
    if p[0] in NODE_SERIES:
        return p[0] if len(p) == 2 and p[1][:1].isdigit() else None   # skip calib dirs
    if p[0] in ("Collected Data", "UFO2") and len(p) >= 3:
        return f"{p[0]}/{p[1]}"
    return None


def when(date: str, t: str) -> datetime | None:
    t = t.zfill(4)
    try:
        return datetime.strptime(date + t[:6].ljust(6, "0"), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def farthest_points(X: np.ndarray, k: int, ok) -> list[int]:
    """Greedy k-center on standardised features; `ok(i, chosen)` enforces the gap."""
    if len(X) == 0:
        return []
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    first = int(np.argmin(((Z - Z.mean(0)) ** 2).sum(1)))    # the most typical frame
    chosen = [first]
    d = ((Z - Z[first]) ** 2).sum(1)
    while len(chosen) < k:
        order = np.argsort(-d)
        nxt = next((int(i) for i in order if d[i] > 0 and ok(int(i), chosen)), None)
        if nxt is None:
            break
        chosen.append(nxt)
        d = np.minimum(d, ((Z - Z[nxt]) ** 2).sum(1))
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, default=SESSIONS,
                    help="local session plan (see routing/sessions.example.json)")
    ap.add_argument("--brightness", type=Path,
                    default=HERE / "archive_audit/capture_brightness-20260927.csv")
    ap.add_argument("--label-log", type=Path, default=HERE / "annotator/work/label_log.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "annotator/work")
    ap.add_argument("--gap-min", type=float, default=30.0,
                    help="minimum minutes between a pick and any labeled or picked frame")
    ap.add_argument("--min-brightness", type=float, default=30.0,
                    help="visible-frame mean luma; <5 is night, 5-30 dusk")
    ap.add_argument("--per-session-cap", type=int, default=60,
                    help="frames per session to compute features for (time-spread)")
    ap.add_argument("--extra", type=float, default=1.4, help="propose this x the count")
    a = ap.parse_args()
    load_sessions(a.sessions)

    sys.path.insert(0, str(HERE / "annotator"))
    from rl_features import scene_features

    labeled = set()
    if a.label_log.exists():
        for r in csv.DictReader(a.label_log.open()):
            if r.get("scene_dir"):
                labeled.add(str(Path(r["scene_dir"]).resolve()))
        # Old rows can carry a path that no longer resolves (the example_data/data
        # symlink target moved); the manifest's relocation finds where they are now.
        try:
            from training import manifest as mf
            rows, _ = mf.build(a.label_log, search_roots=[HERE / "example_data"])
            labeled |= {str(Path(r["scene_dir"]).resolve()) for r in rows}
        except Exception as e:                          # noqa: BLE001
            print(f"warning: manifest relocation unavailable ({e}); stale log paths not matched")

    # candidates per session, duplicate copies collapsed on (session name, stem)
    by_sess: dict[str, dict] = defaultdict(dict)
    for r in csv.DictReader(a.brightness.open()):
        s = session_of(r["dir"])
        if s is None or r["has_tiff"] != "True" or not r["off_mean"]:
            continue
        if float(r["off_mean"]) < a.min_brightness:
            continue
        d = (UFO / r["dir"]).resolve()
        if not (d / "color_preserved_5_band.tiff").exists():
            continue
        key = (s.split("/", 1)[-1], d.name)            # same session name, same frame
        pl = placement(s, r["date"])
        if pl is None:                                  # node not deployed that day
            continue
        by_sess[s][key] = {"dir": str(d), "t": when(r["date"], r["time"]), "stem": d.name,
                           "labeled": str(d) in labeled, "placement": pl, "session": s}
    # a frame whose copy lives under another top dir (some sessions are in
    # both Collected Data and UFO2) counts as labeled if either copy is
    names = defaultdict(list)
    for s in by_sess:
        names[s.split("/", 1)[-1]].append(s)
    for alias in names.values():
        if len(alias) > 1:
            lab = {k for s in alias for k, v in by_sess[s].items() if v["labeled"]}
            for s in alias:
                for k in lab & by_sess[s].keys():
                    by_sess[s][k]["labeled"] = True

    import json
    fc_path = a.out_dir / "select_frames.features.json"      # scene_features is slow
    fcache = json.loads(fc_path.read_text()) if fc_path.exists() else {}
    gap = timedelta(minutes=a.gap_min)

    def stratum(v):
        """Spread a node series over its placements when it has several, else months."""
        names = {n for n, _, _ in DEPLOY.get(v["session"], [])}
        return v["placement"] if len(names) > 1 else v["t"].strftime("%Y-%m")

    taken: dict[str, list[datetime]] = defaultdict(list)   # per session, across sets
    out_eval, out_pool, report = [], [], []
    for st, sess, n in PLAN:
        pool = list(by_sess.get(sess, {}).values())
        # A time gap only means something for a node's time series; within a
        # single field visit, diversity comes from appearance alone. Reward
        # sessions may re-use already-labeled frames: they are relabeled from
        # scratch (the old masks were seeded). Test and pool frames must be new.
        g = gap if sess in NODE_SERIES else timedelta(0)
        reuse = st == "reward"
        busy = ([] if reuse else [v["t"] for v in pool if v["labeled"] and v["t"]]) + taken[sess]
        cand = [v for v in pool if (reuse or not v["labeled"]) and v["t"]
                and all(abs(v["t"] - b) >= g for b in busy)]
        cand.sort(key=lambda v: v["t"])
        # Time-spread subsample. For a node series it runs per stratum (each
        # placement, or each month): one November burst of 4,268 captures
        # would otherwise crowd everything else out of UFO007.
        if sess in NODE_SERIES:
            groups = defaultdict(list)
            for v in cand:
                groups[stratum(v)].append(v)
            per_m = max(3, a.per_session_cap // max(1, len(groups)))
        else:
            groups, per_m = {"all": cand}, a.per_session_cap
        sub = []
        for vs in groups.values():
            if len(vs) > per_m:
                idx = np.linspace(0, len(vs) - 1, per_m).round().astype(int)
                vs = [vs[i] for i in sorted(set(idx))]
            sub += vs
        # Reward sessions: frames already labeled (with a seed) are always
        # proposed, first, so they are relabeled from scratch. Left out, they
        # would land in the pool from a session that is being evaluated on.
        forced = [v for v in cand if v["labeled"]] if reuse else []
        cand = forced + [v for v in sub if not (reuse and v["labeled"])]
        # 139 UFO007 TIFFs have intact headers but corrupt LZW data (archive
        # audit): unreadable frames are dropped, never labeled.
        feats = []
        for v in cand:
            f = fcache.get(v["dir"])
            if f is None:
                try:
                    f = scene_features(Path(v["dir"]) / "color_preserved_5_band.tiff")
                    f = [f[k] for k in FEATS]
                except Exception:                  # noqa: BLE001 - corrupt TIFF
                    f = []
                fcache[v["dir"]] = f
            feats.append(f)
        cand = [v for v, f in zip(cand, feats) if f]
        X = np.array([f for f in feats if f]) if cand else np.zeros((0, len(FEATS)))
        want = int(np.ceil(n * a.extra))

        def ok(i, chosen, cand=cand, g=g):
            return all(abs(cand[i]["t"] - cand[j]["t"]) >= g for j in chosen)

        # forced frames lead `cand`; count them after corrupt ones were dropped
        nf = sum(1 for v in cand if reuse and v["labeled"])
        if sess in NODE_SERIES and cand:               # spread over placements / months first
            months = sorted({stratum(v) for v in cand[nf:]})
            per = {m: [i for i, v in enumerate(cand) if i >= nf
                       and stratum(v) == m] for m in months}
            want = max(0, want - nf)                   # forced frames count toward the total
            quota = {m: want // len(months) + (i < want % len(months)) for i, m in enumerate(months)}
            picks = list(range(nf))
            for m in months:
                ii = per[m]
                sub = farthest_points(X[ii], quota[m], lambda k, ch, ii=ii: ok(ii[k], [ii[c] for c in ch]))
                picks += [ii[k] for k in sub]
        else:
            if nf:                                     # seed k-center with the forced frames
                Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
                picks = list(range(nf))
                d = np.min([((Z - Z[j]) ** 2).sum(1) for j in picks], axis=0)
                while len(picks) < max(want, nf):
                    order = np.argsort(-d)
                    nxt = next((int(i) for i in order if d[i] > 0 and ok(int(i), picks)), None)
                    if nxt is None:
                        break
                    picks.append(nxt)
                    d = np.minimum(d, ((Z - Z[nxt]) ** 2).sum(1))
            else:
                picks = farthest_points(X, want, ok)
        for i in picks:
            v = cand[i]
            taken[sess].append(v["t"])
            row = {"set": st, "session": sess, "stem": v["stem"],
                   "relabel": "yes" if v["labeled"] else "", "placement": v["placement"],
                   "time": v["t"].isoformat(sep=" "), "scene_dir": v["dir"],
                   "rank": picks.index(i) + 1}
            (out_eval if st in ("reward", "test") else out_pool).append(row)
        report.append((st, sess, n, len(pool), len(cand), len(picks)))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    fc_path.write_text(json.dumps(fcache))
    cols = ["set", "scene_dir", "session", "placement", "stem", "time", "rank", "relabel"]
    for path, rows in ((a.out_dir / "eval_sets.proposed.csv", out_eval),
                       (a.out_dir / "pool_queue.proposed.csv", out_pool)):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)                       # the annotator matches on scene_dir
        print(f"wrote {len(rows)} rows -> {path}")
    print(f"\n{'set':7s} {'session':50s} {'need':>4s} {'frames':>6s} {'eligible':>8s} {'proposed':>8s}")
    for st, sess, n, tot, el, got in report:
        flag = "  <- short" if got < n else ""
        print(f"{st:7s} {sess:50s} {n:4d} {tot:6d} {el:8d} {got:8d}{flag}")


if __name__ == "__main__":
    main()
