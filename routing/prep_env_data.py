"""Package everything the environment needs per scene into one JSON.

    cd photo_processing
    annotator/.venv/bin/python -m routing.prep_env_data \
        --manifest <scenes.csv> --auto <auto_masks dir> --out <env_scenes.json>

Per scene: split, session, the annotator's label-free scene features
(`rl_features.scene_features`), the measured manual cost when this scene has a
timing-valid log row (`rl_features.timing_rows`, so the excluded pilot rows
never count), and the auto masks with their quality against gold. Paths are
stored relative to the scene key, so the file travels to the GPU box as is.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--auto", type=Path, required=True)
    ap.add_argument("--label-log", type=Path, default=HERE / "annotator/work/label_log.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(HERE / "annotator"))
    import rl_features as rlf
    from .gen_auto_masks import scene_key

    timing = {}
    for r in rlf.timing_rows(a.label_log):
        timing[os.path.realpath(r["scene_dir"])] = {
            "route": r["route"], "active_seconds": float(r["active_seconds"])}
    auto_q = {}
    for line in (a.auto / "auto_masks.jsonl").read_text().splitlines():
        j = json.loads(line)
        auto_q[(j["scene"], j["backend"])] = j

    out = {}
    for r in csv.DictReader(a.manifest.open()):
        k = scene_key(r)
        autos = {}
        for p in sorted((a.auto / k).glob("auto_*.png")):
            b = p.stem[len("auto_"):]
            q = auto_q.get((k, b), {})
            autos[b] = {"file": p.name, "iou_vs_gold": q.get("iou_vs_gold"),
                        "water_frac": q.get("water_frac"), "elapsed_s": q.get("elapsed_s")}
        t = timing.get(os.path.realpath(r["scene_dir"]))
        out[k] = {"split": r["split"], "group": r["group"], "stem": r["stem"],
                  "features": rlf.scene_features(Path(r["tiff_path"])),
                  "manual": t,                         # None = no valid timing for this scene
                  "auto": autos}
    a.out.write_text(json.dumps(out, indent=1))
    n_t = sum(v["manual"] is not None for v in out.values())
    print(f"{len(out)} scenes -> {a.out}; {n_t} with measured manual time")


if __name__ == "__main__":
    main()
