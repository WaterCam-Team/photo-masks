"""Precompute what the "Auto" route would hand back, for every scene in a manifest.

    cd photo_processing
    annotator/.venv/bin/python -m routing.gen_auto_masks \
        --manifest <scenes.csv> --out <dir> [--backends sam2 sam spectral]

The environment cannot run SAM inside every step, and the auto mask of a scene
does not depend on the agent, so it is computed once here with the annotator's
own backends and default parameters (the masks a human would see as the seed).
Writes <out>/<group>/<stem>/auto_<backend>.png (0/255, TIFF grid) and one
auto_masks.jsonl row per (scene, backend) with runtime, water fraction and the
IoU against the gold mask. Existing outputs are skipped, so it resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent          # photo_processing/


def scene_key(r: dict) -> str:
    return f'{r["group"]}/{r["stem"]}'.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backends", nargs="+", default=["sam2", "sam", "spectral"])
    ap.add_argument("--config", type=Path, default=HERE / "annotator" / "annotator_config.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE / "annotator"))
    import backends as B
    from PIL import Image
    from rl_features import mask_iou

    cfg = json.loads(a.config.read_text()).get("backends", {}) if a.config.exists() else {}
    for v in cfg.values():                     # weights paths are relative to photo_processing/
        if isinstance(v, dict) and v.get("weights") and not Path(v["weights"]).is_absolute():
            v["weights"] = str(HERE / v["weights"])
    reg = B.build_registry(cfg, device=a.device)
    for name in a.backends:
        ok, why = reg[name].available() if name in reg else (False, "not in registry")
        if not ok:
            raise SystemExit(f"backend {name} unavailable: {why}")

    rows = list(csv.DictReader(a.manifest.open()))
    log = a.out / "auto_masks.jsonl"
    a.out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        d = a.out / scene_key(r)
        d.mkdir(parents=True, exist_ok=True)
        gold = np.array(Image.open(r["mask_path"]).convert("L"))
        for name in a.backends:
            png = d / f"auto_{name}.png"
            if png.exists():
                continue
            t = time.time()
            res = reg[name].run(r["scene_dir"], r["tiff_path"], {})
            dt = time.time() - t
            row = {"scene": scene_key(r), "split": r["split"], "backend": name,
                   "elapsed_s": round(dt, 2), "error": res.error}
            if res.mask is not None:
                Image.fromarray(res.mask.astype(np.uint8), "L").save(png)
                row.update(water_frac=round(float((res.mask > 0).mean()), 4),
                           iou_vs_gold=round(mask_iou(res.mask, gold), 4))
            with log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{i + 1}/{len(rows)}] {row}", flush=True)


if __name__ == "__main__":
    main()
