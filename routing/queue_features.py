"""Frozen-SegFormer features for the budgeted queue MDP (README, second scope cut).

    cd photo_processing
    annotator/.venv/bin/python -m routing.queue_features \
        --manifest <scenes.csv> --scenes <env_scenes.json> --auto <auto_masks dir> \
        --out <dir> [--device cpu]

The queue MDP never retrains SegFormer: one frozen model supplies the image
features and the RIPU score. Until the R / T labelling session exists, the only
labelled images are the 46 gold scenes the routing env already uses, so the
frozen model here is **leave-one-session-out**: each session's features come
from a model trained on every other session's gold masks, never on its own.
That is the stand-in for "trained once on the pool, applied to new images";
a feature computed by a model that saw the image's own mask would make the
state look more informative than it can be in a real campaign.

With `--train-groups` it instead trains **one** model on those sessions and
computes features for every other scene: the proposal's protocol once the R / T
frames exist ("trained once on the gold masks I already have, from sessions that
supply no queue images").

Writes:
    <out>/models/<session>.pt     fp16 weights per held-out session, or
                                  <out>/models/frozen-<hash of the sessions>.pt in --train-groups mode (resumes)
    <out>/queue_scenes.json       per scene: session, the state features and the
                                  c_hat regressors, q_sam2 = IoU(SAM2, gold),
                                  measured manual s; plus the median SAM2 review
                                  time c_s when the label log has any

RIPU is the region-based (RA) score of Xie et al., CVPR 2022, Sec. 3.3: per
pixel, the impurity (entropy of the predicted-class histogram) of its
(2k+1) x (2k+1) neighbourhood times the mean pixel entropy in it, with the
paper's RA setting k = 1, i.e. `--ripu-k 3` here is the window *width*. The
paper selects regions; averaging the map over the frame to get one image score
is this code's addition. With 2 classes each factor is at most ln 2, so a pixel
scores at most ln^2 2 = 0.48. In practice the image mean is ~0.39 x the
fraction of predicted-boundary pixels (impurity is zero off the boundary), so
it mostly measures how much waterline the model predicts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent

from . import finetune as F
from .gen_auto_masks import scene_key


def ripu_stats(model, rows: list[dict], camp: F.Campaign, sam2: list[np.ndarray],
               k: int = 3, min_px: int = 100) -> list[dict]:
    """What the frozen model says about each image: label-free apart from the
    /32 padding SceneDataset marks as IGNORE (the mask's labels are not read)."""
    import torch
    import torch.nn.functional as TF
    from scipy import ndimage
    from torch.utils.data import DataLoader
    from training.data import Geometry, SceneDataset

    ds = SceneDataset(rows, F.MODALITY, camp.norm,
                      Geometry(eval_long_side=camp.eval_long_side), train=False, cache=False)
    model.to(camp.device).eval()
    out = []
    with torch.no_grad():
        for i, (x, y) in enumerate(DataLoader(ds, batch_size=1)):
            p = TF.softmax(model.scores(x.to(camp.device)).float(), dim=1)    # 1,C,H,W
            valid = (y[0] != 255).to(camp.device)
            ent = -(p * torch.log(p + 1e-9)).sum(1, keepdim=True)            # 1,1,H,W
            onehot = TF.one_hot(p.argmax(1), p.shape[1]).permute(0, 3, 1, 2).float()
            frac = TF.avg_pool2d(onehot, k, stride=1, padding=k // 2,
                                 count_include_pad=False)
            impurity = -(frac * torch.log(frac + 1e-9)).sum(1)               # 1,H,W
            unc = TF.avg_pool2d(ent, k, stride=1, padding=k // 2,
                                count_include_pad=False)[:, 0]
            ripu = (impurity * unc)[0]
            pred = p[0].argmax(0) == 1
            h, w = sam2[i].shape
            pv, av = pred[:h, :w].cpu().numpy(), sam2[i] > 0
            # c_hat regressors: how many separate water bodies and how much
            # waterline the model predicts (a person traces both by hand)
            lab, ncomp = ndimage.label(pv, structure=np.ones((3, 3)))
            big = int((np.bincount(lab.ravel())[1:] >= min_px).sum()) if ncomp else 0
            edge = (pv[1:, :] != pv[:-1, :]).sum() + (pv[:, 1:] != pv[:, :-1]).sum()
            u = np.logical_or(pv, av).sum()
            out.append({
                "ripu": float(ripu[valid].mean()),
                "entropy": float(ent[0, 0][valid].mean()),
                "low_conf": float((p[0].max(0).values < 0.6)[valid].float().mean()),
                "pred_water": float(pred[valid].float().mean()),
                "iou_seg_sam2": float(np.logical_and(pv, av).sum() / u) if u else 1.0,
                "sam2_water": float(av.mean()),
                "pred_components": float(big),
                "pred_boundary_frac": float(edge / pv.size),
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True, help="env_scenes.json")
    ap.add_argument("--auto", type=Path, required=True, help="gen_auto_masks output dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ripu-k", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-groups", nargs="+", default=None,
                    help="train one frozen model on these sessions (the proposal's protocol)")
    ap.add_argument("--label-log", type=Path, default=HERE / "annotator/work/label_log.csv",
                    help="read the median SAM2 review time c_s from here")
    a = ap.parse_args()

    import torch
    from PIL import Image

    rows = {scene_key(r): r for r in csv.DictReader(a.manifest.open())}
    scenes = json.loads(a.scenes.read_text())
    if set(rows) != set(scenes):
        raise SystemExit(f"manifest and {a.scenes.name} disagree on "
                         f"{len(set(rows) ^ set(scenes))} scene(s)")
    manual = [v["manual"]["active_seconds"] for v in scenes.values() if v.get("manual")]
    median_s = float(np.median(manual))
    groups = sorted({v["group"] for v in scenes.values()})
    (a.out / "models").mkdir(parents=True, exist_ok=True)
    if a.train_groups:
        unknown = set(a.train_groups) - set(groups)
        if unknown:
            raise SystemExit(f"--train-groups not in the data: {sorted(unknown)}")
        tg = set(a.train_groups)
        tag = hashlib.sha1("|".join(sorted(tg)).encode()).hexdigest()[:8]
        plan = [(f"frozen-{tag}", sorted(k for k, v in scenes.items() if v["group"] not in tg),
                 [rows[k] for k, v in sorted(scenes.items()) if v["group"] in tg])]
        protocol = f"one frozen SegFormer-B0 trained on {sorted(tg)}"
    else:
        plan = [(g, sorted(k for k, v in scenes.items() if v["group"] == g),
                 [rows[k] for k, v in sorted(scenes.items()) if v["group"] != g]) for g in groups]
        protocol = "leave-one-session-out frozen SegFormer-B0"

    feats: dict[str, dict] = {}
    for g, held, train in plan:
        camp = F.Campaign.from_rows(train, [], device=a.device)
        ck = a.out / "models" / f'{g.replace(" ", "_")}.pt'
        m = F.new_model()
        if ck.exists():
            m.module.load_state_dict({k: v.float() for k, v in torch.load(ck).items()})
            info = "loaded"
        else:
            t = time.time()
            F.train_steps(m, train, camp, F.full_steps(len(train), camp, a.epochs), seed=a.seed)
            torch.save({k: v.half() for k, v in m.module.state_dict().items()}, ck)
            info = f"trained on {len(train)} scenes in {time.time() - t:.0f} s"
        sam2 = [np.array(Image.open(a.auto / k / scenes[k]["auto"]["sam2"]["file"]).convert("L"))
                for k in held]
        for k, s in zip(held, ripu_stats(m, [rows[k] for k in held], camp, sam2, a.ripu_k)):
            v = scenes[k]
            feats[k] = {"group": v["group"],
                        "features": {**s, "edge_density": v["features"]["edge_density"]},
                        "q_sam2": v["auto"]["sam2"]["iou_vs_gold"],
                        "manual_s": v["manual"]["active_seconds"] if v.get("manual") else None}
        print(f"{g}: {len(held)} scene(s), model {info}", flush=True)

    review = []
    if a.label_log.exists():
        sys.path.insert(0, str(HERE / "annotator"))
        import rl_features as rlf
        review = [float(r["review_seconds"]) for r in rlf.timing_rows(a.label_log)
                  if (r.get("review_seconds") or "").strip()]
    out = {"median_manual_s": median_s,
           "median_review_s": float(np.median(review)) if review else None,
           "n_review_timings": len(review),
           "ripu_k": a.ripu_k, "epochs": a.epochs, "protocol": protocol, "scenes": feats}
    (a.out / "queue_scenes.json").write_text(json.dumps(out, indent=1))
    print(f"{len(feats)} scenes -> {a.out / 'queue_scenes.json'}")


if __name__ == "__main__":
    main()
