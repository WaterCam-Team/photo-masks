"""Benchmark the inner loop: is a fast fine-tune a trustworthy stand-in for a full one?

    cd photo_processing
    annotator/.venv/bin/python -m routing.bench_finetune \
        --manifest <scenes.csv> --out routing/results/bench-finetune-cpu

A seed set S0 is labeled; several candidate batches B_i of K scenes are drawn
from the rest of the pool. For each B_i the reference is a full retrain on
S0 + B_i (the training package's epoch recipe, from ImageNet weights), and each
fast variant starts from a base model trained on S0 alone. What matters for the
agent is not the absolute score but whether a variant **ranks the candidate
batches the same way the reference does**, so the summary reports Spearman rho
of each variant's per-batch gain against the reference, next to the
reference's own seed-to-seed rho (the noise ceiling: no variant can be expected
to agree with the reference better than the reference agrees with itself).

Results append to results.jsonl one run at a time, so an interrupted benchmark
resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from training import manifest as mf

from . import finetune as F


def pick_batches(pool: list[dict], k: int, n: int, seed: int) -> list[list[dict]]:
    """n batches of k, deliberately varied in session mix so they differ in value.

    Half are drawn from a single session (redundant with each other), half
    across sessions; a batch whose value the reference cannot tell apart from
    another's teaches us nothing about ranking.
    """
    rng = np.random.default_rng(seed)
    by = {}
    for r in pool:
        by.setdefault(r["group"], []).append(r)
    groups = sorted(by, key=lambda g: -len(by[g]))
    out = []
    for i in range(n):
        if i % 2 == 0:                                  # single session
            big = [g for g in groups if len(by[g]) >= k]
            g = big[(i // 2) % len(big)]
            idx = rng.choice(len(by[g]), size=k, replace=False)
            out.append([by[g][j] for j in idx])
        else:                                           # mixed sessions
            idx = rng.choice(len(pool), size=k, replace=False)
            out.append([pool[j] for j in idx])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s0-size", type=int, default=10)
    ap.add_argument("--s0-group", default="Brooklyn Dec 2025",
                    help="S0 comes from one session, like a campaign's first labels")
    ap.add_argument("--k", type=int, default=4, help="scenes per candidate batch")
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--warm-steps", type=int, nargs="*", default=[60, 150])
    ap.add_argument("--decoder-steps", type=int, nargs="*", default=[100])
    ap.add_argument("--fast-eval-long-side", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="fp32 even on CUDA")
    ap.add_argument("--full-epochs", type=int, default=40,
                    help="the reference recipe's epochs (1 for a dry run of the harness)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    res_path = a.out / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            done.add(json.loads(line)["key"])

    def record(key: str, **kw):
        row = {"key": key, **kw}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    train_pool = mf.load(a.manifest, split="train")
    val = mf.load(a.manifest, split="val")
    rng = np.random.default_rng(0)
    s0_cands = [r for r in train_pool if r["group"] == a.s0_group]
    s0 = [s0_cands[i] for i in sorted(rng.choice(len(s0_cands), a.s0_size, replace=False))]
    s0_stems = {r["stem"] for r in s0}
    pool = [r for r in train_pool if r["stem"] not in s0_stems]
    batches = pick_batches(pool, a.k, a.n_batches, seed=1)

    # One set of band statistics for the whole campaign (see finetune.py).
    dev = dict(device=a.device, amp=not a.no_amp)
    camp = F.Campaign.from_rows(s0, val, **dev)
    fast = F.Campaign(val_rows=val, stats=camp.stats, eval_long_side=a.fast_eval_long_side, **dev)
    (a.out / "design.json").write_text(json.dumps({
        "s0": [r["stem"] for r in s0], "val": [r["stem"] for r in val],
        "val_groups": sorted({r["group"] for r in val}),
        "batches": [[f'{r["group"]}/{r["stem"]}' for r in b] for b in batches],
        "args": {k: str(v) for k, v in vars(a).items()}}, indent=2))

    def both_scores(model):
        t = time.time(); s_full = F.score(model, camp); t_full = time.time() - t
        t = time.time(); s_fast = F.score(model, fast); t_fast = time.time() - t
        return {"miou": s_full["miou"], "iou_water": s_full["iou_water"],
                "miou_fast": s_fast["miou"], "iou_water_fast": s_fast["iou_water"],
                "eval_s": round(t_full, 1), "eval_fast_s": round(t_fast, 1)}

    # base model on S0: the start point of every fast variant, and the zero of "gain"
    base_dir = a.out / "base_s0"
    if not (base_dir / "config.json").exists():
        m = F.new_model()
        info = F.train_steps(m, s0, camp, F.full_steps(len(s0), camp, a.full_epochs), seed=0)
        m.save(base_dir)
        record("base", variant="base", batch=-1, seed=0, **info, **both_scores(m))
    base = F.new_model(str(base_dir))
    cache = F.FeatureCache(base, camp, views=8)

    for seed in a.seeds:
        for bi, b in enumerate(batches):
            rows = s0 + b
            key = f"full/b{bi}/s{seed}"
            if key not in done:
                m = F.new_model()
                info = F.train_steps(m, rows, camp, F.full_steps(len(rows), camp, a.full_epochs), seed=seed)
                record(key, variant="full", batch=bi, seed=seed, **info, **both_scores(m))
            for n in a.warm_steps:
                key = f"warm{n}/b{bi}/s{seed}"
                if key not in done:
                    m = F.clone(base)
                    info = F.train_steps(m, rows, camp, n, seed=seed)
                    record(key, variant=f"warm{n}", batch=bi, seed=seed, **info, **both_scores(m))
            for n in a.decoder_steps:
                key = f"decoder{n}/b{bi}/s{seed}"
                if key not in done:
                    m = F.clone(base)
                    enc0 = cache.encode_s
                    info = F.train_decoder(m, rows, cache, n, seed=seed)
                    t = time.time(); s = F.score_decoder(m, cache); ev = time.time() - t
                    record(key, variant=f"decoder{n}", batch=bi, seed=seed, **info,
                           miou=s["miou"], iou_water=s["iou_water"], eval_s=round(ev, 1),
                           encode_s=round(cache.encode_s - enc0, 1))

    summarize(res_path, a.out / "summary.md")


def summarize(res_path: Path, out_md: Path) -> str:
    from scipy.stats import spearmanr

    rows = [json.loads(l) for l in res_path.read_text().splitlines()]
    base = next(r for r in rows if r["variant"] == "base")
    variants = sorted({r["variant"] for r in rows} - {"base"},
                      key=lambda v: (v != "full", v))
    seeds = sorted({r["seed"] for r in rows if r["variant"] == "full"})

    def table(variant, metric, seed=None):
        d = {}
        for r in rows:
            if r["variant"] == variant and metric in r and (seed is None or r["seed"] == seed):
                d.setdefault(r["batch"], []).append(r[metric])
        return {b: float(np.mean(v)) for b, v in d.items()}

    lines = [f"# Inner-loop benchmark\n",
             f"base (S0 only): mIoU {base['miou']:.4f}, water IoU {base['iou_water']:.4f}\n",
             "| variant | train s/batch | eval s | mean mIoU | seed spread | rho vs full (mIoU) | rho vs full (water IoU) |",
             "|---|---|---|---|---|---|---|"]
    ref_m, ref_w = table("full", "miou"), table("full", "iou_water")
    bs = sorted(ref_m)
    for v in variants:
        for metric_suffix in ([""] if v.startswith("decoder") else ["", "_fast"]):
            m = table(v, "miou" + metric_suffix)
            w = table(v, "iou_water" + metric_suffix)
            if not m:
                continue
            common = [b for b in bs if b in m]
            rho_m = spearmanr([ref_m[b] for b in common], [m[b] for b in common])[0] \
                if v != "full" or metric_suffix else float("nan")
            rho_w = spearmanr([ref_w[b] for b in common], [w[b] for b in common])[0] \
                if v != "full" or metric_suffix else float("nan")
            vr = [r for r in rows if r["variant"] == v]
            spread = np.mean([np.std([r["miou" + metric_suffix] for r in vr if r["batch"] == b])
                              for b in common])
            ev = "eval_fast_s" if metric_suffix else "eval_s"
            lines.append(
                f"| {v}{' @fast-eval' if metric_suffix else ''} "
                f"| {np.mean([r['train_s'] for r in vr]):.0f} "
                f"| {np.mean([r.get(ev, 0) for r in vr]):.1f} "
                f"| {np.mean(list(m.values())):.4f} | {spread:.4f} "
                f"| {rho_m:.2f} | {rho_w:.2f} |")
    if len(seeds) >= 2:
        a_, b_ = table("full", "miou", seeds[0]), table("full", "miou", seeds[1])
        common = [b for b in a_ if b in b_]
        rho = spearmanr([a_[b] for b in common], [b_[b] for b in common])[0]
        aw, bw = table("full", "iou_water", seeds[0]), table("full", "iou_water", seeds[1])
        rhow = spearmanr([aw[b] for b in common], [bw[b] for b in common])[0]
        lines.append(f"\nNoise ceiling, full seed {seeds[0]} vs seed {seeds[1]}: "
                     f"rho {rho:.2f} (mIoU), {rhow:.2f} (water IoU) over {len(common)} batches.")
    # The agent gets ONE run per batch, and seed-averaging removes noise, so the
    # comparison that matters is single run against single run, which is also
    # how the noise ceiling above is measured.
    def vec(v, sd, metric):
        d = {r["batch"]: r[metric] for r in rows
             if r["variant"] == v and r["seed"] == sd and metric in r}
        return [d.get(b, np.nan) for b in bs]
    lines += ["\n## Single run vs single run (the fair comparison)\n",
              "Mean Spearman rho over (variant seed, reference seed) pairs, [min, max].\n",
              "| variant | rho mIoU | rho water IoU |", "|---|---|---|"]
    for v in variants:
        for suf in ([""] if v.startswith("decoder") else ["", "_fast"]):
            cells = []
            for metric in ("miou", "iou_water"):
                rs = [spearmanr(vec("full", rs_, metric), vec(v, vs, metric + suf))[0]
                      for vs in seeds for rs_ in seeds
                      if not (v == "full" and not suf and vs == rs_)]
                cells.append(f"{np.mean(rs):.2f} [{min(rs):.2f}, {max(rs):.2f}]" if rs else "n/a")
            lines.append(f"| {v}{' @fast-eval' if suf else ''} | {cells[0]} | {cells[1]} |")
    lines.append("\nPer-batch reference (full, seed-mean): " + ", ".join(
        f"b{b} {ref_m[b]:.4f}" for b in bs))
    text = "\n".join(lines) + "\n"
    out_md.write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
