"""Roll policies through the environment and log every step.

    python -m routing.run_episodes --data data --out results/<name> \
        --policies all_manual all_auto random --seeds 0 1 2

One JSONL row per step (policy, seed, t, scenes, actions, minutes, mIoU,
gain, reward) plus one per episode (return, final mIoU, total minutes).
Resumable: (policy, seed) pairs already in episodes.jsonl are skipped. The
same seed gives every policy the same pool, S0 and batch order, so policies
are compared on identical campaigns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, RoutingEnv
from .policies import BASELINES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policies", nargs="+", default=[p.name for p in BASELINES])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--auto-backend", default="sam2")
    ap.add_argument("--s0", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-groups", nargs="*", default=[])
    a = ap.parse_args()

    pols = {p.name: p for p in BASELINES}
    unknown = [p for p in a.policies if p not in pols]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; have {sorted(pols)}")
    a.out.mkdir(parents=True, exist_ok=True)
    env = RoutingEnv(EnvConfig(data=a.data, auto_backend=a.auto_backend, s0_size=a.s0,
                               batch=a.batch, steps=a.steps, alpha=a.alpha, lam=a.lam,
                               epochs=a.epochs, device=a.device,
                               test_groups=tuple(a.test_groups)))
    (a.out / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(a).items()},
         "cost": {"manual_default_s": env.cost.manual_default_s, "auto_s": env.cost.auto_s,
                  "n_measured_manual": len(env.cost.per_scene_manual)}}, indent=2))
    ep_path, st_path = a.out / "episodes.jsonl", a.out / "steps.jsonl"
    done = set()
    if ep_path.exists():
        done = {(j["policy"], j["seed"]) for j in map(json.loads, ep_path.read_text().splitlines())}

    for seed in a.seeds:
        for name in a.policies:
            if (name, seed) in done:
                continue
            rng = np.random.default_rng(1000 + seed)
            obs, ret, fin = env.reset(seed), 0.0, False
            while not fin:
                state = {"candidates": obs["candidates"].tolist(),
                         "global": obs["global"].tolist()}     # what the policy saw
                obs, r, fin, info = env.step(pols[name](obs, rng))
                ret += r
                with st_path.open("a") as fh:
                    fh.write(json.dumps({"policy": name, "seed": seed, **info, **state}) + "\n")
            row = {"policy": name, "seed": seed, "return": ret, "miou0": env.miou0,
                   "miou_final": env.miou, "minutes": env.minutes,
                   "n_labeled": len(env.labeled),
                   "test_miou0": env.test_miou0, "test_miou_final": env.test_miou}
            with ep_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


    print(summarize(ep_path, a.out / "summary.md"))


def summarize(ep_path: Path, out_md: Path, ref: str = "all_manual",
              lam: float | None = None, alpha: float = 100.0) -> str:
    """Per policy: mean +- sd over campaigns, and the paired difference against
    `ref` on the same campaigns (every policy sees the same seeds, so pairing
    removes the campaign-to-campaign spread, which is the larger one)."""
    rows = [json.loads(l) for l in ep_path.read_text().splitlines()]
    if lam is not None:     # re-price a fixed policy's campaign at another lam
        for r in rows:
            r["return"] = alpha * (r["miou_final"] - r["miou0"]) - lam * r["minutes"]
    by = {}
    for r in rows:
        by.setdefault(r["policy"], {})[r["seed"]] = r
    refd = by.get(ref, {})

    def ms(v):
        return f"{np.mean(v):+.2f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}"

    has_test = any(r.get("test_miou_final") is not None for r in rows)
    lines = ["| policy | n | return | final mIoU | gain (pts) | minutes | "
             f"return - {ref} (paired) | wins vs {ref} |" + (" TEST gain (pts) | TEST gain - ref |" if has_test else ""),
             "|---|---|---|---|---|---|---|---|" + ("---|---|" if has_test else "")]
    order = sorted(by, key=lambda p: -np.mean([r["return"] for r in by[p].values()]))
    for p in order:
        d = by[p]
        ret = [r["return"] for r in d.values()]
        gain = [100 * (r["miou_final"] - r["miou0"]) for r in d.values()]
        common = [s for s in d if s in refd]
        diff = [d[s]["return"] - refd[s]["return"] for s in common]
        wins = sum(x > 0 for x in diff)
        lines.append(f"| {p} | {len(d)} | {ms(ret)} | {np.mean([r['miou_final'] for r in d.values()]):.3f} "
                     f"| {ms(gain)} | {np.mean([r['minutes'] for r in d.values()]):.2f} "
                     f"| {ms(diff) if p != ref else '-'} | {f'{wins}/{len(common)}' if p != ref else '-'} |"
                     + (_test_cells(d, refd, common, p == ref, ms) if has_test else ""))
    text = "\n".join(lines) + "\n"
    out_md.write_text(text)
    return text


def _test_cells(d, refd, common, is_ref, ms) -> str:
    """Gain on the TEST sessions, which never enter the reward."""
    g = lambda r: 100 * (r["test_miou_final"] - r["test_miou0"])
    tg = [g(r) for r in d.values() if r.get("test_miou_final") is not None]
    diff = [g(d[s]) - g(refd[s]) for s in common
            if d[s].get("test_miou_final") is not None and refd[s].get("test_miou_final") is not None]
    return f" {ms(tg) if tg else '-'} | {ms(diff) if diff and not is_ref else '-'} |"


if __name__ == "__main__":
    main()
