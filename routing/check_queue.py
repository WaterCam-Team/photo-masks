"""Objective 2's checks for the queue MDP (67-proposal-revised, Objectives; RL-project-next-steps, Phase 2).

    annotator/.venv/bin/python -m routing.check_queue [--data results/<features>/queue_scenes.json]

1. On 8-image queues, the hindsight optimum by subset enumeration and by DP over
   (t, b_t) both equal brute force over all 2^8 action sequences played through
   the environment's own step(), for c_s = 0 and 5 s and both budget rules.
2. The optimum is never below any baseline's return.
3. All-SAM2 and First-come match a hand calculation on a 3-image queue.
4. CostModel recovers known coefficients from noiseless synthetic times.
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from .queue_dp import dp_optimum
from .queue_env import COST_FEATURES, MANUAL, SAM2, CostModel, QueueConfig, QueueEnv, load, optimum
from .queue_policies import baselines

HERE = Path(__file__).resolve().parent


def brute_force(env: QueueEnv, seed: int) -> float:
    best = -np.inf
    for acts in itertools.product((SAM2, MANUAL), repeat=env.cfg.n):
        env.reset(seed)
        ret, ok = 0.0, True
        for a in acts:
            if a not in env.legal():
                ok = False
                break
            _, r, _, _ = env.step(a)
            ret += r
        if ok:
            best = max(best, ret)
    return best


def synthetic(qs, secs, median=10.0):
    feats = lambda i: {"ripu": 0.1 * i, "entropy": 0, "low_conf": 0, "pred_water": 0.1 * i,
                       "iou_seg_sam2": 0, "sam2_water": 0, "edge_density": 0.2 * i,
                       "pred_components": i, "pred_boundary_frac": 0.01 * i}
    return {"median_manual_s": median, "scenes": {
        f"s{i}": {"group": "g", "features": feats(i), "q_sam2": q, "manual_s": c}
        for i, (q, c) in enumerate(zip(qs, secs))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=HERE / "results/queue-features-20261001/queue_scenes.json")
    a = ap.parse_args()
    data = load(a.data)
    keys = sorted(data["scenes"])
    cost = CostModel.fit(data, keys)
    checks = []

    for review_s, rule in itertools.product((0.0, 5.0), ("proposal", "review_inclusive")):
        for rho in (0.25, 0.5):
            env = QueueEnv(data, keys, QueueConfig(n=8, rho=rho, review_s=review_s, budget=rule), cost)
            bad = 0
            for s in range(15):
                env.reset(s)
                e, d = optimum(env)[0], dp_optimum(env)
                b = brute_force(env, s)
                bad += not (abs(e - b) < 1e-6 and abs(d - b) < 1e-6)
            checks.append((f"enumeration = DP = brute force over 2^8 sequences "
                           f"(c_s {review_s:g}, {rule}, rho {rho}, 15 queues)", bad == 0))

    for review_s in (0.0, 5.0):
        env = QueueEnv(data, keys, QueueConfig(n=12, rho=0.25, review_s=review_s), cost)
        pols = baselines()
        for p in pols:
            p.fit(env, env, np.random.default_rng(0))
        worst = np.inf
        for s in range(50):
            env.reset(s)
            g = optimum(env)[0]
            for p in pols:
                obs, done, ret = env.reset(s), False, 0.0
                rng = np.random.default_rng(s)
                while not done:
                    obs, r, done, _ = env.step(p(obs, rng))
                    ret += r
                worst = min(worst, g - ret)
        checks.append((f"optimum >= every baseline on 50 queues of 12 (c_s {review_s:g}); "
                       f"smallest margin {worst:.2f}", worst >= -1e-6))

    # Hand calculation: q_SAM2 = 0.2, 0.9, 0.5; manual 12, 8, 30 s; median 10 s;
    # N = 3, rho = 1/3, so B = 10 s; c_s = 0.
    #   All-SAM2:   100 * (0.2 + 0.9 + 0.5) = 160
    #   First-come: MANUAL on image 0 (10 > 0, overdraws to -2), then SAM2 twice:
    #               100 * (1 + 0.9 + 0.5) = 240
    #   Optimum:    of the feasible MANUAL sets {0}, {1}, {2}, {1, 2} (after image 1,
    #               2 s remain), {0} is best: 240.
    toy = synthetic([0.2, 0.9, 0.5], [12, 8, 30])
    tk = sorted(toy["scenes"])
    tcost = CostModel(np.zeros(1 + len(COST_FEATURES)), 0, 10.0)
    tenv = QueueEnv(toy, tk, QueueConfig(n=3, rho=1 / 3), tcost)
    rng = np.random.default_rng(0)
    order = None
    for s in range(100):                       # find the seed that queues s0, s1, s2 in order
        tenv.reset(s)
        if tenv.queue == ["s0", "s1", "s2"]:
            order = s
            break
    pol = {p.name: p for p in baselines()}
    rets = {}
    for name in ("all_sam2", "first_come"):
        obs, done, ret = tenv.reset(order), False, 0.0
        while not done:
            obs, r, done, _ = tenv.step(pol[name](obs, rng))
            ret += r
        rets[name] = ret
    tenv.reset(order)
    checks.append(("hand calculation, 3-image queue: All-SAM2 = 160, First-come = 240, optimum = 240",
                   abs(rets["all_sam2"] - 160) < 1e-9 and abs(rets["first_come"] - 240) < 1e-9
                   and abs(optimum(tenv)[0] - 240) < 1e-9 and abs(dp_optimum(tenv) - 240) < 1e-9))

    true = np.array([5.0, 3.0, 20.0, 2.0, 100.0])
    syn = synthetic([0.5] * 8, [0.0] * 8)
    for k, v in syn["scenes"].items():
        i = int(k[1:])                           # independent regressors, not all prop. to i
        v["features"].update(edge_density=float(i), pred_water=float(i * i),
                             pred_components=float((7 * i) % 5), pred_boundary_frac=float(np.sin(i)))
        x = np.array([1.0] + [v["features"][n] for n in COST_FEATURES])
        v["manual_s"] = float(x @ true)
    fit = CostModel.fit(syn, sorted(syn["scenes"]))
    checks.append(("CostModel recovers known coefficients from noiseless times",
                   np.allclose(fit.coef, true, atol=1e-6)))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} PASS")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
