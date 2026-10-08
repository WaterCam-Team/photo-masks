"""Run every queue-MDP policy on identical queues and write the evaluation metrics.

    python -m routing.run_queue --data results/<features>/queue_scenes.json \
        --out results/<name> [--review-s 5 --review-placeholder] [--rho 0.25 0.5]

Protocol:

* Fit and evaluation scenes never overlap. `--eval-groups` names the evaluation
  sessions (the T set once it exists); without it, two session folds run on the
  interim data: A evaluates on --split-group and fits on the rest, B the reverse.
* Everything that is tuned is tuned on training queues only: c_hat (linear fit),
  the RIPU-threshold tau (grid, per rho) and Sarsa's step size (grid; the alpha
  with the best mean held-out training return over the seeds is kept).
* Sarsa: 5 seeds, epsilon 0.2 -> 0.01 over 5,000 episodes, learning curve every
  250 episodes on held-out training queues.
* 100 evaluation queues per budget; queue seed s is the same queue for every
  policy, so all differences are paired. Sarsa's per-queue return is the mean
  over its seeds; the seed spread is reported separately.
* The hindsight optimum is computed twice, by subset enumeration and by DP over
  (t, b_t), and the run stops if they disagree.

Primary metric: share of the possible gain captured,
(G - G_all_sam2) / (G_optimum - G_all_sam2), as a ratio of means. Sarsa beats
RIPU-threshold at a budget if the paired difference in return (equivalently in
share: the denominator is common to both) has a 95% CI above zero.

c_s, the SAM2 review time, is read from queue_scenes.json when the label log has
review timings; until then it must be given with --review-s, and
--review-placeholder marks every output as using an unmeasured value.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .queue_dp import dp_optimum
from .queue_env import CostModel, QueueConfig, QueueEnv, load, optimum
from .queue_policies import Sarsa, baselines

Z95 = 1.96


def rollout(env: QueueEnv, policy, seed: int, rng) -> dict:
    obs, done, ret, steps = env.reset(seed), False, 0.0, []
    while not done:
        a = policy(obs, rng)
        obs, r, done, info = env.step(a)
        ret += r
        steps.append(info)
    return {"return": ret, "steps": steps, "budget_left_s": env.budget}


def train_sarsa(env, train_env, alphas, seeds, episodes, interact, log):
    """Grid over alpha; keep the seeds of the alpha with the best mean final
    held-out training return."""
    best = None
    for alpha in alphas:
        agents = []
        for sd in seeds:
            ag = Sarsa(alpha=alpha, seed=sd, episodes=episodes, interact=interact)
            ag.fit(env, train_env, None)
            agents.append(ag)
        score = float(np.mean([ag.curve[-1][1] for ag in agents]))
        log.append({"alpha": alpha, "held_out_return": score,
                    "per_seed": [ag.curve[-1][1] for ag in agents]})
        if best is None or score > best[0]:
            best = (score, alpha, agents)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--episodes", type=int, default=100, help="evaluation queues per budget")
    ap.add_argument("--sarsa-episodes", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    ap.add_argument("--no-ablation", action="store_true", help="skip sarsa_x")
    ap.add_argument("--review-s", type=float, default=None,
                    help="c_s in seconds; required while the label log has no review timings")
    ap.add_argument("--review-placeholder", action="store_true",
                    help="--review-s is not a measurement; say so in every output")
    ap.add_argument("--budget", choices=["review_inclusive", "labels_only"], default="review_inclusive")
    ap.add_argument("--eval-groups", nargs="+", default=None)
    ap.add_argument("--split-group", default="Brooklyn Dec 2025")
    a = ap.parse_args()

    data = load(a.data)
    sc = data["scenes"]
    measured = data.get("median_review_s")
    if a.review_s is None and measured is None:
        raise SystemExit("no SAM2 review timings in the data: pass --review-s (and "
                         "--review-placeholder if it is not a measurement)")
    review_s = a.review_s if a.review_s is not None else measured
    review_note = ("PLACEHOLDER, not measured" if a.review_placeholder else
                   f"measured median of {data.get('n_review_timings')} reviews"
                   if a.review_s is None else "given on the command line")
    groups = sorted({v["group"] for v in sc.values()})
    if a.eval_groups:
        unknown = set(a.eval_groups) - set(groups)
        if unknown:
            raise SystemExit(f"--eval-groups not in the data: {sorted(unknown)}")
        ev = set(a.eval_groups)
        folds = {"T": ([k for k in sc if sc[k]["group"] not in ev], [k for k in sc if sc[k]["group"] in ev])}
    else:
        if a.split_group not in groups:
            raise SystemExit(f"--split-group {a.split_group!r} not in {groups}")
        side = {k: v["group"] == a.split_group for k, v in sc.items()}
        folds = {"A": ([k for k in sc if not side[k]], [k for k in sc if side[k]]),
                 "B": ([k for k in sc if side[k]], [k for k in sc if not side[k]])}
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seeds))
    variants = [False] if a.no_ablation else [False, True]

    lines = ["# Queue MDP: policies against the hindsight optimum", "",
             f"Features: {data.get('protocol')}. N = {a.n}; {a.episodes} paired evaluation queues per "
             f"fold and budget; Sarsa {a.sarsa_episodes} episodes x {a.seeds} seeds, alpha from "
             f"{a.alphas} by held-out training return. Median manual time c_bar = "
             f"{data['median_manual_s']:.1f} s. **SAM2 review time c_s = {review_s:g} s ({review_note}).** "
             f"Budget rule: {a.budget}.", ""]
    if a.review_placeholder:
        lines += ["> c_s is a placeholder. Every number below depends on it and changes once "
                  "review times are measured.", ""]
    config = {**{k: str(v) for k, v in vars(a).items()}, "review_s_used": review_s,
              "review_note": review_note, "features_protocol": data.get("protocol"), "folds": {}}
    ep_rows, curves, success = [], {}, {}

    for fold, (fit_keys, eval_keys) in folds.items():
        cost = CostModel.fit(data, fit_keys)
        mae, n_mae = cost.mae(data, eval_keys)
        config["folds"][fold] = {"fit_groups": sorted({sc[k]["group"] for k in fit_keys}),
                                 "eval_groups": sorted({sc[k]["group"] for k in eval_keys}),
                                 "c_hat_coef": cost.coef.tolist(), "c_hat_n_fit": cost.n_fit,
                                 "c_hat_mae_s": mae, "c_hat_mae_n": n_mae}
        for rho in a.rho:
            cfg = QueueConfig(n=a.n, rho=rho, review_s=review_s, budget=a.budget)
            env = QueueEnv(data, eval_keys, cfg, cost)
            train_env = QueueEnv(data, fit_keys, cfg, cost)
            pols = baselines()
            rng = np.random.default_rng(0)
            for p in pols:
                p.fit(env, train_env, rng)
            sarsa_log, groups_of = {}, {}
            for inter in variants:
                name = "sarsa_x" if inter else "sarsa"
                sarsa_log[name] = []
                alpha, agents = train_sarsa(env, train_env, a.alphas, seeds, a.sarsa_episodes,
                                            inter, sarsa_log[name])
                groups_of[name] = (alpha, agents)
                curves[f"{fold}/rho{rho}/{name}"] = {"alpha": alpha,
                                                     "seeds": [ag.curve for ag in agents],
                                                     "grid": sarsa_log[name]}

            res = defaultdict(list)
            for s in range(a.episodes):
                env.reset(s)
                g_enum, _ = optimum(env)
                g_dp = dp_optimum(env)
                if abs(g_enum - g_dp) > 1e-6:
                    raise SystemExit(f"optimum disagrees on queue {s}: enumeration {g_enum}, DP {g_dp}")
                res["optimum"].append({"return": g_enum, "steps": []})
                for p in pols:
                    res[p.name].append(rollout(env, p, s, np.random.default_rng((1, s))))
                for name, (_, agents) in groups_of.items():
                    runs = [rollout(env, ag, s, np.random.default_rng((1, s))) for ag in agents]
                    res[name].append({"return": float(np.mean([r["return"] for r in runs])),
                                      "runs": runs, "steps": [st for r in runs for st in r["steps"]],
                                      "budget_left_s": float(np.mean([r["budget_left_s"] for r in runs]))})
            for name, eps in res.items():
                for s, e in enumerate(eps):
                    ep_rows.append({"fold": fold, "rho": rho, "policy": name, "seed": s,
                                    "return": e["return"],
                                    "n_manual": sum(st["action"] == "manual" for st in e["steps"])
                                    / max(1, len(e.get("runs", [None]))),
                                    "budget_left_s": e.get("budget_left_s")})
            block, ok = _report(fold, rho, res, env, train_env, pols, groups_of, cost, mae, n_mae,
                                config["folds"][fold], sc, a)
            lines += block
            success[(fold, rho)] = ok

    lines += ["## Success criterion", "",
              "Sarsa beats RIPU-threshold (paired 95% CI above zero) at every budget:", ""]
    for fold in folds:
        verdict = all(success[(fold, rho)] for rho in a.rho)
        lines.append(f"- fold {fold}: " + ", ".join(f"rho {rho}: {'yes' if success[(fold, rho)] else 'no'}"
                                                   for rho in a.rho)
                     + f" -> **{'met' if verdict else 'not met'}**")
    (a.out / "config.json").write_text(json.dumps(config, indent=2))
    (a.out / "learning_curves.json").write_text(json.dumps(curves))
    with (a.out / "episodes.jsonl").open("w") as fh:
        for r in ep_rows:
            fh.write(json.dumps(r) + "\n")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _ci(d: np.ndarray) -> tuple[float, float, float]:
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - Z95 * se, d.mean() + Z95 * se


def _report(fold, rho, res, env, train_env, pols, groups_of, cost, mae, n_mae, fcfg, sc, a):
    n = env.cfg.n
    G = {k: np.array([e["return"] for e in v]) for k, v in res.items()}
    floor, ceil = G["all_sam2"], G["optimum"]
    room = (ceil - floor).mean()
    ref = G["ripu_threshold"]
    ripu = next(p for p in pols if p.name == "ripu_threshold")
    out = [f"## Fold {fold}, rho = {rho}", "",
           f"Fit on: {', '.join(fcfg['fit_groups'])}. Evaluated on: {', '.join(fcfg['eval_groups'])}.", "",
           f"- Budget B = {env.budget0:.1f} s; optimum's mean gain over All-SAM2: {room:.1f} reward units per queue.",
           f"- c_hat: linear fit on {cost.n_fit} timed training scenes; MAE on {n_mae} timed evaluation "
           f"scenes = " + (f"{mae:.1f} s" if mae is not None else "n/a") + ".",
           f"- RIPU-threshold tau = {ripu.tau:.5g} (grid on training queues).",
           "- Sarsa alpha = " + ", ".join(f"{k} {v[0]:g}" for k, v in groups_of.items()) + ".", "",
           "| policy | return (mean ± se) | mask IoU | share of optimum gain | vs RIPU-threshold, mean [95% CI] "
           "| better / worse | MANUAL per queue | s used | overrun s | SAM2-failure recall | q_SAM2: MANUAL vs SAM2 images |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["all_sam2", "random", "first_come", "ripu_threshold", "ripu_ranked"] + list(groups_of) + ["optimum"]
    ok = False
    for name in order:
        g = G[name]
        d = g - ref
        m, lo, hi = _ci(d)
        share = (g - floor).mean() / room if room > 0 else float("nan")
        row = [name, f"{g.mean():.1f} ± {g.std(ddof=1) / np.sqrt(len(g)):.1f}", f"{g.mean() / (100 * n):.3f}",
               f"{share:.2f}", "—" if name == "ripu_threshold" else f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]",
               "—" if name == "ripu_threshold" else f"{int((d > 1e-9).sum())} / {int((d < -1e-9).sum())}"]
        steps = [st for e in res[name] for st in e["steps"]]
        if steps:
            per = len(res[name]) * len(res[name][0].get("runs", [None]))
            man = [st for st in steps if st["action"] == "manual"]
            sam = [st for st in steps if st["action"] == "sam2"]
            fails = [st for st in steps if st["q_sam2"] < 0.5]
            left = [r["budget_left_s"] for e in res[name] for r in e.get("runs", [e])]
            row += [f"{len(man) / per:.1f}", f"{env.budget0 - np.mean(left):.0f}",
                    f"{np.mean([max(0.0, -b) for b in left]):.1f}",
                    f"{sum(st['action'] == 'manual' for st in fails) / len(fails):.2f}" if fails else "n/a",
                    (f"{np.mean([st['q_sam2'] for st in man]):.2f}" if man else "—") + " vs "
                    + (f"{np.mean([st['q_sam2'] for st in sam]):.2f}" if sam else "—")]
        else:
            row += [""] * 5
        out.append("| " + " | ".join(row) + " |")
        if name == "sarsa":
            ok = lo > 0
    for name, (alpha, agents) in groups_of.items():
        shares = []
        for i in range(len(agents)):
            gi = np.array([e["runs"][i]["return"] for e in res[name]])
            shares.append((gi - floor).mean() / room if room > 0 else float("nan"))
        curve_end = [ag.curve[-1][1] for ag in agents]
        curve_start = [ag.curve[0][1] for ag in agents]
        out.append(f"\n{name}: share per seed {', '.join(f'{x:.2f}' for x in shares)}; held-out training "
                   f"return {np.mean(curve_start):.1f} at episode 0 -> {np.mean(curve_end):.1f} "
                   f"(sd {np.std(curve_end):.1f} over seeds) at episode {agents[0].curve[-1][0]}.")
    if "sarsa" in groups_of:
        m, lo, hi = _ci(G["sarsa"] - G["first_come"])
        out.append(f"\nSarsa vs First-come (does it hold budget back?): {m:+.1f} [{lo:+.1f}, {hi:+.1f}].")
        # B4: P(MANUAL) against budget left per remaining image, Sarsa's decision rule
        bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf]
        st = [x for e in res["sarsa"] for x in e["steps"] if x["ratio"] > 0]
        cells = []
        for lo_, hi_ in zip(bins[:-1], bins[1:]):
            sel = [x for x in st if lo_ <= x["ratio"] < hi_]
            if sel:
                cells.append(f"[{lo_:g}, {hi_:g}): {np.mean([x['action'] == 'manual' for x in sel]):.2f} (n={len(sel)})")
        out.append("Sarsa P(MANUAL) by budget left per remaining image (in median labels): " + "; ".join(cells) + ".")
    sess = sorted({sc[s["key"]]["group"] for e in res["all_sam2"] for s in e["steps"]})
    if len(sess) > 1:
        out += ["", "Per session, mean q gain over SAM2 per image (x100):", "",
                "| policy | " + " | ".join(sess) + " |", "|---" * (len(sess) + 1) + "|"]
        for name in ["first_come", "ripu_threshold", "ripu_ranked"] + list(groups_of):
            acc = defaultdict(list)
            for e in res[name]:
                for s in e["steps"]:
                    acc[sc[s["key"]]["group"]].append(s["q"] - s["q_sam2"])
            out.append(f"| {name} | " + " | ".join(f"{100 * np.mean(acc[g]):.1f}" for g in sess) + " |")
    return out + [""], ok


if __name__ == "__main__":
    main()
