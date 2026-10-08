"""Contextual bandit router: Thompson sampling over an additive per-scene gain model.

The reward arrives per batch, so credit is assigned by assuming the batch's
accuracy gain is a sum of per-scene contributions:

    alpha * dmIoU_t  ~  sum_i  theta_{a_i} . phi(x_i, g_t)     (SKIP contributes 0)

phi = standardised candidate features, global state and a bias; one weight
vector per non-skip action. The minutes term needs no learning (the cost model
knows it exactly), so each scene takes

    argmax_a  theta_a . phi_i - lam * minutes(i, a)   over {AUTO, MANUAL, SKIP=0}

with theta drawn from the Bayesian-linear-regression posterior (Thompson) while
training and its mean when evaluating. The noise variance is set from the
measured retrain spread, so the posterior knows a single step's reward is noisy.

It is a bandit, not a full RL agent: each step is valued by its immediate
gain. Whether lookahead (DQN) adds anything over this is the
bandit-vs-MDP question; this is the bandit rung.

    python -m routing.bandit --data data --out results/bandit \
        --train-seeds 100-159 --eval-seeds 0-19
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import AUTO, MANUAL, SKIP, EnvConfig, RoutingEnv

LEARNED = (AUTO, MANUAL)                 # actions with a weight vector


def seeds(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in spec.split(",")]


class Bandit:
    def __init__(self, dim: int, prior_sd: float = 3.0, noise_sd: float = 3.3):
        self.d = dim + 1                  # + bias
        self.noise_var = noise_sd ** 2
        n = len(LEARNED) * self.d
        self.A = np.eye(n) / prior_sd ** 2
        self.b = np.zeros(n)
        self.mu = self.sd = None          # feature standardisation, fixed once fitted
        self.n_updates = 0

    # -- features --------------------------------------------------------------
    @staticmethod
    def raw(obs) -> np.ndarray:
        g = np.repeat(obs["global"][None], len(obs["keys"]), 0)
        return np.concatenate([obs["candidates"], g], 1).astype(np.float64)

    def fit_scaler(self, rows: np.ndarray) -> None:
        self.mu, self.sd = rows.mean(0), rows.std(0) + 1e-6

    def phi(self, obs) -> np.ndarray:
        z = (self.raw(obs) - self.mu) / self.sd
        return np.concatenate([z, np.ones((len(z), 1))], 1)

    def design(self, obs, actions) -> np.ndarray:
        p = self.phi(obs)
        return np.concatenate([p[np.array(actions) == a].sum(0) for a in LEARNED])

    # -- posterior ---------------------------------------------------------------
    def update(self, x: np.ndarray, y: float) -> None:
        self.A += np.outer(x, x) / self.noise_var
        self.b += x * y / self.noise_var
        self.n_updates += 1

    def theta(self, rng=None) -> np.ndarray:
        cov = np.linalg.inv(self.A)
        mean = cov @ self.b
        if rng is None:
            return mean
        return rng.multivariate_normal(mean, (cov + cov.T) / 2)

    def act(self, obs, env: RoutingEnv, rng=None, lam: float | None = None) -> list[int]:
        lam = env.cfg.lam if lam is None else lam
        th = self.theta(rng).reshape(len(LEARNED), self.d)
        p = self.phi(obs)
        out = []
        for i, k in enumerate(obs["keys"]):
            val = {SKIP: 0.0}
            for j, a in enumerate(LEARNED):
                # the PREDICTED cost: a scene's own labeling time is unknown until labeled
                val[a] = float(th[j] @ p[i]) - lam * env.cost.predicted(k, a) / 60.0
            out.append(max(val, key=val.get))
        return out

    def save(self, path: Path) -> None:
        np.savez(path, A=self.A, b=self.b, mu=self.mu, sd=self.sd,
                 noise_var=self.noise_var, n_updates=self.n_updates)

    @classmethod
    def load(cls, path: Path) -> "Bandit":
        z = np.load(path)
        bd = cls(len(z["mu"]))
        bd.A, bd.b, bd.mu, bd.sd = z["A"], z["b"], z["mu"], z["sd"]
        bd.noise_var, bd.n_updates = float(z["noise_var"]), int(z["n_updates"])
        return bd

    def to_json(self) -> dict:
        th = self.theta().reshape(len(LEARNED), self.d)
        return {"n_updates": self.n_updates, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "theta_mean": {("auto", "manual")[j]: th[j].tolist() for j in range(len(LEARNED))}}


def run_episode(env, seed, policy, log_path, name, learn: Bandit | None = None):
    obs, ret, fin = env.reset(seed), 0.0, False
    while not fin:
        acts = policy(obs)
        x = learn.design(obs, acts) if learn is not None and learn.mu is not None else None
        state = {"candidates": obs["candidates"].tolist(), "global": obs["global"].tolist()}
        obs, r, fin, info = env.step(acts)
        ret += r
        if x is not None:
            learn.update(x, env.cfg.alpha * info["gain"])
        with log_path.open("a") as fh:
            fh.write(json.dumps({"policy": name, "seed": seed, **info, **state}) + "\n")
    return {"policy": name, "seed": seed, "return": ret, "miou0": env.miou0,
            "miou_final": env.miou, "minutes": env.minutes, "n_labeled": len(env.labeled),
            "test_miou0": env.test_miou0, "test_miou_final": env.test_miou}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-seeds", default="100-159")
    ap.add_argument("--eval-seeds", default="0-19")
    ap.add_argument("--explore-episodes", type=int, default=5,
                    help="uniform-random episodes first: scaler fit + initial data")
    ap.add_argument("--noise-sd", type=float, default=3.3,
                    help="reward noise per step (100x mIoU); measured retrain sd 0.023 x sqrt 2")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--eval-lams", type=float, nargs="*", default=[],
                    help="also evaluate the same posterior at these lam (gain model is lam-free)")
    ap.add_argument("--test-groups", nargs="*", default=[])
    ap.add_argument("--exclude-groups", nargs="*", default=[],
                    help="sessions never drawn during TRAINING; evaluation uses the full pool")
    ap.add_argument("--eval-only", type=Path, default=None, help="posterior.npz to evaluate")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    base = dict(data=a.data, alpha=a.alpha, lam=a.lam, test_groups=tuple(a.test_groups))
    env = RoutingEnv(EnvConfig(**base, exclude_groups=tuple(a.exclude_groups)))
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    rng = np.random.default_rng(7)
    train_log, eval_log = a.out / "train_steps.jsonl", a.out / "eval_steps.jsonl"
    for f in (train_log, eval_log, a.out / "train_episodes.jsonl", a.out / "episodes.jsonl"):
        f.unlink(missing_ok=True)       # a bandit run is not resumable: posterior is in memory

    tr = [] if a.eval_only else seeds(a.train_seeds)
    assert not set(tr) & set(seeds(a.eval_seeds)), "train and eval campaigns must differ"
    dim = None
    explore_rows, explore_data = [], []
    bandit = None
    for n, seed in enumerate(tr):
        if n < a.explore_episodes:
            # random actions; keep (obs, actions, gain) to replay once the scaler exists
            obs, fin = env.reset(seed), False
            while not fin:
                acts = list(rng.integers(0, 3, size=len(obs["keys"])))
                raw = Bandit.raw(obs)
                explore_rows.append(raw)
                o2, r, fin, info = env.step(acts)
                explore_data.append((obs, acts, a.alpha * info["gain"]))
                obs = o2
            row = {"policy": "explore", "seed": seed, "return": None}
            if n == a.explore_episodes - 1:
                dim = explore_rows[0].shape[1]
                bandit = Bandit(dim, noise_sd=a.noise_sd)
                bandit.fit_scaler(np.concatenate(explore_rows))
                for o, ac, y in explore_data:
                    bandit.update(bandit.design(o, ac), y)
        else:
            row = run_episode(env, seed, lambda o: bandit.act(o, env, rng), train_log,
                              "bandit_ts", learn=bandit)
        with (a.out / "train_episodes.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if bandit is not None:
            (a.out / "posterior.json").write_text(json.dumps(bandit.to_json()))

    if a.eval_only:
        bandit = Bandit.load(a.eval_only)
    else:
        bandit.save(a.out / "posterior.npz")
    if a.exclude_groups:                             # evaluate on the full pool
        env = RoutingEnv(EnvConfig(**base))
    for lam in [a.lam] + [l for l in a.eval_lams if l != a.lam]:
        name = "bandit" if lam == a.lam else f"bandit@lam{lam:g}"
        for seed in seeds(a.eval_seeds):             # greedy, no learning
            row = run_episode(env, seed, lambda o: bandit.act(o, env, lam=lam), eval_log, name)
            row["lam"] = lam                         # the env priced time at a.lam; re-price
            row["return"] = a.alpha * (row["miou_final"] - row["miou0"]) - lam * row["minutes"]
            with (a.out / "episodes.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
