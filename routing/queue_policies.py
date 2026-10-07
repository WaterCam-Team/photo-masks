"""Policies for the budgeted queue MDP (67-proposal-revised, "Policies compared").

Every policy is called as `policy(obs, rng) -> action` and must return an action
in `obs["legal"]`. `fit(env, train_env, rng)` may set anything it needs from the
training environment only (a threshold, feature scaling, weights); it never sees
an evaluation scene.

    all_sam2        never spends the budget on MANUAL: the floor
    random          MANUAL with probability rho while the budget lasts
    first_come      MANUAL while the budget lasts: the one-step-greedy (gamma = 0)
                    answer, since MANUAL's reward of 100 is never below SAM2's 100 q
    ripu_threshold  MANUAL if RIPU >= tau and budget remains; tau picked from a grid
                    by mean return on training queues, per rho. The main baseline
    ripu_ranked     sees the whole queue: MANUAL when the image is among the top-m
                    remaining by RIPU, m = labels the budget left still buys
    sarsa           episodic semi-gradient Sarsa, linear in the standardized state
                    features plus a bias (S&B Sec. 10.1). The method
    sarsa_x         ablation, not in the proposal: sarsa plus image-feature x
                    budget-per-image products, so the value of a label can depend on
                    how scarce labels are
"""
from __future__ import annotations

import math

import numpy as np

from .queue_env import MANUAL, SAM2, STATE_FEATURES, QueueEnv


class Policy:
    name = "policy"

    def fit(self, env: QueueEnv, train_env: QueueEnv, rng) -> None:
        pass

    def __call__(self, obs: dict, rng) -> int:
        raise NotImplementedError


def greedy_return(env: QueueEnv, policy, seeds, rng=None) -> float:
    """Mean return of `policy` over the queues env.reset(s) draws for s in seeds."""
    rng = rng or np.random.default_rng(0)
    tot = 0.0
    for s in seeds:
        obs, done = env.reset(s), False
        while not done:
            obs, r, done, _ = env.step(policy(obs, rng))
            tot += r
    return tot / len(seeds)


class AllSAM2(Policy):
    name = "all_sam2"

    def __call__(self, obs, rng):
        return SAM2


class Random(Policy):
    name = "random"

    def fit(self, env, train_env, rng):
        self.p = env.cfg.rho

    def __call__(self, obs, rng):
        return MANUAL if MANUAL in obs["legal"] and rng.random() < self.p else SAM2


class FirstCome(Policy):
    name = "first_come"

    def __call__(self, obs, rng):
        return MANUAL if MANUAL in obs["legal"] else SAM2


_RIPU = STATE_FEATURES.index("ripu")


class RIPUThreshold(Policy):
    """tau from a grid of training-RIPU quantiles (plus 'never'), chosen by mean
    return on `tune_queues` training queues. Ties go to the lower tau."""

    name = "ripu_threshold"

    def __init__(self, tune_queues: int = 500, seed_offset: int = 2_000_000):
        self.tune_queues, self.seed_offset = tune_queues, seed_offset

    def __call__(self, obs, rng):
        return MANUAL if MANUAL in obs["legal"] and obs["image"][_RIPU] >= self.tau else SAM2

    def fit(self, env, train_env, rng):
        r = np.array([train_env.features(k)[_RIPU] for k in train_env.pool])
        grid = sorted(set(np.quantile(r, np.linspace(0, 1, 21)).tolist())) + [math.inf]
        seeds = range(self.seed_offset, self.seed_offset + self.tune_queues)
        self.grid = []
        for tau in grid:
            self.tau = tau
            self.grid.append((tau, greedy_return(train_env, self, seeds)))
        self.tau = max(self.grid, key=lambda tg: tg[1])[0]


class RIPURanked(Policy):
    """Not a causal queue policy: it sees every image still to come, as RIPU does
    in the paper's own setting. Each step it works out m, how many hand labels the
    budget left still buys after reviewing the rest with SAM2 (at the median
    manual time), and hand-labels this image if it is among the top m remaining
    by RIPU. When the reviews alone would exhaust the budget (possible under the
    proposal's budget rule) m is 1 while any budget is left: it then labels an
    image only if no later one scores higher."""

    name = "ripu_ranked"

    def fit(self, env, train_env, rng):
        self.env = env

    def __call__(self, obs, rng):
        if MANUAL not in obs["legal"]:
            return SAM2
        e = self.env
        n_left = len(obs["remaining"])
        b = obs["budget"][0] * e.median_s
        per_label = max(1e-9, e.median_s - e.cfg.review_s)
        m = max(1, math.ceil(max(0.0, b - n_left * e.cfg.review_s) / per_label))
        r = [e.features(k)[_RIPU] for k in obs["remaining"]]
        rank = int(np.sum(np.array(r) > r[0]))           # images ahead of this one
        return MANUAL if rank < m else SAM2


class Sarsa(Policy):
    """Episodic semi-gradient Sarsa(0) with linear q(s, a) = w_a . psi(s), gamma = 1.

    psi(s) = [1, z(image features), budget features / fixed scales], z standardised
    on the training scenes; `interact=True` appends z * budget_per_image (sarsa_x).
    Rewards are learned in units of q (r / alpha). Epsilon-greedy over legal
    actions, epsilon decaying linearly from eps0 to eps1 over the training episodes.
    Every `curve_every` episodes the greedy policy is scored on fixed held-out
    training queues (the proposal's learning curve, metric B1). Evaluation is greedy.
    """

    def __init__(self, alpha: float, seed: int, episodes: int = 5000, eps0: float = 0.2,
                 eps1: float = 0.01, interact: bool = False, curve_every: int = 250,
                 curve_queues: int = 100, seed_offset: int = 1_000_000):
        self.alpha, self.seed, self.episodes = alpha, seed, episodes
        self.eps0, self.eps1, self.interact = eps0, eps1, interact
        self.curve_every, self.curve_queues = curve_every, curve_queues
        self.seed_offset = seed_offset          # training queues never reuse eval seeds
        self.name = "sarsa_x" if interact else "sarsa"

    def phi(self, obs) -> np.ndarray:
        z = (obs["image"] - self.mu) / self.sd
        b = obs["budget"] / self.bscale
        parts = [[1.0], z, b] + ([z * b[2]] if self.interact else [])
        return np.concatenate(parts)

    def _act(self, obs, rng, eps: float) -> int:
        legal = obs["legal"]
        if len(legal) > 1 and rng.random() < eps:
            return int(rng.choice(legal))
        qv = self.w @ self.phi(obs)
        return max(legal, key=lambda a: qv[a])

    def fit(self, env, train_env, rng):
        rng = np.random.default_rng(self.seed)
        f = np.array([train_env.features(k) for k in train_env.pool])
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-9
        b0 = train_env.budget0 / train_env.median_s
        self.bscale = np.array([max(b0, 1e-9), train_env.cfg.n, max(b0 / train_env.cfg.n, 1e-9)])
        obs = train_env.reset(self.seed_offset)
        self.w = np.zeros((2, len(self.phi(obs))))
        held = range(self.seed_offset - self.curve_queues, self.seed_offset)
        base = self.seed_offset + 10_000 * self.seed
        self.curve = []
        for ep in range(self.episodes):
            if ep % self.curve_every == 0:
                self.curve.append((ep, greedy_return(train_env, self, held)))
            eps = self.eps0 + (self.eps1 - self.eps0) * ep / max(1, self.episodes - 1)
            obs = train_env.reset(base + ep)
            a = self._act(obs, rng, eps)
            done = False
            while not done:
                x = self.phi(obs)
                nobs, r, done, _ = train_env.step(a)
                target = r / train_env.cfg.alpha
                if not done:
                    na = self._act(nobs, rng, eps)
                    target += self.w[na] @ self.phi(nobs)
                self.w[a] += self.alpha * (target - self.w[a] @ x) * x
                if not done:
                    obs, a = nobs, na
        self.curve.append((self.episodes, greedy_return(train_env, self, held)))

    def __call__(self, obs, rng):
        return self._act(obs, rng, 0.0)


def baselines() -> list[Policy]:
    return [AllSAM2(), Random(), FirstCome(), RIPUThreshold(), RIPURanked()]
