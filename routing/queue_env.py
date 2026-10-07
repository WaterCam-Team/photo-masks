"""The budgeted queue MDP (67-proposal-revised, "The MDP").

    cost = CostModel.fit(data, train_keys)
    env = QueueEnv(data, pool_keys, QueueConfig(n=12, rho=0.25, review_s=c_s), cost)
    obs = env.reset(seed)
    obs, r, done, info = env.step(MANUAL | SAM2)

One episode is a queue of N images and an annotator budget B. Each step shows one
image and the agent routes it:

    MANUAL  allowed only while b_t > 0; charged the image's measured seconds c(x),
            so the last MANUAL may overdraw. q = 1.
    SAM2    always allowed; charged c_s, the time to glance at a SAM2 mask and
            accept it. q = IoU(SAM2 mask, hand mask) on the water class.

Reward every step is alpha * q (alpha = 100); time enters only through the budget.
SegFormer is frozen and only supplies image features (`queue_features.py`).

State = the image's 7 label-free features, one of which is the predicted manual
time c_hat(x) from `CostModel` (a linear fit on label-free features, fitted on
training images only), plus budget left in median manual labels, images left
(including this one), and budget per remaining image. The agent sees c_hat,
never c, and never q, before deciding.

Budget. The proposal sets B = rho * N * c_bar ("enough time to hand-label a
quarter or half of the queue"). That only holds when reviews are free: with
c_s > 0 the reviews of the other images come out of the same B. `budget =
"review_inclusive"` adds them, B = rho*N*c_bar + (1 - rho)*N*c_s, so B again
buys rho*N hand labels plus a review of every other image.

`optimum()` is the hindsight ceiling: with every q and cost known, the best
action sequence under the same budget rule, found exactly by enumerating all
2^N MANUAL subsets (4096 at N = 12). `queue_dp.py` computes the same value by
dynamic programming over (t, b_t); `run_queue.py` checks they agree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAM2, MANUAL = 0, 1
ACTIONS = ("sam2", "manual")
# Written by queue_features.py; manual_pred_min is filled in by CostModel.
STATE_FEATURES = ("ripu", "entropy", "low_conf", "pred_water", "iou_seg_sam2",
                  "sam2_water", "manual_pred_min")
BUDGET_FEATURES = ("budget_left", "images_left", "budget_per_image")
# Regressors for c_hat (RL-project-next-steps, Phase 2): all label-free.
COST_FEATURES = ("edge_density", "pred_water", "pred_components", "pred_boundary_frac")


def load(path: Path) -> dict:
    """queue_scenes.json from queue_features.py."""
    return json.loads(Path(path).read_text())


class CostModel:
    """c_hat(x): least-squares fit of measured manual seconds on COST_FEATURES.

    Fitted only on the given (training) scenes that have a timing-valid log row.
    `mae()` reports the error on any other scenes with measured times, which is
    the proposal's metric A for c_hat.
    """

    def __init__(self, coef: np.ndarray, n_fit: int, median_s: float):
        self.coef, self.n_fit, self.median_s = coef, n_fit, median_s

    @staticmethod
    def _x(f: dict) -> np.ndarray:
        return np.array([1.0] + [float(f[n]) for n in COST_FEATURES])

    @classmethod
    def fit(cls, data: dict, keys: list[str]) -> "CostModel":
        sc = data["scenes"]
        rows = [(cls._x(sc[k]["features"]), sc[k]["manual_s"]) for k in keys
                if sc[k]["manual_s"] is not None]
        if len(rows) < len(COST_FEATURES) + 2:
            raise ValueError(f"c_hat needs at least {len(COST_FEATURES) + 2} timed training "
                             f"scenes, have {len(rows)}")
        X = np.array([x for x, _ in rows])
        y = np.array([s for _, s in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return cls(coef, len(rows), float(data["median_manual_s"]))

    def predict_s(self, features: dict) -> float:
        # A linear fit can go negative on an odd image; no label takes under a second.
        return max(1.0, float(self._x(features) @ self.coef))

    def mae(self, data: dict, keys: list[str]) -> tuple[float | None, int]:
        sc = data["scenes"]
        err = [abs(self.predict_s(sc[k]["features"]) - sc[k]["manual_s"])
               for k in keys if sc[k]["manual_s"] is not None]
        return (float(np.mean(err)) if err else None), len(err)


@dataclass
class QueueConfig:
    n: int = 12
    rho: float = 0.25
    review_s: float = 0.0                 # c_s, seconds charged for a SAM2 step
    alpha: float = 100.0
    budget: str = "proposal"              # "proposal" | "review_inclusive"


class QueueEnv:
    def __init__(self, data: dict, pool_keys: list[str], cfg: QueueConfig, cost: CostModel):
        if len(pool_keys) < cfg.n:
            raise ValueError(f"queue of {cfg.n} needs {cfg.n} scenes, pool has {len(pool_keys)}")
        if cfg.budget not in ("proposal", "review_inclusive"):
            raise ValueError(f"unknown budget rule {cfg.budget!r}")
        self.cfg, self.data, self.cost = cfg, data, cost
        self.scenes = data["scenes"]
        self.pool = sorted(pool_keys)
        self.median_s = float(data["median_manual_s"])
        self.budget0 = cfg.rho * cfg.n * self.median_s
        if cfg.budget == "review_inclusive":
            self.budget0 += (1.0 - cfg.rho) * cfg.n * cfg.review_s
        self._feat = {}

    # -- per-image quantities ----------------------------------------------------
    def features(self, key: str) -> np.ndarray:
        if key not in self._feat:
            f = dict(self.scenes[key]["features"])
            f["manual_pred_min"] = self.cost.predict_s(f) / 60.0
            self._feat[key] = np.array([f[n] for n in STATE_FEATURES], dtype=np.float64)
        return self._feat[key]

    def manual_s(self, key: str) -> float:
        """Charged time. Scenes without a timing-valid log row are charged the
        median (only possible on the interim 46-scene data; every R / T frame is timed)."""
        s = self.scenes[key]["manual_s"]
        return float(s) if s is not None else self.median_s

    def cost_s(self, key: str, action: int) -> float:
        return self.manual_s(key) if action == MANUAL else self.cfg.review_s

    def q(self, key: str, action: int) -> float:
        return 1.0 if action == MANUAL else float(self.scenes[key]["q_sam2"])

    # -- API ---------------------------------------------------------------------
    def reset(self, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        self.queue = [self.pool[i] for i in rng.permutation(len(self.pool))[:self.cfg.n]]
        self.t, self.budget = 0, self.budget0
        return self._obs()

    def legal(self) -> tuple[int, ...]:
        return (SAM2, MANUAL) if self.budget > 0 else (SAM2,)

    def _obs(self) -> dict:
        left = self.cfg.n - self.t
        b = self.budget / self.median_s                 # budget in median manual labels
        return {"key": self.queue[self.t], "image": self.features(self.queue[self.t]),
                "budget": np.array([b, left, b / left]), "legal": self.legal(),
                "t": self.t, "remaining": self.queue[self.t:]}

    def step(self, action: int) -> tuple[dict | None, float, bool, dict]:
        if action not in self.legal():
            raise ValueError(f"action {ACTIONS[action]} is not legal with budget {self.budget:.1f} s")
        k = self.queue[self.t]
        q = self.q(k, action)
        ratio = self.budget / (self.median_s * (self.cfg.n - self.t))
        self.budget -= self.cost_s(k, action)
        self.t += 1
        info = {"key": k, "action": ACTIONS[action], "q": q, "ratio": ratio,
                "q_sam2": float(self.scenes[k]["q_sam2"]), "budget_s": self.budget}
        done = self.t >= self.cfg.n
        return (None if done else self._obs()), self.cfg.alpha * q, done, info


def optimum(env: QueueEnv, queue: list[str] | None = None) -> tuple[float, np.ndarray]:
    """Hindsight-optimal return for a queue, and its MANUAL mask.

    A set of MANUAL images is feasible iff, walking the queue in order, the budget
    is still positive before each of them, i.e. everything charged before it
    (measured seconds for earlier MANUALs, c_s for earlier SAM2s) is below B.
    """
    queue = queue if queue is not None else env.queue
    n = len(queue)
    man = np.array([env.manual_s(k) for k in queue])
    qs = np.array([env.q(k, SAM2) for k in queue])
    masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    step_cost = np.where(masks, man, env.cfg.review_s)
    spent_before = np.cumsum(step_cost, axis=1) - step_cost
    feasible = np.all(~masks | (spent_before < env.budget0), axis=1)
    gain = masks.astype(float) @ (1.0 - qs)
    gain[~feasible] = -np.inf
    best = int(np.argmax(gain))
    return env.cfg.alpha * (qs.sum() + gain[best]), masks[best]
