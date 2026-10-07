"""Hand-designed routing policies: the baselines a learned policy must beat.

Each maps an observation to one action per candidate. Column indices come
from env.SCENE_FEATURES plus the model-dependent block env._obs appends, so
keep COL in step with env.py.
"""
from __future__ import annotations

import numpy as np

from .env import AUTO, MANUAL, SCENE_FEATURES, SKIP

_TAIL = ("entropy", "low_conf", "pred_water", "iou_model_auto", "auto_water", "manual_min",
         "sess_frac", "sess_count", "batch_same_sess")
COL = {n: i for i, n in enumerate(SCENE_FEATURES + _TAIL)}


class Constant:
    def __init__(self, action: int):
        self.action, self.name = action, ("all_auto", "all_manual", "all_skip")[action]

    def __call__(self, obs, rng):
        return [self.action] * len(obs["keys"])


class Random:
    name = "random"

    def __call__(self, obs, rng):
        return list(rng.integers(0, 3, size=len(obs["keys"])))


class Disagreement:
    """Send a scene to a person when the auto mask and the current model
    disagree (IoU below tau); otherwise accept the auto mask. Label-free:
    disagreement is a proxy for the auto labeler being wrong."""

    def __init__(self, tau: float = 0.5):
        self.tau, self.name = tau, f"disagree<{tau}"

    def __call__(self, obs, rng):
        c = obs["candidates"][:, COL["iou_model_auto"]]
        return [MANUAL if v < self.tau else AUTO for v in c]


class Uncertainty:
    """RIPU-flavoured greedy rule at image level: the model's `frac` most
    uncertain candidates go to a person, the rest to the auto labeler."""

    def __init__(self, frac: float = 0.5, skip_confident: bool = False):
        self.frac, self.skip = frac, skip_confident
        self.name = f"uncert{frac}" + ("+skip" if skip_confident else "")

    def __call__(self, obs, rng):
        e = obs["candidates"][:, COL["entropy"]]
        n = int(round(self.frac * len(e)))
        top = set(np.argsort(-e)[:n].tolist())
        rest = SKIP if self.skip else AUTO
        return [MANUAL if i in top else rest for i in range(len(e))]


BASELINES = [Constant(AUTO), Constant(MANUAL), Constant(SKIP), Random(),
             Disagreement(0.5), Disagreement(0.7), Uncertainty(0.5), Uncertainty(0.5, True)]
