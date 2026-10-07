"""Hindsight optimum by dynamic programming over (t, b_t) (67-proposal-revised, policies).

    V(t, b) = max over legal a of  alpha * q(x_t, a) + V(t + 1, b - cost(x_t, a)),
    V(N, b) = 0,  legal = {MANUAL, SAM2} if b > 0 else {SAM2}

with every future image, cost and SAM2 IoU known. Budgets are tracked in
tenths of a second (the label log's resolution), so the recursion is exact.
`queue_env.optimum()` enumerates all 2^N MANUAL subsets instead; the two must
agree, which `run_queue.py` checks on every evaluation queue.
"""
from __future__ import annotations

from functools import lru_cache

from .queue_env import MANUAL, SAM2, QueueEnv


def dp_optimum(env: QueueEnv, queue: list[str] | None = None) -> float:
    queue = queue if queue is not None else env.queue
    tenths = lambda s: int(round(10 * s))
    man = [tenths(env.manual_s(k)) for k in queue]
    rev = tenths(env.cfg.review_s)
    qs = [env.q(k, SAM2) for k in queue]
    n, alpha = len(queue), env.cfg.alpha

    @lru_cache(maxsize=None)
    def V(t: int, b: int) -> float:
        if t == n:
            return 0.0
        best = alpha * qs[t] + V(t + 1, b - rev)
        if b > 0:
            best = max(best, alpha * env.q(queue[t], MANUAL) + V(t + 1, b - man[t]))
        return best

    return V(0, tenths(env.budget0))
