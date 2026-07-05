"""Random search baseline with the suggest()/observe()/best_config interface.

Fully any-space: draws each configuration independently and uniformly from the
box (log-uniform for log axes, uniform integer, uniform categorical), exactly
like D-TTTS's reservoir but with no bandit/posterior layer on top.  This is the
honest floor -- it isolates what the bandit (MOSS) and oracle (TPE) layers of
IMABO, and the posterior machinery of D-TTTS, actually buy over blind sampling.

Recommendation: the configuration with the highest single observed (noisy)
reward seen so far -- the standard recommendation for random search, which never
re-evaluates a configuration.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.tpe import create_search_space


class RandomSearch:
    def __init__(self, search_space: dict[str, Any], seed: int | None = 42):
        self.param_names = sorted(search_space.keys())
        self.distributions, _ = create_search_space(search_space)
        self.rng = np.random.default_rng(seed)
        self._pending: dict | None = None
        self._best_cfg: dict | None = None
        self._best_reward = -np.inf

    def _random_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for name in self.param_names:
            dist = self.distributions[name]
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    cfg[name] = math.exp(
                        self.rng.uniform(math.log(dist.low), math.log(dist.high))
                    )
                else:
                    cfg[name] = self.rng.uniform(dist.low, dist.high)
            elif isinstance(dist, IntDistribution):
                cfg[name] = int(self.rng.integers(dist.low, dist.high + 1))
            elif isinstance(dist, CategoricalDistribution):
                cfg[name] = dist.choices[int(self.rng.integers(len(dist.choices)))]
        return cfg

    def suggest(self) -> dict[str, Any]:
        self._pending = self._random_config()
        return self._pending

    def observe(self, reward: float) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called before suggest()")
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_cfg = self._pending
        self._pending = None

    @property
    def best_config(self) -> dict[str, Any] | None:
        return self._best_cfg

    @property
    def best_x(self) -> dict[str, Any] | None:
        return self._best_cfg
