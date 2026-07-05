"""
RandomSearch: a true uniform-random-search baseline.

Every ``suggest()`` samples a fresh configuration uniformly from the search
space; it never re-pulls a config and keeps no bandit state. ``best_config``
returns the configuration with the highest *mean* observed reward.
"""

import math
import random
from collections import defaultdict
from typing import Any, Optional

from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.memory import config_to_key, key_to_config
from imabo.tpe import create_search_space


class RandomSearch:
    """Uniform random search over an IMABO-style search space."""

    def __init__(self, search_space: dict[str, Any], seed: int | None = 42, **kwargs):
        self.param_names = list(sorted(search_space.keys()))
        self.distributions, _ = create_search_space(search_space)
        self.rng = random.Random(seed)
        # config-key -> list of observed rewards
        self._rewards: dict = defaultdict(list)
        self._last: Optional[dict[str, Any]] = None

    def _sample(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for name in self.param_names:
            dist = self.distributions[name]
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    config[name] = math.exp(
                        self.rng.uniform(math.log(dist.low), math.log(dist.high))
                    )
                else:
                    config[name] = self.rng.uniform(dist.low, dist.high)
            elif isinstance(dist, IntDistribution):
                config[name] = self.rng.randint(dist.low, dist.high)
            elif isinstance(dist, CategoricalDistribution):
                config[name] = self.rng.choice(dist.choices)
        return config

    def suggest(self) -> dict[str, Any]:
        self._last = self._sample()
        return self._last

    def observe(self, reward: float) -> None:
        if self._last is None:
            raise RuntimeError("observe() called before suggest()")
        self._rewards[config_to_key(self._last, self.param_names)].append(reward)
        self._last = None

    @property
    def best_config(self) -> Optional[dict[str, Any]]:
        if not self._rewards:
            return None
        best_key = max(
            self._rewards,
            key=lambda k: sum(self._rewards[k]) / len(self._rewards[k]),
        )
        return key_to_config(best_key, self.param_names)

    @property
    def best_x(self) -> Optional[dict[str, Any]]:
        return self.best_config
