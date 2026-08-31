"""
RandomSearch: a true uniform-random-search baseline.

Every ``suggest()`` samples a fresh configuration uniformly from the search
space; it never re-pulls a config and keeps no bandit state. ``best_config``
returns the configuration with the highest *mean* observed reward.
"""

import random
from collections import defaultdict
from typing import Any, Optional

from imabo.memory import config_to_key, key_to_config
from imabo.search_space import SearchSpace


class RandomSearch:
    """Uniform random search over an IMABO-style search space."""

    def __init__(self, search_space: dict[str, Any], seed: int | None = 42, **kwargs):
        # A dict, a suggestion function, or a ready-made SearchSpace: the same forms
        # IMABO accepts.
        self.space = (
            search_space
            if isinstance(search_space, SearchSpace)
            else SearchSpace(search_space)
        )
        self.param_names = self.space.names
        self.distributions = self.space.distributions
        self.rng = random.Random(seed)
        # config-key -> list of observed rewards
        self._rewards: dict = defaultdict(list)
        self._last: Optional[dict[str, Any]] = None

    def _sample(self) -> dict[str, Any]:
        """A draw from ``P0``: the same generator and the same order, delegated to the
        space instead of repeating its type ladder here."""
        return self.space.sample(self.rng)

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
