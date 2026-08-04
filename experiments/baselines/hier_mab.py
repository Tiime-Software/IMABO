"""Hier-MAB: the two-level hierarchical bandit of AutoRAG-HP.

Fu et al., "AutoRAG-HP: Automatic Online Hyper-Parameter Tuning for
Retrieval-Augmented Generation", Findings of EMNLP 2024, pp. 3875-3891.

Included as a *factored* baseline for OHPO. Unlike the infinitely-many-armed
baselines, Hier-MAB never materialises the product grid: it keeps an incumbent
configuration and perturbs one coordinate at a time, so its cost scales with
``sum_i |A_i|`` (about 30 on the RF tabular grid) rather than ``prod_i |A_i|``
(2250). That is why it is a meaningful competitor on the discrete OpenML
experiment even though it makes no claim about infinite arm sets, and why
"we handle infinitely many arms" is not by itself an adequate distinction
against it.

Structure (Section 3.2 of the paper):

  * one HIGH-LEVEL bandit over the hyperparameter axes, choosing which
    coordinate to perturb this round;
  * one LOW-LEVEL bandit per axis, choosing a value on that axis;
  * unselected coordinates retain their incumbent values.

Both levels use a UCB1 index with the paper's own ``alpha_h = alpha_l = 1``.
We set ``B = 1`` (no batching): IMABO commits one configuration per served
request, whereas the original work averaged reward over batches of ``B = 4``
queries.

STRUCTURAL ASSUMPTION UNDER TEST. Crediting a scalar reward to the single axis
that was perturbed presumes the reward is close to separable across
coordinates, so that coordinate-wise ascent from the incumbent reaches a good
configuration. On landscapes where good configurations depend on *combinations*
of hyperparameters this credit assignment is misleading and the search can
stall in a local optimum. That is the property this baseline exists to probe;
it is not a defect of the implementation.

Same generator interface as the other baselines (``suggest`` / ``observe`` /
``best_config``), built from a repo-style search-space dict -- so
``RFTabularFiniteBenchmark.get_search_space()`` can be passed straight in, and
its per-axis ``choices`` become the low-level arm sets directly. Rewards are
assumed already normalised to [0, 1] by the caller.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from imabo.memory import config_to_key, key_to_config
from imabo.moss import UCB1
from imabo.mutation import axis_values
from imabo.tpe import create_search_space

# The UCB1 index used by both levels. It moved to imabo.moss unchanged so the
# coordinate-selection bandit of imabo.coord_ucb.IMABOCoordUCB is literally this
# bandit rather than a copy of it.
_UCB1 = UCB1


class HierMAB:
    """Two-level hierarchical MAB over a factored hyperparameter space."""

    def __init__(
        self,
        search_space: dict[str, Any],
        n_points: int = 10,
        alpha_high: float = 1.0,
        alpha_low: float = 1.0,
        seed: int | None = 42,
        **kwargs,
    ):
        """
        Args:
            search_space: repo-style search-space dict. Pass
                ``RFTabularFiniteBenchmark.get_search_space()`` directly.
            n_points: values per continuous/integer axis when discretising.
                Ignored for categorical axes, which are used as given.
            alpha_high: exploration multiplier of the axis-selecting bandit.
            alpha_low: exploration multiplier of the per-axis value bandits.
            seed: RNG seed for the initial incumbent.
        """
        self.param_names = list(sorted(search_space.keys()))
        self.distributions, _ = create_search_space(search_space)
        self.rng = np.random.default_rng(seed)

        self.values: dict[str, list[Any]] = {
            name: axis_values(self.distributions[name], n_points)
            for name in self.param_names
        }

        self.high = _UCB1(len(self.param_names), alpha=alpha_high)
        self.low = {
            name: _UCB1(len(self.values[name]), alpha=alpha_low)
            for name in self.param_names
        }

        # incumbent: one value index per axis, drawn uniformly to start
        self.incumbent_idx: dict[str, int] = {
            name: int(self.rng.integers(len(self.values[name])))
            for name in self.param_names
        }

        self._pending: Optional[tuple[int, str, int]] = None
        self._rewards: dict = {}

    def _config_from(self, idx_map: dict[str, int]) -> dict[str, Any]:
        return {name: self.values[name][idx_map[name]] for name in self.param_names}

    def suggest(self) -> dict[str, Any]:
        axis_i = self.high.select()
        axis = self.param_names[axis_i]
        value_i = self.low[axis].select()

        idx_map = dict(self.incumbent_idx)
        idx_map[axis] = value_i
        config = self._config_from(idx_map)

        self._pending = (axis_i, axis, value_i)
        return config

    def observe(self, reward: float) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called before suggest()")
        axis_i, axis, value_i = self._pending

        # Credit the scalar reward to the perturbed axis and to the value
        # chosen on it -- the separability assumption this baseline probes.
        self.high.update(axis_i, reward)
        self.low[axis].update(value_i, reward)

        idx_map = dict(self.incumbent_idx)
        idx_map[axis] = value_i
        key = config_to_key(self._config_from(idx_map), self.param_names)
        self._rewards.setdefault(key, []).append(reward)

        # Greedy exploit step: accept the best-so-far value on this axis.
        self.incumbent_idx[axis] = int(np.argmax(self.low[axis].means))
        self._pending = None

    @property
    def best_config(self) -> Optional[dict[str, Any]]:
        """Recommend the incumbent: the per-axis argmax of low-level means.

        Deliberately *not* the empirical-best served configuration. Hier-MAB
        perturbs one coordinate at a time, so the number of distinct
        configurations it serves grows roughly linearly in T while each is
        pulled only a handful of times. An argmax of per-configuration
        empirical means over such short histories is dominated by single lucky
        pulls -- with Bernoulli rewards any configuration served once with
        reward 1 attains mean 1.0 -- which yields a recommendation far worse
        than the configuration the search actually converged to. In a smoke
        test on a synthetic separable landscape that rule returned a
        configuration of true mean 0.55 against an achievable 0.90. The
        incumbent aggregates every observation on each axis and is the
        quantity the method itself maintains, so it is the faithful
        recommendation and the one to use when reporting simple regret.
        """
        if not self._rewards:
            return None
        return self._config_from(self.incumbent_idx)

    @property
    def best_config_empirical(self) -> Optional[dict[str, Any]]:
        """Empirical-best served configuration, among those pulled often enough.

        Diagnostics only; see :attr:`best_config` for why the unfiltered
        empirical argmax is unsound for this method.
        """
        if not self._rewards:
            return None
        min_n = max(2, int(np.median([len(v) for v in self._rewards.values()])))
        eligible = {k: v for k, v in self._rewards.items() if len(v) >= min_n}
        if not eligible:
            eligible = self._rewards
        best_key = max(eligible, key=lambda k: float(np.mean(eligible[k])))
        return key_to_config(best_key, self.param_names)

    @property
    def best_x(self) -> Optional[dict[str, Any]]:
        return self.best_config
