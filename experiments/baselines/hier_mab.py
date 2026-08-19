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

import math
from typing import Any, Optional

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo import SearchSpace
from imabo.memory import config_to_key, key_to_config


def axis_values(dist, n_points: int) -> list[Any]:
    """Enumerate the values Hier-MAB may place on one axis.

    Hier-MAB requires each axis to be an explicit finite set -- it cannot admit
    a value outside the declared list, which is one of its genuine limitations
    relative to an oracle that proposes into a continuous space. Categorical
    axes (what ``RFTabularFiniteBenchmark`` hands us) are used verbatim, so on
    the RF tabular grid the low-level arm sets are exactly the benchmark's own
    discretisation and nothing is invented. Continuous and integer axes are
    discretised on an evenly spaced (or log-spaced) grid so the baseline can
    also run on the LR/SVM and HotpotQA spaces.
    """
    if isinstance(dist, CategoricalDistribution):
        return list(dist.choices)
    if isinstance(dist, IntDistribution):
        lo, hi = int(dist.low), int(dist.high)
        if hi - lo + 1 <= n_points:
            return list(range(lo, hi + 1))
        if dist.log:
            raw = np.exp(np.linspace(math.log(max(lo, 1)), math.log(hi), n_points))
        else:
            raw = np.linspace(lo, hi, n_points)
        return sorted({int(round(v)) for v in raw})
    if isinstance(dist, FloatDistribution):
        # Plain Python floats, not np.float64: suggested configs get written
        # to JSON checkpoints by the callers (e.g. hotpotqa_experiment).
        if dist.log:
            # np.geomspace, not exp(linspace(log lo, log hi)): geomspace pins
            # both endpoints exactly, while the exp/log roundtrip lands a hair
            # outside the bounds (e.g. 9.9999...e-06 for lo=1e-05), which
            # strict validators like ConfigSpace reject.
            return [float(v) for v in np.geomspace(dist.low, dist.high, n_points)]
        return [float(v) for v in np.linspace(dist.low, dist.high, n_points)]
    raise TypeError(f"unsupported distribution for Hier-MAB axis: {type(dist)!r}")


class _UCB1:
    """UCB1 over a fixed finite set of choices, shared by both levels."""

    def __init__(self, n_choices: int, alpha: float = 1.0):
        self.n = np.zeros(n_choices, dtype=np.int64)
        self.sum = np.zeros(n_choices, dtype=np.float64)
        self.alpha = alpha
        self.t = 0

    def select(self) -> int:
        self.t += 1
        unpulled = np.flatnonzero(self.n == 0)
        if unpulled.size:
            return int(unpulled[0])
        mean = self.sum / self.n
        bonus = np.sqrt(self.alpha * 2.0 * math.log(max(2, self.t)) / self.n)
        return int(np.argmax(mean + bonus))

    def update(self, idx: int, reward: float) -> None:
        self.n[idx] += 1
        self.sum[idx] += reward

    @property
    def means(self) -> np.ndarray:
        return self.sum / np.maximum(self.n, 1)


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
        self.distributions = SearchSpace(search_space).distributions
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
