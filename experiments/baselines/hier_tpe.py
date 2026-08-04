"""Hier-TPE: Hier-MAB with its low-level value bandit replaced by a TPE.

Hier-MAB / AutoRAG-HP (see :mod:`experiments.baselines.hier_mab`) is two nested
UCB1 bandits: a HIGH-LEVEL one choosing which coordinate of the incumbent to
perturb this round, and one LOW-LEVEL bandit per axis choosing the value on it,
from a fixed finite grid of that axis.

This variant keeps the high-level bandit, the incumbent, and the axis credit
assignment exactly as they are, and swaps the low level for a *univariate TPE*
over the chosen axis: the good/bad split is taken over the configurations served
so far (top ``gamma_func`` fraction by empirical mean reward = "good"), a 1-D
Parzen pair is fit on that axis alone, and the value is the EI-argmax of
``n_ei_candidates`` draws from its ``l`` density
(:func:`imabo.tpe.univariate_tpe_values`).

Two consequences of that swap, both intended:

* **No axis discretisation.** The low-level bandits are what forced each axis to
  a finite grid, so Hier-MAB "cannot admit a value outside the declared list".
  A TPE samples the parameter's own domain, so continuous axes are proposed at
  arbitrary precision and ``n_points`` disappears. On an all-categorical space
  (the RF tabular grid) the two are on equal footing -- the "grid" is the axis
  itself -- so that experiment isolates the *proposal rule* and nothing else.
* **The value proposal sees the whole history, not just its own axis.** A
  low-level bandit only ever aggregates rewards per value of one axis; the TPE
  is fit on the good/bad split of full configurations. The credit assignment
  under test (a scalar reward attributed to one perturbed coordinate) is
  unchanged -- it still drives the high-level bandit and the incumbent.

The incumbent update stays Hier-MAB's greedy coordinate-wise step: after each
observation, the perturbed axis takes the value with the best empirical mean
*among the values served on that axis* -- the same rule as Hier-MAB's
``argmax(low[axis].means)``, computed from the observation log now that no
low-level bandit stores those means.

Same generator interface as the other baselines (``suggest`` / ``observe`` /
``best_config``), built from a repo-style search-space dict. Rewards are assumed
already normalised to [0, 1] by the caller.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.memory import config_to_key, key_to_config
from imabo.moss import UCB1
from imabo.tpe import (
    create_search_space,
    default_gamma,
    default_weights,
    univariate_tpe_values,
)


class HierTPE:
    """Hier-MAB's hierarchy with a univariate-TPE value proposal."""

    def __init__(
        self,
        search_space: dict[str, Any],
        alpha_high: float = 1.0,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        prior_weight: float = 1.0,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        seed: int | None = 42,
        **kwargs,
    ):
        """
        Args:
            search_space: repo-style search-space dict. Pass
                ``RFTabularFiniteBenchmark.get_search_space()`` directly.
            alpha_high: exploration multiplier of the axis-selecting bandit
                (Hier-MAB's own ``alpha_h = 1``).
            n_startup_trials: rounds served with a uniform value draw before the
                TPE takes over -- Optuna's ``TPESampler`` default, and IMABO's
                (via ``n_startup_trials`` random arms). It matters more here than
                in either: this method's history is confined to one-coordinate
                perturbations of the incumbent, so a Parzen pair fit on the first
                one or two observations is maximally peaked and the incumbent can
                lock in from round three.
            n_ei_candidates: draws from the 1-D ``l`` density per proposal; the
                EI-argmax among them is the value served.
            prior_weight: Parzen-estimator prior weight.
            gamma_func: n_served -> n_good, the good/bad split of the served
                configurations (default: top 30%, as in Optuna/IMABO).
            weights_func: Parzen observation weights.
            seed: RNG seed for the initial incumbent and the TPE draws.
            **kwargs: Ignored, so the same call site can build Hier-MAB (which
                takes ``n_points``) and this (which has no axis grid).
        """
        self.param_names = list(sorted(search_space.keys()))
        self.distributions, _ = create_search_space(search_space)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(
            self.rng.randint(0, 2**32 - 1) if seed is not None else None
        )

        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.prior_weight = prior_weight
        self.gamma_func = gamma_func or default_gamma
        self.weights_func = weights_func or default_weights

        self.high = UCB1(len(self.param_names), alpha=alpha_high)
        self.incumbent: dict[str, Any] = self._random_config()

        self._pending: Optional[tuple[int, str, Any]] = None
        self._n_observations = 0
        # Served config -> its rewards, and per-axis value -> its rewards (the
        # log that replaces the low-level bandits' running means).
        self._rewards: dict[tuple, list[float]] = {}
        self._axis_value_rewards: dict[str, dict[Any, list[float]]] = {
            name: {} for name in self.param_names
        }

    def _random_config(self) -> dict[str, Any]:
        """A uniform draw from the search space (the initial incumbent).

        Deliberately not named ``generate_random_config``: the experiment
        harness treats that attribute as "this optimizer exposes a proposal
        oracle to shadow-probe" (see
        rf_arm_distribution_experiment._oracle_propose), which is not what this
        method is.
        """
        config: dict[str, Any] = {}
        for name in self.param_names:
            dist = self.distributions[name]
            if isinstance(dist, CategoricalDistribution):
                config[name] = self.rng.choice(list(dist.choices))
            elif isinstance(dist, IntDistribution):
                config[name] = self.rng.randint(int(dist.low), int(dist.high))
            elif isinstance(dist, FloatDistribution):
                if dist.log:
                    config[name] = math.exp(
                        self.rng.uniform(math.log(dist.low), math.log(dist.high))
                    )
                else:
                    config[name] = self.rng.uniform(dist.low, dist.high)
            else:
                raise TypeError(f"unsupported distribution: {type(dist)!r}")
        return config

    def _good_bad_split(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Served configurations split into good/bad by empirical mean reward."""
        ranked = sorted(
            self._rewards, key=lambda k: float(np.mean(self._rewards[k])), reverse=True
        )
        n_good = max(1, min(self.gamma_func(len(ranked)), len(ranked) - 1))
        to_config = [key_to_config(k, self.param_names) for k in ranked]
        return to_config[:n_good], to_config[n_good:]

    def suggest(self) -> dict[str, Any]:
        """Perturb the incumbent on the axis the high-level bandit picks, with a
        value drawn by a univariate TPE over that axis.

        The first ``n_startup_trials`` rounds draw the value uniformly instead
        (Optuna's own warm-up rule; two distinct configurations are also the
        minimum for a good/bad split to exist at all). Hier-MAB's low-level
        bandits get the same effect from UCB1 pulling every value once.
        """
        axis_i = self.high.select()
        axis = self.param_names[axis_i]

        if self._n_observations < self.n_startup_trials or len(self._rewards) < 2:
            value = self._random_config()[axis]
        else:
            good, bad = self._good_bad_split()
            value = univariate_tpe_values(
                good_configs=good,
                bad_configs=bad,
                name=axis,
                distribution=self.distributions[axis],
                n_candidates=self.n_ei_candidates,
                rng=self.np_rng,
                prior_weight=self.prior_weight,
                weights_func=self.weights_func,
            )[0]

        config = {**self.incumbent, axis: value}
        self._pending = (axis_i, axis, value)
        return config

    def observe(self, reward: float) -> None:
        """Credit the reward to the perturbed axis, then take the greedy step.

        The scalar reward updates the high-level bandit's arm for that axis and
        the axis' own value log; the incumbent then adopts the best-mean value
        served on that axis -- Hier-MAB's ``argmax(low[axis].means)`` step, with
        the log standing in for the bandit that used to hold those means.
        """
        if self._pending is None:
            raise RuntimeError("observe() called before suggest()")
        axis_i, axis, value = self._pending
        reward = float(reward)

        self._n_observations += 1
        self.high.update(axis_i, reward)
        self._axis_value_rewards[axis].setdefault(value, []).append(reward)

        config = {**self.incumbent, axis: value}
        self._rewards.setdefault(config_to_key(config, self.param_names), []).append(
            reward
        )

        per_value = self._axis_value_rewards[axis]
        self.incumbent[axis] = max(
            per_value, key=lambda v: float(np.mean(per_value[v]))
        )
        self._pending = None

    @property
    def best_config(self) -> Optional[dict[str, Any]]:
        """The incumbent -- see :attr:`HierMAB.best_config` for why this method
        must not recommend the empirical argmax over served configurations."""
        if not self._rewards:
            return None
        return dict(self.incumbent)

    @property
    def best_x(self) -> Optional[dict[str, Any]]:
        return self.best_config
