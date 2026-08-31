"""
OptunaBandit: wraps Optuna's TPE sampler in a bandit-style interface.

Suggests the same configuration k times, then averages the k observations
before telling Optuna the result.  This mimics the repeated-evaluation pattern
used in the HPO experiments.
"""

from typing import Any, Optional

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.samplers import TPESampler

from imabo.search_space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaBandit:
    """Optuna TPE wrapped as a bandit with k-observation averaging."""

    def __init__(
        self,
        search_space: dict[str, Any],
        k: int = 100,
        seed: int | None = None,
        **kwargs,
    ):
        self.param_specs = search_space
        # A dict, a suggestion function, or a ready-made SearchSpace: the same forms
        # IMABO accepts.
        self.space = (
            search_space
            if isinstance(search_space, SearchSpace)
            else SearchSpace(search_space)
        )
        self.param_names = self.space.names
        self.k = k
        self.config: Optional[dict[str, Any]] = None
        self.current_trial: Optional[optuna.Trial] = None
        self.current_observations: list[float] = []

        rng_seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        sampler = TPESampler(seed=rng_seed)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="optuna_bandit",
        )

    def suggest(self) -> dict[str, Any]:
        """Return the current config (or ask Optuna for a new one)."""
        if self.config is None:
            trial = self.study.ask()
            self.current_trial = trial
            self.config = {}
            for name in self.param_names:
                distribution = self.space.distributions[name]
                if isinstance(distribution, CategoricalDistribution):
                    self.config[name] = trial.suggest_categorical(
                        name, distribution.choices
                    )
                elif isinstance(distribution, IntDistribution):
                    self.config[name] = trial.suggest_int(
                        name, distribution.low, distribution.high,
                        step=distribution.step, log=distribution.log,
                    )
                else:
                    self.config[name] = trial.suggest_float(
                        name, distribution.low, distribution.high,
                        step=distribution.step, log=distribution.log,
                    )
        return self.config

    def observe(self, y: float, **kwargs) -> None:
        """Accumulate observation; tell Optuna once k samples collected."""
        if self.config is None or self.current_trial is None:
            raise ValueError("Must call suggest() before observe()")
        self.current_observations.append(y)
        if len(self.current_observations) >= self.k:
            avg = float(np.mean(self.current_observations))
            self.study.tell(self.current_trial, avg)
            self.current_trial = None
            self.config = None
            self.current_observations = []

    @property
    def best_config(self):
        """The configuration to report, under the name every optimizer here uses."""
        return self.suggest_best()

    def suggest_best(self) -> Optional[dict[str, Any]]:
        """Return the best configuration found so far."""
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            return self.study.best_params
        if self.config:
            return self.config
        return None
