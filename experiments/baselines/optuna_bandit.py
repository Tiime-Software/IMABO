"""
OptunaBandit: wraps Optuna's TPE sampler in a bandit-style interface.

Suggests the same configuration k times, then averages the k observations
before telling Optuna the result.  This mimics the repeated-evaluation pattern
used in the HPO experiments.
"""

from typing import Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

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
        self.param_names = list(sorted(search_space.keys()))
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
                spec = self.param_specs[name]
                if spec.get("choices"):
                    self.config[name] = trial.suggest_categorical(name, spec["choices"])
                elif spec.get("int", False):
                    self.config[name] = trial.suggest_int(
                        name, int(spec["lower"]), int(spec["upper"]),
                        log=spec.get("log", False),
                    )
                else:
                    self.config[name] = trial.suggest_float(
                        name, spec["lower"], spec["upper"],
                        log=spec.get("log", False),
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
