"""Finite-armed Bernoulli bandit built from HPOBench's real RF tabular benchmark.

Ground truth: validation accuracy of a scikit-learn RandomForestClassifier on
OpenML task 9952 (phoneme), evaluated at HPOBench's max fidelity
(n_estimators=512, subsample=1) and averaged over the 5 seeds HPOBench
provides, for every point on its own discretized hyperparameter grid
(9 values of max_depth x 10 of max_features x 10 of min_samples_leaf x 10 of
min_samples_split = 9000 configs, see experiments/benchmarks/assets/rf_9952_grid.csv).

This module coarsens that grid to a smaller finite arm set and turns the
looked-up accuracy into a genuine Bernoulli success probability: pulling an
arm draws one Bernoulli(f(x)) sample rather than returning the (already
averaged) accuracy directly. This is the natural regime for the finite-armed
bandit indices already in this repo (kl_ucb, moss_anytime, ucb in imabo/moss.py).

Since the arm set is finite and precomputed, the optimum is known exactly and
regret is exact (no need to re-run the actual RandomForestClassifier).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "assets" / "rf_9952_grid.csv"

PARAM_NAMES = ["max_depth", "max_features", "min_samples_leaf", "min_samples_split"]


class RFTabularFiniteBenchmark:
    """Finite RF hyperparameter grid with Bernoulli(accuracy) rewards.

    Each of the 4 RF hyperparameters is coarsened to `n_values[name]` evenly
    spaced points taken from HPOBench's own discretized grid (never invented
    values). max_depth only has 9 distinct grid points in the source data, so
    asking for more than 9 there is a no-op (all 9 are kept).
    """

    def __init__(
        self,
        n_values: dict[str, int] | None = None,
        metric: str = "val_acc",
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        n_values = n_values or {
            "max_depth": 10,
            "max_features": 5,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
        }
        assert metric in ("val_acc", "test_acc")

        table = pd.read_csv(DATA_PATH)
        self.axes: dict[str, list[float]] = {}
        for name in PARAM_NAMES:
            unique_sorted = np.sort(table[name].unique())
            n = min(n_values.get(name, len(unique_sorted)), len(unique_sorted))
            idx = np.linspace(0, len(unique_sorted) - 1, n).round().astype(int)
            idx = sorted(set(idx.tolist()))
            self.axes[name] = unique_sorted[idx].tolist()

        mask = np.logical_and.reduce(
            [table[name].isin(self.axes[name]) for name in PARAM_NAMES]
        )
        grid = table[mask]

        self.lookup: dict[tuple, float] = {
            tuple(row[name] for name in PARAM_NAMES): row[metric]
            for row in grid.to_dict("records")
        }
        self.n_arms = len(self.lookup)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.best_config: dict[str, float] = max(self.lookup, key=self.lookup.get)
        self.best_config = dict(zip(PARAM_NAMES, self.best_config))
        self.max_value = self.lookup[tuple(self.best_config[n] for n in PARAM_NAMES)]

    def get_search_space(self) -> dict[str, dict]:
        """Categorical search space, one entry per hyperparameter."""
        return {name: {"choices": values} for name, values in self.axes.items()}

    def _key(self, x: dict) -> tuple:
        return tuple(x[name] for name in PARAM_NAMES)

    def mean_reward(self, x: dict) -> float:
        return self.lookup[self._key(x)]

    def __call__(self, x: dict, noise: bool = True) -> float:
        p = self.mean_reward(x)
        if not noise:
            return p
        reward = float(self.rng.binomial(1, p))
        if self.noise_std > 0:
            reward += self.rng.normal(0.0, self.noise_std)
        return reward

    def regret(self, x: dict) -> float:
        return self.max_value - self.mean_reward(x)


if __name__ == "__main__":
    bench = RFTabularFiniteBenchmark()
    print(f"n_arms = {bench.n_arms}")
    print(f"best_config = {bench.best_config}")
    print(f"max_value (val_acc) = {bench.max_value:.4f}")
