"""Standard D-TTTS for the infinitely-many-armed setting.

Faithful implementation of Dynamic Top-Two Thompson Sampling
(Shang, Kaufmann & Valko, "A simple dynamic bandit algorithm for hyper-parameter
tuning", AutoML@ICML 2019), exposing the same ``suggest() / observe() /
best_config`` interface as :class:`imabo.IMABO` so it slots into the toy harness.

Key modelling points, exactly as in the paper:

* **Rewards must lie in [0,1].**  The toy objectives do *not* (they are sums over
  ``dim`` coordinates, e.g. sin1/garland in ~[0, 4] and rastrigin in ~[-185, 0]),
  so the caller supplies known bounds ``(reward_low, reward_high)`` and every
  observed reward is affinely mapped into [0,1] and clipped.  This is legitimate:
  D-TTTS *assumes* a known [0,1] support, and offline reward bounds are the
  standard way to meet that assumption on a benchmark.

* **Agrawal-Goyal binarization.**  A normalised reward ``p in [0,1]`` is turned
  into a Bernoulli outcome ``Y' ~ Bernoulli(p)``.  Each queried arm then keeps a
  Beta posterior ``Beta(1 + #successes, 1 + #failures)`` (Beta(1,1) = Uniform
  prior), so sampling an unqueried arm's prior yields a Uniform(0,1) draw.

* **Single pseudo-arm.**  Top-Two only depends on the *maximum* posterior sample,
  and the maximum of ``t - k`` independent Uniform(0,1) prior draws (one per
  not-yet-opened arm, with ``k`` = number of opened arms) is ``Beta(t - k, 1)``.
  All unopened arms therefore collapse into one virtual pseudo-arm whose sample
  is a single ``Beta(t - k, 1)`` draw -- deciding whether to open a fresh arm.
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


class DTTTS:
    """Dynamic Top-Two Thompson Sampling with a Uniform(0,1) reservoir prior.

    Args:
        search_space: same dict format as :class:`imabo.IMABO`.
        reward_low, reward_high: known bounds used to normalise rewards to [0,1]
            (required for the [0,1] assumption; rewards outside are clipped).
        seed: RNG seed.
        beta_tt: Top-Two challenger probability (0.5 = balanced, the paper default).
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        reward_low: float,
        reward_high: float,
        seed: int | None = 42,
        beta_tt: float = 0.5,
    ):
        if not (reward_high > reward_low):
            raise ValueError("reward_high must exceed reward_low")
        self.search_space_specs = search_space
        self.param_names = sorted(search_space.keys())
        self.distributions, self.param_types = create_search_space(search_space)
        self.rng = np.random.default_rng(seed)
        self.lo = float(reward_low)
        self.hi = float(reward_high)
        self.beta_tt = beta_tt

        # per-arm Beta parameters: alpha = 1 + successes, beta = 1 + failures
        self.alpha: list[float] = []
        self.beta: list[float] = []
        self.configs: list[dict[str, Any]] = []
        self.t = 0
        self._pending_idx: int | None = None
        self._pending_new: dict | None = None

    # ---- uniform reservoir over the box ----
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

    def _normalize(self, reward: float) -> float:
        p = (reward - self.lo) / (self.hi - self.lo)
        return min(1.0, max(0.0, p))

    # ---- posterior sample for an opened arm ----
    def _post_sample(self, i: int) -> float:
        return float(self.rng.beta(self.alpha[i], self.beta[i]))

    # ---- pseudo-arm: max of (t - k) Uniform(0,1) prior draws = Beta(t-k, 1) ----
    def _fresh_sample(self) -> float:
        k = len(self.alpha)
        unopened = max(self.t - k, 1)
        return float(self.rng.beta(unopened, 1))

    def _argmax_sample(self):
        """Return (index_or_'fresh', value) of the Top-One under one posterior draw."""
        best_i, best_v = None, -1.0
        for i in range(len(self.alpha)):
            v = self._post_sample(i)
            if v > best_v:
                best_i, best_v = i, v
        fv = self._fresh_sample()
        if fv > best_v:
            return "fresh", fv
        return best_i, best_v

    def suggest(self) -> dict[str, Any]:
        self.t += 1
        if len(self.alpha) < 2:
            return self._open_new()

        leader, _ = self._argmax_sample()
        if self.rng.random() < self.beta_tt:
            winner = leader
        else:
            # best challenger: resample until the top differs from the leader
            winner = leader
            for _ in range(20):
                cand, _ = self._argmax_sample()
                if cand != leader:
                    winner = cand
                    break
        if winner == "fresh":
            return self._open_new()
        self._pending_idx = int(winner)
        return self.configs[winner]

    def _open_new(self) -> dict[str, Any]:
        cfg = self._random_config()
        self._pending_new = cfg
        return cfg

    def observe(self, reward: float) -> None:
        p = self._normalize(reward)
        y = 1.0 if self.rng.random() < p else 0.0  # Agrawal-Goyal binarization
        if self._pending_new is not None:
            self.alpha.append(1.0 + y)
            self.beta.append(1.0 + (1.0 - y))
            self.configs.append(self._pending_new)
            self._pending_new = None
        elif self._pending_idx is not None:
            i = self._pending_idx
            self.alpha[i] += y
            self.beta[i] += (1.0 - y)
            self._pending_idx = None
        else:
            raise RuntimeError("observe() called before suggest()")

    # ---- recommendation: argmax posterior probability of being optimal ----
    # Paper (Shang, Kaufmann & Valko 2019), just above Algorithm 1:
    #   "we choose the ... strategy ... that outputs the arm with the largest
    #    posterior probability of being optimal.  ... we therefore recommend
    #    arm Ihat_t = argmax_{i in A} Pi_t(Theta_i)."
    # Computed by Monte Carlo over the REAL opened arms only (the pseudo-arm mu0,
    # Beta(t-k,1) ~ 1, is excluded -- "open a new arm" is not a valid final guess).
    def _prob_optimal(self, n_samples: int = 4000) -> np.ndarray:
        m = len(self.alpha)
        a = np.asarray(self.alpha)
        b = np.asarray(self.beta)
        # samples[r, i] ~ Beta(alpha_i, beta_i)
        samples = self.rng.beta(a, b, size=(n_samples, m))
        winners = np.argmax(samples, axis=1)
        counts = np.bincount(winners, minlength=m)
        return counts / n_samples

    @property
    def best_config(self) -> dict[str, Any] | None:
        if not self.alpha:
            return None
        p_opt = self._prob_optimal()
        return self.configs[int(np.argmax(p_opt))]

    @property
    def best_config_mean(self) -> dict[str, Any] | None:
        """Alternative recommendation: argmax posterior mean (for comparison)."""
        if not self.alpha:
            return None
        means = [a / (a + b) for a, b in zip(self.alpha, self.beta)]
        return self.configs[int(np.argmax(means))]

    @property
    def best_x(self) -> dict[str, Any] | None:
        return self.best_config
