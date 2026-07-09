"""UCB-AIR: UCB with the Arm-Increasing Rule (Wang, Audibert & Munos 2008).

Infinitely-many-armed bandit strategy for a reservoir of arms whose mean-reward
tail satisfies  P(mu >= mu* - x) = Theta(x^beta)  near the top (the beta-
regularity used throughout the IMAB literature).  It has two moving parts:

  1. Arm-Increasing Rule (AIR).  Rather than reveal all (infinitely many) arms,
     keep an *active set* whose size grows with time:

         K(t) = ceil( t ** (beta / (beta + 1)) )      if beta < 1
         K(t) = ceil( t ** (1 / 2) )                   if beta >= 1

     (For beta >= 1 the optimal exponent saturates at 1/2 -- drawing arms faster
     buys nothing because near-optimal arms are common.)  Whenever the current
     step index t makes K(t) exceed the number of arms already drawn, we draw a
     fresh arm uniformly from the reservoir and add it to the active set.

  2. UCB index on the active set.  Among the active arms, pull the one maximising
     an anytime UCB1 bound

         mu_hat_i + sqrt( 2 * log(t) / n_i ) .

     A never-pulled active arm has an infinite index, so a newly drawn arm is
     always tried once immediately -- this is how AIR injects breadth.

Same generator-based interface as the other baselines (suggest / observe /
best_config).  Rewards are assumed already normalised to [0, 1] by the caller
(the toy harness does this); the recommendation is the active arm with the
highest empirical mean among those pulled at least once.

Reference: Y. Wang, J.-Y. Audibert, R. Munos, "Algorithms for Infinitely
Many-Armed Bandits", NeurIPS 2008.
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


class _Arm:
    __slots__ = ("cfg", "key", "n", "sum")

    def __init__(self, cfg: dict[str, Any], key: tuple):
        self.cfg = cfg
        self.key = key
        self.n = 0          # number of observed rewards
        self.sum = 0.0      # sum of observed rewards

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n > 0 else 0.0


class UCBAIR:
    """UCB with the Arm-Increasing Rule for infinitely-many-armed bandits."""

    def __init__(
        self,
        search_space: dict[str, Any],
        beta: float = 1.0,
        ucb_c: float = 1.0,
        seed: int | None = 42,
    ):
        """
        Args:
            search_space: repo-style search-space dict.
            beta: reservoir tail exponent.  K(t) = ceil(t**(beta/(beta+1))) for
                beta < 1, else ceil(sqrt(t)).  beta=1 is the neutral default
                (uniform-ish reservoir), matching the toy setting where the
                true tail exponent is unknown.
            ucb_c: multiplier on the UCB bonus sqrt(2 log t / n).
            seed: RNG seed for drawing reservoir arms.
        """
        self.param_names = sorted(search_space.keys())
        self.distributions, _ = create_search_space(search_space)
        self.beta = beta
        self.ucb_c = ucb_c
        self.rng = np.random.default_rng(seed)

        self.arms: list[_Arm] = []
        self.t = 0                     # step counter (number of suggest() calls)
        self._pending: _Arm | None = None

    # -- reservoir ----------------------------------------------------------
    def _draw_config(self) -> dict[str, Any]:
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

    def _target_num_arms(self, t: int) -> int:
        exponent = self.beta / (self.beta + 1.0) if self.beta < 1.0 else 0.5
        return max(1, math.ceil(t ** exponent))

    def _maybe_add_arm(self) -> None:
        """Arm-Increasing Rule: grow the active set toward K(t)."""
        target = self._target_num_arms(self.t)
        while len(self.arms) < target:
            cfg = self._draw_config()
            key = tuple((n, cfg[n]) for n in self.param_names)
            self.arms.append(_Arm(cfg, key))

    # -- generator interface -----------------------------------------------
    def suggest(self) -> dict[str, Any]:
        self.t += 1
        self._maybe_add_arm()

        # any active arm never pulled -> index = +inf, pull it (breadth)
        unpulled = [a for a in self.arms if a.n == 0]
        if unpulled:
            arm = unpulled[0]
        else:
            log_t = math.log(max(2.0, self.t))
            arm = max(
                self.arms,
                key=lambda a: a.mean + self.ucb_c * math.sqrt(2.0 * log_t / a.n),
            )
        self._pending = arm
        return arm.cfg

    def observe(self, reward: float) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called before suggest()")
        self._pending.n += 1
        self._pending.sum += reward
        self._pending = None

    @property
    def best_config(self) -> dict[str, Any] | None:
        pulled = [a for a in self.arms if a.n > 0]
        if not pulled:
            return None
        return max(pulled, key=lambda a: a.mean).cfg

    @property
    def best_x(self) -> dict[str, Any] | None:
        return self.best_config


class MOSSAIR(UCBAIR):
    """AIR schedule (same as UCB-AIR) but with the MOSS index instead of UCB1.

    This isolates the *index* choice: MOSSAIR and UCBAIR draw arms on the
    identical arm-increasing schedule K(t), so any performance difference is
    attributable to MOSS-anytime vs the UCB1 bonus, not to how fast the active
    set grows.  The MOSS bonus is an O(1) radius calibrated for rewards in
    [0, 1] (same assumption the toy harness enforces by normalising).
    """

    def suggest(self) -> dict[str, Any]:
        self.t += 1
        self._maybe_add_arm()

        unpulled = [a for a in self.arms if a.n == 0]
        if unpulled:
            arm = unpulled[0]
        else:
            from imabo.moss import moss_anytime

            k = len(self.arms)
            arm = max(
                self.arms,
                key=lambda a: moss_anytime(
                    mean_reward=a.mean,
                    n_arms=k,
                    step_counter=self.t,
                    nb_rewarded_arm=a.n,
                ),
            )
        self._pending = arm
        return arm.cfg
