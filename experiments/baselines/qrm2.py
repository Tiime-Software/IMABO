"""QRM2: parameter-free quantile-regret minimisation (Roy Chaudhuri &
Kalyanakrishnan, UAI 2018), Algorithm 2.

A doubling wrapper around a finite-armed MOSS inner loop, designed for
infinitely-many-armed bandits *without* knowing the quantile fraction rho:

    alpha = 0.347,  K_0 = {}
    for phase r = 1, 2, 3, ...:
        t_r = 2**r                 # phase horizon (pulls in this phase)
        n_r = ceil(t_r ** alpha)   # target pool size this phase
        draw n_r - |K_{r-1}| fresh reservoir arms, add to the pool K_r
        run MOSS(K_r, t_r)         # fresh fixed-horizon MOSS for t_r pulls

The pool of arms *accumulates* across phases, but the MOSS reward statistics
and horizon *restart* every phase — the per-phase regret in the paper's proof
is C*sqrt(n_r * t_r), the regret of an isolated length-t_r MOSS run on n_r arms,
which is only correct under a fresh restart.  The exponent alpha=0.347 minimises
the horizon dependence of the rho-regret and is deliberately much smaller than
UCB-AIR / MOSS-AIR's effective 0.5 (QRM2 opens far fewer arms: ~t^0.35 vs
~t^0.5), trading breadth for depth on each pool.

Same generator interface as the other baselines (suggest / observe /
best_config).  Rewards are assumed normalised to [0,1] by the caller.  The inner
index is the repo's `moss_anytime` (identical to MOSS-AIR's), invoked with the
fixed phase horizon t_r as its step counter so it behaves as fixed-horizon MOSS.
The recommendation is the arm with the best *lifetime* empirical mean (across all
phases), among arms pulled at least once.
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
from imabo.moss import moss_anytime


class _Arm:
    __slots__ = ("cfg", "n_life", "sum_life", "n_ph", "sum_ph")

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.n_life = 0        # lifetime pulls (for recommendation)
        self.sum_life = 0.0
        self.n_ph = 0          # within-current-phase pulls (for MOSS index)
        self.sum_ph = 0.0

    @property
    def mean_ph(self) -> float:
        return self.sum_ph / self.n_ph if self.n_ph > 0 else 0.0

    @property
    def mean_life(self) -> float:
        return self.sum_life / self.n_life if self.n_life > 0 else 0.0


class QRM2:
    def __init__(self, search_space: dict[str, Any], alpha: float = 0.347,
                 seed: int | None = 42):
        self.param_names = sorted(search_space.keys())
        self.distributions, _ = create_search_space(search_space)
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.arms: list[_Arm] = []
        self.t = 0                      # global pull counter
        self.r = 0                      # current phase index
        self.phase_horizon = 0          # t_r = 2**r
        self.phase_step = 0             # pulls done in the current phase
        self._pending: _Arm | None = None
        self._start_phase()             # sets up phase r=1

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

    def _start_phase(self) -> None:
        self.r += 1
        self.phase_horizon = 2 ** self.r
        self.phase_step = 0
        target = math.ceil(self.phase_horizon ** self.alpha)     # n_r = ceil(t_r^alpha)
        while len(self.arms) < target:
            self.arms.append(_Arm(self._draw_config()))
        for a in self.arms:                                       # restart MOSS stats
            a.n_ph = 0
            a.sum_ph = 0.0

    # -- generator interface -----------------------------------------------
    def suggest(self) -> dict[str, Any]:
        if self.phase_step >= self.phase_horizon:
            self._start_phase()
        self.t += 1
        self.phase_step += 1

        unpulled = [a for a in self.arms if a.n_ph == 0]          # round-robin init
        if unpulled:
            arm = unpulled[0]
        else:
            k = len(self.arms)
            arm = max(
                self.arms,
                key=lambda a: moss_anytime(
                    mean_reward=a.mean_ph,
                    n_arms=k,
                    step_counter=self.phase_horizon,              # fixed-horizon MOSS
                    nb_rewarded_arm=a.n_ph,
                ),
            )
        self._pending = arm
        return arm.cfg

    def observe(self, reward: float) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called before suggest()")
        a = self._pending
        a.n_ph += 1
        a.sum_ph += reward
        a.n_life += 1
        a.sum_life += reward
        self._pending = None

    @property
    def best_config(self) -> dict[str, Any] | None:
        pulled = [a for a in self.arms if a.n_life > 0]
        if not pulled:
            return None
        return max(pulled, key=lambda a: a.mean_life).cfg

    @property
    def best_x(self) -> dict[str, Any] | None:
        return self.best_config
