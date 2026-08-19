from __future__ import annotations

import math
import random
from typing import Literal

from imabo.memory import ArmStats, CurrentState, Memory
from imabo.policies.imoss import anytime_moss_index
from imabo.policy import AllocationPolicy
from imabo.search_space import SearchSpace
from imabo.types import ArmKey


def ucb(
    mean_reward: float,
    n_pulls: int,
    budget: int,
    c: float = 1.5,
    delta: float = 0.05,
) -> float:
    """An anytime UCB index with an explicit confidence level.

    ``mu_hat + c * sqrt(log(budget / delta / n_pulls) / n_pulls)``
    """
    n_pulls = max(1, n_pulls)
    return mean_reward + c * math.sqrt(math.log(budget / delta / n_pulls) / n_pulls)


class BudgetedUCB(AllocationPolicy):
    """Serve arms by a horizon-aware UCB index until the budget is spent.

    Not part of the paper. Where :class:`~imabo.policies.imoss.IMOSS` is anytime,
    this one is told the horizon up front and spends it: its confidence width is
    computed against the total budget rather than the current round, it caps how
    often one configuration may be served, and it reports the most-served arm
    rather than the best mean.

    Example:
        >>> optimizer = IMABO(space, BudgetedUCB(budget=500), TPEOracle())

    Args:
        budget: Total pulls the run is allowed, entering the confidence width as the
            horizon.
        ef: Schedule exponent -- admit a new arm while ``|rewarded| < t ** ef``.
        max_pulls_per_config: Cap on rewards per configuration. Once every arm is
            capped, the policy admits new ones regardless of the schedule.
        index: ``"ucb"`` for :func:`ucb`, or ``"kl_ucb"`` for the Bernoulli KL index,
            which is tighter when rewards sit near the top of [0, 1].
        n_warmup: Configurations drawn from ``P0`` before the index takes over.
    """

    def __init__(
        self,
        budget: int = 200,
        ef: float = 0.5,
        max_pulls_per_config: int = 1000,
        index: Literal["ucb", "kl_ucb"] = "ucb",
        n_warmup: int = 10,
        alpha: float = 0.1,
    ):
        if index not in ("ucb", "kl_ucb"):
            raise ValueError(f"index must be 'ucb' or 'kl_ucb', got {index!r}")
        self.budget = budget
        self.ef = ef
        self.max_pulls_per_config = max_pulls_per_config
        self.index = index
        self.n_warmup = n_warmup
        self.alpha = alpha

    def setup(self, space: SearchSpace, rng: random.Random, memory: Memory) -> None:
        super().setup(space, rng, memory)
        for _ in range(self.n_warmup):
            memory.set(space.encode(space.sample(rng)), ArmStats())

    def expand(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> bool:
        if self._unrewarded(state) is not None:
            return False
        all_saturated = len(rewarded_arms) > 0 and all(
            stats.nb_rewarded >= self.max_pulls_per_config for _, stats in rewarded_arms
        )
        return len(rewarded_arms) < state.nb_steps**self.ef or all_saturated

    def select(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> ArmKey:
        unrewarded = self._unrewarded(state)
        if unrewarded is not None:
            return unrewarded
        if not rewarded_arms:
            return self.space.encode(self.space.sample(self.rng))
        return max(rewarded_arms, key=lambda arm: self._index(arm[1], state))[0]

    def score(self, key: ArmKey, state: CurrentState) -> float:
        """The MOSS index, as IMOSS exposes it.

        Deliberately not :meth:`_index`: this is what an oracle ranks the active set
        by, and the paper's oracles expect the MOSS ordering whatever the policy
        serves by.
        """
        stats = state.arms[key]
        return anytime_moss_index(
            stats.mean_reward,
            stats.nb_rewarded,
            len(state.arms),
            state.nb_steps,
            self.alpha,
        )

    def best_arm(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> ArmKey | None:
        """The most-served arm, ties broken by mean.

        With a budget spent under a cap, how much evidence an arm accumulated is the
        more reliable signal; the best mean may be one lucky arm with few pulls.
        """
        if not rewarded_arms:
            return None
        return max(
            rewarded_arms, key=lambda arm: (arm[1].nb_rewarded, arm[1].mean_reward)
        )[0]

    def _index(self, stats: ArmStats, state: CurrentState) -> float:
        if self.index == "kl_ucb":
            # Lives with the KL-UCB coordinate bandit of Sec. 4.1.4, the paper's only
            # other user of it. Imported here so this module stays a leaf.
            from imabo.oracles.mutate_kl_tpe_oracle import kl_ucb

            score = kl_ucb(stats.mean_reward, stats.nb_rewarded, state.nb_steps)
        else:
            score = ucb(stats.mean_reward, stats.nb_rewarded, self.budget)
        # Push saturated arms out of contention rather than filtering them, so the
        # comparison stays a single pass.
        if stats.nb_rewarded >= self.max_pulls_per_config:
            score -= 1e9
        return score

    def _unrewarded(self, state: CurrentState) -> ArmKey | None:
        """The first arm still owed a reward. No pending cap, unlike IMOSS."""
        for key, stats in state.arms.items():
            if stats.nb_rewarded == 0:
                return key
        return None
