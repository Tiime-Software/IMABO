from __future__ import annotations

import random
from abc import ABC, abstractmethod

from imabo.memory import ArmStats, CurrentState, Memory
from imabo.search_space import SearchSpace
from imabo.types import ArmKey


class AllocationPolicy(ABC):
    """
    A minimal policy is two methods::

        class RoundRobin(AllocationPolicy):
            def expand(self, state, rewarded_arms):
                return len(state.arms) < 10

            def select(self, state, rewarded_arms):
                return rewarded_arms[state.nb_steps % len(rewarded_arms)][0]
    """

    space: SearchSpace
    rng: random.Random
    memory: Memory

    def setup(self, space: SearchSpace, rng: random.Random, memory: Memory) -> None:
        """Bind the run's space, RNG and memory.

        Override to seed an initial active set, as
        :class:`~imabo.policies.imoss.IMOSS` does for its warmup. Whatever an
        override draws from ``rng`` happens before the first round, so it comes
        first in the run's random stream.
        """
        self.space = space
        self.rng = rng
        self.memory = memory

    @abstractmethod
    def expand(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> bool:
        """Should this round admit a new arm from the oracle?"""

    @abstractmethod
    def select(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> ArmKey:
        """Which arm to serve this round.

        Normally an arm of ``state.arms``, but a policy with nothing worth serving
        yet may return the key of a fresh configuration; the memory admits it on
        being pulled.
        """

    def score(self, key: ArmKey, state: CurrentState) -> float:
        """How this policy ranks an arm, for any oracle that ranks the active set.

        This is not required to be the index :meth:`select` uses -- a policy may
        serve arms by one rule and still expose a different one here. The paper's
        oracles rank the active set by the MOSS index, whatever the
        policy serves by.
        """
        return state.arms[key].mean_reward

    def best_arm(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> ArmKey | None:
        """The arm to report as the answer -- the paper's ``x(T)``.

        The default is the highest empirical mean;
        :class:`~imabo.policies.budgeted_ucb.BudgetedUCB` reports the most-pulled
        arm instead.
        """
        if not rewarded_arms:
            return None
        return max(rewarded_arms, key=lambda arm: arm[1].mean_reward)[0]
