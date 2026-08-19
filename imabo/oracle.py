from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable

from imabo.memory import ArmStats, CurrentState, Memory
from imabo.search_space import SearchSpace
from imabo.types import ArmConfig, ArmKey


class Oracle(ABC):
    """Proposes the next configuration to admit.

    A minimal oracle is one method::

        class MyOracle(Oracle):
            def suggest(self, state, rewarded_arms, score):
                return self.space.sample(self.rng)
    """

    space: SearchSpace
    rng: random.Random
    memory: Memory

    def setup(self, space: SearchSpace, rng: random.Random, memory: Memory) -> None:
        self.space = space
        self.rng = rng
        self.memory = memory

    @abstractmethod
    def suggest(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        score: Callable[[ArmKey], float],
    ) -> ArmConfig:
        """The configuration to admit next.

        Args:
            state: Snapshot of the active set and the step counter.
            rewarded_arms: The (key, stats) pairs of arms with at least one reward,
                in admission order.
            score: How the policy ranks an arm. The paper's TPE oracles split the
                active set on this rather than on the empirical mean, so that an arm
                which looks strong after two pulls is not yet classified as reliably
                good. Oracles that do not rank the active set ignore it.
        """
