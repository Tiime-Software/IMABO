from __future__ import annotations

from typing import Callable

from imabo.imabo import IMABO
from imabo.memory import ArmStats, CurrentState
from imabo.oracle import Oracle
from imabo.policies.imoss import IMOSS
from imabo.types import ArmConfig, ArmKey


class RandomOracle(Oracle):
    """Draw every new arm from ``P0``, ignoring the history.

    This is the standard infinitely many-armed bandit: the reservoir the paper's
    regret bound is stated against, and the floor the learned oracles are measured
    from. It satisfies the top-rho coverage condition with ``p_rho = rho`` by
    construction.
    """

    def suggest(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        score: Callable[[ArmKey], float],
    ) -> ArmConfig:
        return self.space.sample(self.rng)


class IMOSSRandom(IMABO):
    """IMOSS paired with the uniform oracle (paper: IMOSS-Random)."""

    def __init__(self, search_space, *, beta: float = 0.5, seed: int | None = None):
        super().__init__(search_space, IMOSS(beta=beta), RandomOracle(), seed=seed)
