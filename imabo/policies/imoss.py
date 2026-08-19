from __future__ import annotations

import math
import random

from imabo.memory import ArmStats, CurrentState, Memory
from imabo.policy import AllocationPolicy
from imabo.search_space import SearchSpace
from imabo.types import ArmKey


def anytime_moss_index(
    mean_reward: float,
    n_pulls: float,
    n_arms: float,
    t: float,
    alpha: float = 0.1,
) -> float:
    """The anytime MOSS index.

    ``mu_hat + sqrt((1 + alpha) / 2 * max(0, log(t / (n_arms * n_pulls))) / n_pulls)``

    The exploration bonus shrinks as the arm's own pull count grows and rises with
    the size of the active set, so admitting a new arm raises the bonus of every arm
    already in it.

    Appendix delayed variant is this same formula with effective counts
    substituted for ``t``, ``n_arms`` and ``n_pulls`` -- see :meth:`IMOSS.score`.
    """
    n_pulls = max(n_pulls, 1)
    bonus = math.sqrt(
        (1 + alpha) / 2 * max(0.0, math.log(t / (n_arms * n_pulls))) / n_pulls
    )
    return mean_reward + bonus


class IMOSS(AllocationPolicy):
    """Grow the active set as ``t ** beta``, serving the arm of highest MOSS index.

    Each round: admit a new arm while ``|M_t| < t ** beta``, otherwise serve the
    ``argmax`` of :func:`anytime_moss_index` over the arms that have returned a reward. The
    exponent trades discovery against estimation -- too slow and a good region is
    found late, too fast and no arm gets enough observations to rank.

    Example:
        >>> optimizer = IMABO(space, IMOSS(beta=0.5), TPEOracle())

    Args:
        beta: Active-set growth exponent, in (0, 1). The paper uses 0.5 throughout.
        alpha: Confidence parameter scaling the exploration bonus.
        n_warmup: Configurations drawn from ``P0`` before the index takes over
            (``Ns`` in Algorithm 2). They are admitted when the policy is set up and
            served one at a time until each has returned a reward.
        max_pending: How many times an arm that has never returned a reward may be
            re-served. Only binds under delayed feedback, where it is what stops the
            active set growing faster than the rewards can resolve it.
        min_rewards: Rewards an arm needs before :meth:`best_arm` will report it,
            falling back to any rewarded arm if none qualifies.
        delayed: Run the delay-aware rules of Appendix C.1, counting each pending
            pull as a fraction ``p`` of a reward. With immediate feedback the two
            agree exactly.
    """

    def __init__(
        self,
        beta: float = 0.5,
        alpha: float = 0.1,
        n_warmup: int = 10,
        max_pending: int = 20,
        min_rewards: int = 1,
        delayed: bool = False,
    ):
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        self.beta = beta
        self.alpha = alpha
        self.n_warmup = n_warmup
        self.max_pending = max_pending
        self.min_rewards = min_rewards
        self.delayed = delayed

    def setup(self, space: SearchSpace, rng: random.Random, memory: Memory) -> None:
        super().setup(space, rng, memory)
        for _ in range(self.n_warmup):
            memory.set(space.encode(space.sample(rng)), ArmStats())

    def expand(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> bool:
        # Arms that have never returned a reward are served before anything else, so
        # the index is never asked to rank an arm it knows nothing about.
        if self._unrewarded(state) is not None:
            return False
        return len(state.arms) < self._horizon(state) ** self.beta

    def select(
        self, state: CurrentState, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> ArmKey:
        unrewarded = self._unrewarded(state)
        if unrewarded is not None:
            return unrewarded
        if not rewarded_arms:
            return self.space.encode(self.space.sample(self.rng))
        return max(rewarded_arms, key=lambda arm: self.score(arm[0], state))[0]

    def score(self, key: ArmKey, state: CurrentState) -> float:
        stats = state.arms[key]
        if self.delayed:
            # Appendix: count each pending pull as a fraction `p` of a reward,
            # then read Equation (1) off those effective counts. The active-set size
            # becomes the size the schedule targets rather than the size reached.
            horizon = self._horizon(state)
            frequency = self.memory.get_reward_frequency()
            return anytime_moss_index(
                stats.mean_reward,
                stats.nb_rewarded + frequency * stats.nb_pending + 1,
                horizon**self.beta,
                horizon,
                self.alpha,
            )
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
        reliable = [
            arm for arm in state.arms.items() if arm[1].nb_rewarded >= self.min_rewards
        ]
        candidates = reliable or rewarded_arms
        if not candidates:
            return None
        return max(candidates, key=lambda arm: arm[1].mean_reward)[0]

    def _horizon(self, state: CurrentState) -> float:
        """The round count the schedule and the index run on."""
        if not self.delayed:
            return state.nb_steps
        nb_rewarded = sum(s.nb_rewarded for s in state.arms.values())
        nb_pending = sum(s.nb_pending for s in state.arms.values())
        return nb_rewarded + self.memory.get_reward_frequency() * nb_pending

    def _unrewarded(self, state: CurrentState) -> ArmKey | None:
        """The first admitted arm still owed a reward and under the pending cap."""
        for key, stats in state.arms.items():
            if stats.nb_rewarded == 0 and stats.nb_pending < self.max_pending:
                return key
        return None
