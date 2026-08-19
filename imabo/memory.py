"""Memory interfaces and in-memory storage for IMABO."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from frozendict import frozendict

from imabo.types import ArmConfig, ArmKey


@dataclass
class ArmStats:
    """Statistics for a single arm (configuration)."""

    mean_reward: float = 0.0
    nb_rewarded: int = 0
    nb_pending: int = 0


@dataclass(frozen=True)
class CurrentState:
    """Snapshot of the optimizer state at a given time step."""

    nb_steps: int
    arms: frozendict[ArmKey, ArmStats]


class Decision(NamedTuple):
    """One explore step of a mutation oracle: which coordinate produced which arm.

    ``credited`` is what this decision has already contributed to the oracle's
    coordinate bandit, so a revision adjusts by the difference instead of voting
    twice.
    """

    coord: int
    arm_key: ArmKey
    parent_key: ArmKey
    credited: float | None = None


class CoordCredits(NamedTuple):
    """A coordinate bandit's counters: pulls, reward totals, and its own clock."""

    n: list[float]
    total: list[float]
    t: int


def config_to_key(config: ArmConfig, param_names: list[str]) -> ArmKey:
    """Convert a configuration dict to a hashable, order-independent key."""
    sorted_param_names = sorted(param_names)
    return tuple(config[name] for name in sorted_param_names)


def key_to_config(key: ArmKey, param_names: list[str]) -> ArmConfig:
    """Inverse of :func:`config_to_key`: rebuild a config dict from a key."""
    sorted_param_names = sorted(param_names)
    return dict(zip(sorted_param_names, key))


class Memory(ABC):
    """Abstract memory interface for IMABO.

    Implement this interface to plug custom storage backends (e.g., database,
    Redis) while keeping the optimizer logic unchanged.
    """

    @abstractmethod
    def set(self, key: ArmKey, stats: ArmStats) -> None:
        """Set arm stats for a configuration key."""

    @abstractmethod
    def get_reward_frequency(self) -> float:
        """Return the fraction of trials that have received a reward."""

    @abstractmethod
    def increment_step_counter(self) -> None:
        """Increment the global step counter."""

    @abstractmethod
    def get_current_state(self) -> CurrentState:
        """Return a snapshot of all arms and the current step count."""

    @abstractmethod
    def pull_arm(self, arm_key: ArmKey) -> None:
        """Record that an arm has been pulled (pending observation)."""

    @abstractmethod
    def observe(self, config: ArmConfig, reward: float) -> None:
        """Record an observed reward for a configuration."""


class InMemoryStorage(Memory):
    """Dict-based in-memory storage.

    Implements the `Memory` contract, then adds the blocks that specific oracles of
    this package read -- see the comment before them at the end of the class.
    """

    def __init__(self, param_names: list[str]):
        self.memory: dict[ArmKey, ArmStats] = {}
        self.step_counter: int = 0
        self.param_names: list[str] = param_names
        # Running totals kept in sync by set/pull_arm/observe so that
        # get_reward_frequency() is O(1) instead of O(K).  Scoring MOSS for
        # every arm on every step previously made this O(K^2) per step.
        self._total_rewarded: int = 0
        self._total_pending: int = 0
        # Backing store for the oracle blocks at the end of this class. Empty and
        # untouched unless an oracle that needs them is in use.
        self._rewards: dict[ArmKey, list[float]] = {}
        self._decisions: list[Decision] = []
        self._coord_credits: CoordCredits | None = None

    def set(self, key: ArmKey, stats: ArmStats) -> None:
        if key in self.memory:
            old = self.memory[key]
            self._total_rewarded -= old.nb_rewarded
            self._total_pending -= old.nb_pending
        self.memory[key] = stats
        self._total_rewarded += stats.nb_rewarded
        self._total_pending += stats.nb_pending

    def get_reward_frequency(self) -> float:
        total = self._total_rewarded + self._total_pending
        if total < 100:
            return 1.0
        return self._total_rewarded / total

    def increment_step_counter(self) -> None:
        self.step_counter += 1

    def get_current_state(self) -> CurrentState:
        return CurrentState(
            nb_steps=self.step_counter,
            arms=frozendict(self.memory),
        )

    def pull_arm(self, key: ArmKey) -> None:
        if key not in self.memory:
            self.memory[key] = ArmStats()
        self.memory[key].nb_pending += 1
        self._total_pending += 1
        self.increment_step_counter()

    def observe(self, config: ArmConfig, reward: float) -> None:
        key = config_to_key(config, self.param_names)
        if key not in self.memory:
            self.memory[key] = ArmStats()
        stats = self.memory[key]
        stats.nb_rewarded += 1
        self._total_rewarded += 1
        stats.mean_reward = (
            (1 - 1 / stats.nb_rewarded) * stats.mean_reward
            + reward / stats.nb_rewarded
        )
        if stats.nb_pending > 0:
            stats.nb_pending -= 1
            self._total_pending -= 1
        self._rewards.setdefault(key, []).append(float(reward))

    # ------------------------------------------------------------------
    # Below: not part of the `Memory` contract.
    #
    # These serve specific oracles of this package, named for each one. A custom
    # backend needs a block only if it is paired with that oracle, and the oracle
    # says so and fails loudly when the block is missing. Nothing in `Memory`, in
    # `IMABO` or in the policies reads any of it.
    # ------------------------------------------------------------------

    # For every oracle that needs the individual rewards rather than their mean:
    # `MutateKLTPEOracle` (its arm means) and `TabPFNOracle(fit_granularity="pull")`
    # (one training row per pull).

    def get_rewards(self, key: ArmKey) -> list[float]:
        """Individual rewards observed for a configuration, in arrival order."""
        return self._rewards.get(key, [])

    # For `MutateKLTPEOracle` only: the decisions it has taken, and the credits of
    # the bandit it runs over coordinates.

    def add_decision(self, decision: Decision) -> None:
        """Record which coordinate produced which arm, from which parent."""
        self._decisions.append(decision)

    def get_decisions(self) -> list[Decision]:
        """Every recorded decision, in the order it was made."""
        return self._decisions

    def save_decisions(self, decisions: list[Decision]) -> None:
        """Replace the recorded decisions, whose credits have been revised."""
        self._decisions = list(decisions)

    def get_coord_credits(self) -> CoordCredits | None:
        """The coordinate bandit's counters, or None if it never saved any."""
        return self._coord_credits

    def save_coord_credits(self, credits: CoordCredits) -> None:
        """Store the coordinate bandit's counters."""
        self._coord_credits = credits
