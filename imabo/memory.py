"""Memory interfaces and in-memory storage for IMABO."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    """Dict-based in-memory storage."""

    def __init__(self, param_names: list[str]):
        self.memory: dict[ArmKey, ArmStats] = {}
        self.step_counter: int = 0
        self.param_names: list[str] = param_names

    def set(self, key: ArmKey, stats: ArmStats) -> None:
        self.memory[key] = stats

    def get_reward_frequency(self) -> float:
        total = sum(s.nb_rewarded + s.nb_pending for s in self.memory.values())
        rewarded = sum(s.nb_rewarded for s in self.memory.values())
        if total < 100:
            return 1.0
        return rewarded / total

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
        self.increment_step_counter()

    def observe(self, config: ArmConfig, reward: float) -> None:
        key = config_to_key(config, self.param_names)
        if key not in self.memory:
            self.memory[key] = ArmStats()
        stats = self.memory[key]
        stats.nb_rewarded += 1
        stats.mean_reward = (
            (1 - 1 / stats.nb_rewarded) * stats.mean_reward
            + reward / stats.nb_rewarded
        )
        stats.nb_pending = max(0, stats.nb_pending - 1)
