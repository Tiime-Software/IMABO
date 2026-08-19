from __future__ import annotations

import random
from functools import partial
from typing import Callable

from imabo.memory import ArmStats, CurrentState, InMemoryStorage, Memory
from imabo.oracle import Oracle
from imabo.policy import AllocationPolicy
from imabo.search_space import SearchSpace
from imabo.types import ArmConfig, ArmKey


class IMABO:
    """Optimize a configuration online, from the rewards of serving it.

    Every round IMABO makes one binary choice: admit a new configuration proposed by
    the oracle, or serve a configuration it already knows, chosen by the policy.

        for t = 1, 2, ...
            if policy.expand(): x = oracle.suggest(); admit x
            else:               x = policy.select()
            serve x, observe its reward, update x's statistics

    A concrete algorithm is a choice of the two components.

        optimizer = IMABO(space)                       # IMOSS(beta=0.5) + TPEOracle()

    With anything else you want to pair:

        space = SearchSpace({
            "temperature": {"lower": 0.0, "upper": 2.0},
            "model": {"choices": ["haiku", "sonnet", "opus"]},
        })
        optimizer = IMABO(space, IMOSS(beta=0.5), TPEOracle(), seed=0)

        for _ in range(5000):
            config = optimizer.suggest()
            optimizer.observe(serve_one_request(config))

        print(optimizer.best_config)

    Args:
        search_space: A :class:`~imabo.search_space.SearchSpace`, or the dict spec to
            build one from.
        policy: How every pull is allocated -- ``A`` in Algorithm 1. Defaults to
            :class:`~imabo.policies.imoss.IMOSS`.
        oracle: What to admit when the policy asks for a new arm -- ``O`` in
            Algorithm 1. Defaults to :class:`~imabo.oracles.tpe_oracle.TPEOracle`.
        seed: Seeds one RNG shared by the engine, the policy and the oracle, which is
            what makes a run reproducible.
        memory: Where the active set lives. Defaults to an in-process dict.
    """

    def __init__(
        self,
        search_space: SearchSpace | dict,
        policy: AllocationPolicy | None = None,
        oracle: Oracle | None = None,
        *,
        seed: int | None = None,
        memory: Memory | None = None,
    ):
        if policy is None or oracle is None:
            # Imported here rather than at the top of the module: the named presets in
            # `oracles/` import this class, so a module-level import would be circular.
            from imabo.oracles.tpe_oracle import TPEOracle
            from imabo.policies.imoss import IMOSS

            policy = IMOSS() if policy is None else policy
            oracle = TPEOracle() if oracle is None else oracle

        self.space = (
            search_space
            if isinstance(search_space, SearchSpace)
            else SearchSpace(search_space)
        )
        self.rng = random.Random(seed)
        self.policy = policy
        self.oracle = oracle
        self.memory = (
            memory
            if memory is not None
            else InMemoryStorage(param_names=self.space.names)
        )
        self.oracle.setup(self.space, self.rng, self.memory)
        self.policy.setup(self.space, self.rng, self.memory)
        self.last_suggested: ArmConfig | None = None

    def suggest(self) -> ArmConfig:
        """The configuration to serve next."""
        state = self.memory.get_current_state()
        rewarded_arms = self.rewarded_arms(state)

        if self.policy.expand(state, rewarded_arms):
            config = self.oracle.suggest(
                state, rewarded_arms, partial(self.policy.score, state=state)
            )
        else:
            config = self.space.decode(self.policy.select(state, rewarded_arms))

        self.memory.pull_arm(self.space.encode(config))
        self.last_suggested = config
        return config

    def observe(self, reward: float, config: ArmConfig | None = None) -> None:
        """Record a reward.

        Args:
            reward: The observed reward.
            config: Which configuration the reward belongs to. Defaults to the last
                one :meth:`suggest` returned; pass it explicitly when feedback
                arrives out of order, as it does under delay.
        """
        if config is None:
            if self.last_suggested is None:
                raise RuntimeError("observe() called before suggest()")
            config = self.last_suggested
            self.last_suggested = None
        self.memory.observe(config, reward)

    def run(self, objective: Callable[[ArmConfig], float], n_rounds: int) -> ArmConfig:
        """Serve ``objective`` for ``n_rounds`` and return the reported configuration.

        A convenience for offline use. Online, drive :meth:`suggest` and
        :meth:`observe` from the request stream instead.
        """
        for _ in range(n_rounds):
            config = self.suggest()
            self.observe(objective(config))
        return self.best_config

    @staticmethod
    def rewarded_arms(state: CurrentState) -> list[tuple[ArmKey, ArmStats]]:
        """The (key, stats) pairs of arms with at least one reward.

        In admission order: the policy and the oracles break ties on the first arm
        admitted, so this must not be sorted.
        """
        return [
            (key, stats) for key, stats in state.arms.items() if stats.nb_rewarded > 0
        ]

    @property
    def state(self) -> CurrentState:
        """A snapshot of the active set and the step counter."""
        return self.memory.get_current_state()

    @property
    def best_config(self) -> ArmConfig | None:
        """The configuration the policy reports as its answer, or None if it has none."""
        state = self.memory.get_current_state()
        key = self.policy.best_arm(state, self.rewarded_arms(state))
        return None if key is None else self.space.decode(key)

    @property
    def best_x(self) -> ArmConfig | None:
        """Alias for :attr:`best_config`."""
        return self.best_config

    def propose(self) -> ArmConfig:
        """What the oracle would admit right now, without serving it.

        For diagnostics -- measuring proposal quality over a run. This advances the
        oracle's state (it draws from the shared RNG, and a model-based oracle may
        refit), so call it on a ``copy.deepcopy`` of the optimizer if the run must
        stay unaffected.
        """
        state = self.memory.get_current_state()
        return self.oracle.suggest(
            state, self.rewarded_arms(state), partial(self.policy.score, state=state)
        )
