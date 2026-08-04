"""IMOSS whose explore oracle is a coordinate bandit over incumbent mutations.

:class:`IMABOCoordUCB` keeps IMABO's MOSS exploit phase unchanged and proposes
new arms with no surrogate model at all -- three cheap decisions per explore
step:

1. **Which arm to improve.** The incumbent: the arm with the highest empirical
   mean (:func:`imabo.mutation.best_config`). Not the arm MOSS is pulling.
2. **Which coordinate to change.** A KL-UCB bandit over the ``d`` parameters
   (:class:`imabo.moss.KLUCB1`), credited with the empirical mean of the arm the
   mutation produced.
3. **Which value to give it.** A univariate TPE over that one coordinate: the
   EI-argmax of ``n_ei_candidates`` draws from the 1-D ``l`` density fit on the
   good arms of IMABO's own :meth:`IMABO.tpe_split`
   (:func:`imabo.tpe.univariate_tpe_values`).

Credit assignment. Each proposed arm credits its coordinate exactly once -- later
exploit pulls of the same arm are not new decisions -- but the value that single
vote carries is the arm's *running mean* over every reward observed for it, not
its first reward. Between two oracle calls the exploit phase pulls the proposed
arm many more times (median 50 on the RF grid at beta=0.5, 96% of them above 10),
so this is the same target measured with roughly 7x less standard error.
:meth:`imabo.moss.KLUCB1.revise` keeps the vote count at one while its value is
corrected. Re-scoring happens lazily in :meth:`_refresh_credits` at the top of an
oracle call: nothing reads a bandit index between oracle calls, so revising on
every reward would be work no one observes (verified to give identical selections
against a per-reward implementation).

Already-open arms. The proposal is served as-is, even when it reproduces a
configuration already opened. Measured on the RF grid that is most of the explore
budget -- 95-97% of oracle calls return a known arm -- and it is self-sustaining:
a repeat does not grow ``|arms|``, so the ``|arms| < t**beta`` switch stays true
and the oracle fires again next round, giving it ~1100-1900 calls rather than the
~112 the rule nominally allows. The round is not wasted (it pulls a mutation of
the incumbent, a good arm), and each repeat registers its own vote, so one
configuration casts 17-32 votes. Those duplicate votes are what let the bandit's
intervals shrink and the choice commit. Forcing novelty instead measured the same
regret (515.8 vs 516.6) with ~5x fewer oracle calls -- a compute win, not a
quality win, and it is on the `oracles-archive` branch along with every other
rejected variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from imabo.memory import ArmStats, CurrentState, Memory, config_to_key, key_to_config
from imabo.moss import KLUCB1
from imabo.mutation import best_config
from imabo.optimizer import IMABO
from imabo.tpe import univariate_tpe_values
from imabo.types import ArmConfig, ArmKey


@dataclass
class _Decision:
    """One explore-step proposal: which coordinate produced which arm, from which
    parent.

    ``credited`` remembers what this decision has already contributed to the
    bandit, so a revision adjusts by the difference instead of casting a second
    vote (see :meth:`IMABOCoordUCB._cast`).
    """

    coord: int
    arm_key: ArmKey
    parent_key: ArmKey
    credited: float | None = None


class IMABOCoordUCB(IMABO):
    """
    Example:
        >>> optimizer = IMABOCoordUCB(
        ...     search_space={"x0": {"choices": [0, 1, 2, 3, 4, 5]}},
        ...     beta=0.5,
        ... )
        >>> for _ in range(100):
        ...     config = optimizer.suggest()
        ...     reward = evaluate(config)
        ...     optimizer.observe(reward)
        >>> print(optimizer.best_config)
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        seed: int | None = 42,
        n_min_rewarded: int = 1,
        max_nb_pending_per_unrewarded_arm: int = 20,
        n_startup_trials: int = 10,
        switch_strategy: str = "beta",
        beta: float = 0.5,
        memory: Memory | None = None,
        n_ei_candidates: int = 24,
        prior_weight: float = 1.0,
        multivariate: bool = True,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        min_arms_for_mutation: int = 10,
    ):
        """Initialize IMABOCoordUCB.

        Args:
            min_arms_for_mutation: Rewarded arms required before mutating at all;
                below it the oracle returns uniform random configurations. A
                good/bad split needs at least two, and a 1-D density fit on one
                arm is meaningless.
            n_ei_candidates: Values drawn from ``l`` per coordinate, ranked by
                ``l/g``; the best is proposed.

        The remaining arguments are IMABO's own. ``beta`` defaults to 0.5 here
        rather than IMABO's 0.8: measured across every oracle tried, 0.8 is much
        worse.
        """
        super().__init__(
            search_space=search_space,
            seed=seed,
            n_min_rewarded=n_min_rewarded,
            max_nb_pending_per_unrewarded_arm=max_nb_pending_per_unrewarded_arm,
            n_startup_trials=n_startup_trials,
            switch_strategy=switch_strategy,
            beta=beta,
            n_ei_candidates=n_ei_candidates,
            prior_weight=prior_weight,
            multivariate=multivariate,
            gamma_func=gamma_func,
            weights_func=weights_func,
            # The oracle proposes; IMABO must not fall back to its own TPE.
            use_tpe=True,
            memory=memory,
        )
        self.min_arms_for_mutation = min_arms_for_mutation
        self.coord_bandit = KLUCB1(len(self.param_names))
        # Arm key -> the decisions that proposed it, one per explore step that
        # landed on this config. Each is one vote, revised rather than repeated.
        self._pending_choice: dict[ArmKey, list[_Decision]] = {}
        # Arm key -> [reward count, reward total], i.e. its empirical mean.
        self._arm_scores: dict[ArmKey, list[float]] = {}

    def observe(self, reward: float) -> None:
        """Record the reward and update this arm's running mean.

        The vote that depends on it is re-scored at the next oracle call, which is
        the only place a bandit index is read -- see :meth:`_refresh_credits`.
        Kept here rather than read back from memory: O(1) per observation instead
        of an O(K) state snapshot.
        """
        key = (
            config_to_key(self.last_suggested, self.param_names)
            if self.last_suggested is not None
            else None
        )
        super().observe(reward)
        if key is None:
            return
        score = self._arm_scores.setdefault(key, [0.0, 0.0])
        score[0] += 1
        score[1] += float(reward)

    def _arm_mean(self, key: ArmKey) -> float | None:
        """This arm's empirical mean, or None if it has no reward yet."""
        score = self._arm_scores.get(key)
        return score[1] / score[0] if score and score[0] else None

    def _refresh_credits(self) -> None:
        """Re-score every registered decision from the current arm means.

        Cost is O(#decisions) = O(t**beta) once per oracle call, against O(1) per
        reward for the bookkeeping in :meth:`observe`.
        """
        for decisions in self._pending_choice.values():
            for decision in decisions:
                arm_mean = self._arm_mean(decision.arm_key)
                if arm_mean is not None:
                    self._cast(decision, arm_mean)

    def _cast(self, decision: _Decision, value: float) -> None:
        """Cast (or re-cast) a decision's single vote on the coordinate bandit."""
        if decision.credited is None:
            self.coord_bandit.update(decision.coord, value)
        else:
            self.coord_bandit.revise(decision.coord, value - decision.credited)
        decision.credited = value

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new arm: mutate the incumbent on a KL-UCB-chosen coordinate,
        with the new value drawn by a univariate TPE."""
        if len(rewarded_arms) < max(2, self.min_arms_for_mutation):
            return self.generate_random_config()

        self._refresh_credits()

        parent = best_config(rewarded_arms, self.param_names)
        coord = self.coord_bandit.select()
        name = self.param_names[coord]
        current = parent[name]

        good, bad = self.tpe_split(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )
        ranked = univariate_tpe_values(
            good_configs=[key_to_config(k, self.param_names) for k, _ in good],
            bad_configs=[key_to_config(k, self.param_names) for k, _ in bad],
            name=name,
            distribution=self.distributions[name],
            n_candidates=self.n_ei_candidates,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            weights_func=self.weights_func,
        )
        for value in ranked:
            if value == current:
                continue  # a mutation must change something
            mutant = {**parent, name: value}
            key = config_to_key(mutant, self.param_names)
            self._pending_choice.setdefault(key, []).append(
                _Decision(
                    coord=coord,
                    arm_key=key,
                    parent_key=config_to_key(parent, self.param_names),
                )
            )
            return mutant

        # Every draw reproduced the parent's current value on this axis: take a
        # uniform draw rather than return the parent unchanged.
        return self.generate_random_config()
