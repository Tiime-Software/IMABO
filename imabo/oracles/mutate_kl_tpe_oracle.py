"""IMOSS whose explore oracle is a coordinate bandit over incumbent mutations.

:class:`MutateKLTPEOracle` keeps IMOSS's MOSS exploit phase unchanged and proposes
new arms with no surrogate model at all -- three cheap decisions per explore
step:

1. Which arm to improve. The incumbent: the arm with the highest empirical
   mean (:func:`imabo.oracles.candidate_pool.best_config`). Not the arm MOSS is pulling.
2. Which coordinate to change. A KL-UCB bandit over the ``d`` parameters
   (:class:`KLUCB`), credited with the empirical mean of the arm the
   mutation produced.
3. Which value to give it. A univariate TPE over that one coordinate: the
   EI-argmax of ``n_candidates`` draws from the 1-D ``l`` density fit on the
   good arms of
   :func:`imabo.oracles.parzen.split_good_bad`.

Credit assignment. Each proposed arm credits its coordinate exactly once -- later
exploit pulls of the same arm are not new decisions -- but the value that single
vote carries is the arm's *running mean* over every reward observed for it, not
its first reward. Between two oracle calls the exploit phase pulls the proposed
arm many more times (median 50 on the RF grid at beta=0.5, 96% of them above 10),
so this is the same target measured with roughly 7x less standard error.
:meth:`KLUCB.revise` keeps the vote count at one while its value is
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

import math
from typing import Any, Callable

import numpy as np

from imabo.imabo import IMABO
from imabo.memory import ArmStats, CoordCredits, CurrentState, Decision, key_to_config
from imabo.oracle import Oracle
from imabo.oracles.candidate_pool import best_config
from imabo.oracles.parzen import (
    default_gamma,
    default_weights,
    split_good_bad,
    univariate_tpe_values,
)
from imabo.policies.imoss import IMOSS
from imabo.types import ArmConfig, ArmKey


def kl_divergence(p: float, q: float) -> float:
    """KL divergence between two Bernoulli distributions, KL(p || q)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb(mean: float, pulls: int, t: int, c: float = 0.0) -> float:
    """KL-UCB index for Bernoulli bandits (alternative to MOSS).

    Returns the largest q in [mean, 1] such that::

        pulls * KL(mean, q) <= log(t) + c * log(log(t))

    Args:
        mean: Empirical mean of the arm.
        pulls: Number of pulls of the arm.
        t: Total number of steps (time horizon).
        c: Exploration constant (0 for asymptotic optimality).

    Returns:
        KL-UCB upper confidence bound.
    """
    pulls = max(pulls, 1)

    rhs = (math.log(t) + c * math.log(max(math.log(t), 1))) / pulls
    lo, hi = mean, 1.0 - 1e-12

    if mean > 1 - 1e-6:
        return 1.0

    for _ in range(40):
        mid = (lo + hi) / 2
        if kl_divergence(mean, mid) <= rhs:
            lo = mid
        else:
            hi = mid
    return lo



class KLUCB:
    """KL-UCB over a fixed finite set of choices.

    Used for :class:`MutateKLTPEOracle`'s coordinate bandit: the
    choices are the ``d`` parameters of the search space, and a choice is
    credited with the empirical mean of the arm that mutating it produced.

    Every choice is forced once (its index is ``+inf`` until credited), then the
    index of choice ``i`` is the largest ``q`` in ``[mean_i, 1]`` with
    ``n_i * KL(mean_i, q) <= log(t)`` (:func:`kl_ucb`), where ``t`` counts this
    bandit's own selections. Rewards must lie in [0, 1].

    Why KL and not a Hoeffding width: on rewards whose spread is far below 1
    (validation accuracies) the Hoeffding bonus dominates the mean gaps and the
    choice degenerates towards round-robin -- at ``t=112, n=25, mean=0.9``, the
    regime this bandit actually runs in, the widths are 0.61 (Hoeffding) against
    0.09 (KL), for a signal of 0.09-0.16 to resolve. Measured, the two are within
    noise of each other (KL better by 19.4 +- 18.7 on the RF grid), so this is a
    choice on direction rather than on evidence; the Hoeffding variant and the
    non-stationary alternatives that were tried (discounted-UCB, EXP3, EXP3.S,
    per-parent bandits -- all worse) live on the `oracles-archive` branch.
    """

    def __init__(self, n_choices: int):
        self.n = np.zeros(n_choices, dtype=np.float64)
        self.sum = np.zeros(n_choices, dtype=np.float64)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        unpulled = np.flatnonzero(self.n == 0)
        if unpulled.size:
            return int(unpulled[0])
        return int(np.argmax(self.indices()))

    def indices(self) -> np.ndarray:
        """The KL-UCB index of every choice, ``+inf`` if never credited."""
        pulled = self.n > 0
        index = np.full(self.n.size, np.inf)
        if not pulled.any():
            # Nothing credited yet: every choice is maximally uncertain. Votes can
            # lag well behind select() calls, so this is reachable.
            return index
        n = np.where(pulled, self.n, 1.0)
        mean = self.sum / n
        t = max(2, self.t)
        for i in np.flatnonzero(pulled):
            index[i] = kl_ucb(min(max(mean[i], 0.0), 1.0), float(self.n[i]), t)
        return index

    def update(self, idx: int, reward: float) -> None:
        self.n[idx] += 1.0
        self.sum[idx] += reward

    def revise(self, idx: int, delta: float) -> None:
        """Adjust a choice's accumulated reward WITHOUT counting a new vote.

        The coordinate bandit credits one vote per proposed arm but keeps
        sharpening its estimate of that arm as more pulls arrive, so the vote
        count stays at one while its value is corrected (see
        :class:`MutateKLTPEOracle`).
        """
        self.sum[idx] += delta

    @property
    def means(self) -> np.ndarray:
        return self.sum / np.where(self.n > 0, self.n, 1.0)



class MutateKLTPEOracle(Oracle):
    """
    Example:
        >>> optimizer = IMABO(space, IMOSS(beta=0.5), MutateKLTPEOracle())

    Args:
        min_arms: Rewarded arms required before mutating at all; below it the
            oracle returns uniform random configurations. A good/bad split needs
            at least two, and a 1-D density fit on one arm is meaningless.
        n_candidates: Values drawn from ``l`` per coordinate, ranked by ``l/g``;
            the best is proposed.
        categorical_distance_func: See :class:`imabo.oracles.tpe_oracle.TPEOracle`.
            Fed to the univariate TPE that picks the mutated coordinate's new
            value (:func:`imabo.oracles.parzen.univariate_tpe_values`); ``None``
            (default) auto-derives it from the search space.
    """

    def __init__(
        self,
        n_candidates: int = 24,
        min_arms: int = 10,
        prior_weight: float = 1.0,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        categorical_distance_func: dict[str, Callable[[Any, Any], float]] | None = None,
    ):
        self.n_candidates = n_candidates
        self.min_arms = min_arms
        self.prior_weight = prior_weight
        self.gamma_func = gamma_func or default_gamma
        self.weights_func = weights_func or default_weights
        self.categorical_distance_func = categorical_distance_func

    #: What this oracle needs from the memory beyond the `Memory` contract.
    MEMORY_BLOCK = (
        "get_rewards",
        "add_decision",
        "get_decisions",
        "save_decisions",
        "get_coord_credits",
        "save_coord_credits",
    )

    def setup(self, space, rng, memory) -> None:
        super().setup(space, rng, memory)
        missing = [
            name
            for name in self.MEMORY_BLOCK
            if not callable(getattr(memory, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(memory).__name__} is missing {', '.join(missing)}. This oracle "
                "keeps nothing of its own, so the memory must carry its decisions and "
                "its coordinate credits -- see the oracle blocks in InMemoryStorage."
            )

    def _bandit(self) -> KLUCB:
        """The coordinate bandit, rebuilt from the credits the memory holds."""
        bandit = KLUCB(len(self.space.names))
        credits = self.memory.get_coord_credits()
        if credits is not None:
            bandit.n = np.asarray(credits.n, dtype=np.float64)
            bandit.sum = np.asarray(credits.total, dtype=np.float64)
            bandit.t = credits.t
        return bandit

    def _save_bandit(self, bandit: KLUCB) -> None:
        self.memory.save_coord_credits(
            CoordCredits(n=bandit.n.tolist(), total=bandit.sum.tolist(), t=bandit.t)
        )

    def _arm_mean(self, key: ArmKey) -> float | None:
        """This arm's empirical mean, or None if it has no reward yet.

        Accumulated in arrival order, one reward at a time. Not ``sum()``, whose
        pairwise summation rounds differently and would move the bandit's credits.
        """
        rewards = self.memory.get_rewards(key)
        if not rewards:
            return None
        total = 0.0
        for reward in rewards:
            total += reward
        return total / len(rewards)

    def _refresh_credits(self, bandit: KLUCB) -> None:
        """Re-score every recorded decision from the current arm means.

        Decisions are grouped by the arm they proposed before being replayed, which
        is the order the bandit's credits were accumulated in. Float addition is not
        associative, so replaying them chronologically instead would drift.

        Cost is O(#decisions) = O(t**beta) once per oracle call.
        """
        grouped: dict[ArmKey, list[Decision]] = {}
        for decision in self.memory.get_decisions():
            grouped.setdefault(decision.arm_key, []).append(decision)

        replayed: list[Decision] = []
        for decisions in grouped.values():
            for decision in decisions:
                arm_mean = self._arm_mean(decision.arm_key)
                if arm_mean is None:
                    replayed.append(decision)
                    continue
                if decision.credited is None:
                    bandit.update(decision.coord, arm_mean)
                else:
                    bandit.revise(decision.coord, arm_mean - decision.credited)
                replayed.append(decision._replace(credited=arm_mean))
        self.memory.save_decisions(replayed)

    def suggest(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        score: Callable[[ArmKey], float],
    ) -> ArmConfig:
        """Propose a new arm: mutate the incumbent on a KL-UCB-chosen coordinate,
        with the new value drawn by a univariate TPE."""
        if len(rewarded_arms) < max(2, self.min_arms):
            return self.space.sample(self.rng)

        bandit = self._bandit()
        self._refresh_credits(bandit)

        parent = best_config(rewarded_arms, self.space.names)
        coord = bandit.select()
        self._save_bandit(bandit)
        name = self.space.names[coord]
        current = parent[name]

        good, bad = split_good_bad(rewarded_arms, score, self.gamma_func)
        ranked = univariate_tpe_values(
            good_configs=[key_to_config(k, self.space.names) for k, _ in good],
            bad_configs=[key_to_config(k, self.space.names) for k, _ in bad],
            name=name,
            distribution=self.space.distributions[name],
            n_candidates=self.n_candidates,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            weights_func=self.weights_func,
            categorical_distance_func=self.categorical_distance_func,
        )
        for value in ranked:
            if value == current:
                continue  # a mutation must change something
            mutant = {**parent, name: value}
            key = self.space.encode(mutant)
            self.memory.add_decision(
                Decision(
                    coord=coord,
                    arm_key=key,
                    parent_key=self.space.encode(parent),
                )
            )
            return mutant

        # Every draw reproduced the parent's current value on this axis: take a
        # uniform draw rather than return the parent unchanged.
        return self.space.sample(self.rng)


class IMOSSMutateKLTPE(IMABO):
    """IMOSS paired with the mutate-KLxTPE oracle."""

    def __init__(
        self,
        search_space,
        *,
        beta: float = 0.5,
        seed: int | None = None,
        **oracle: Any,
    ):
        super().__init__(
            search_space, IMOSS(beta=beta), MutateKLTPEOracle(**oracle), seed=seed
        )
