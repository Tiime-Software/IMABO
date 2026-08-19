from __future__ import annotations

import math
from typing import Any, Callable, Literal

import numpy as np

from imabo.imabo import IMABO
from imabo.memory import ArmStats, CurrentState, key_to_config
from imabo.oracle import Oracle
from imabo.oracles.parzen import (
    CategoricalDistanceFunc,
    default_gamma,
    default_weights,
    split_good_bad,
    tpe_suggest,
)
from imabo.policies.imoss import IMOSS
from imabo.types import ArmConfig, ArmKey


def lcb(mean_reward: float, n_pulls: int, t: float, c: float = 1.5) -> float:
    """A UCB1 width subtracted rather than added -- a pessimistic arm score.

    Only used by :class:`TPEOracle` with ``split_index="lcb"``, which ranks the good/bad
    split pessimistically instead of optimistically.
    """
    return mean_reward - c * math.sqrt(2.0 * math.log(max(2.0, t)) / max(1, n_pulls))


class TPEOracle(Oracle):
    """Propose from a density ratio fitted on the whole active set.

    Splits the arms at a score quantile into a good and a bad group, fits a Parzen density
    to each, draws candidates from the good density ``l`` and returns the one maximising
    ``l(x) / g(x)`` -- a proxy for expected improvement. No metric on the space is needed,
    and continuous, integer and categorical parameters are handled natively.

    The split ranks arms by the policy's index rather than by their empirical mean, so an
    arm that looks strong after two pulls is not yet treated as reliably good.

    Example:
        >>> optimizer = IMABO(space, IMOSS(beta=0.5), TPEOracle())

    Args:
        n_candidates: Candidates drawn from ``l`` per call, ranked by ``l/g``.
        gamma_func: Maps the number of rewarded arms to the size of the good group.
            Defaults to the top 30%.
        weights_func: Per-observation weights for the Parzen estimator.
        prior_weight: Weight of the uniform prior mixed into each density.
        multivariate: Fit one joint density over the whole space (default), rather than one
            independent density per parameter.
        split_index: What ranks the good/bad split. ``"policy"`` (default) uses the
            allocation policy's own index, as the paper does; ``"lcb"`` ignores it and
            ranks pessimistically by :func:`lcb` instead.
        categorical_distance_func: Per-parameter distance for categorical parameters.
            ``None`` derives it from the space, giving numeric categoricals an
            absolute-difference distance and leaving the rest one-hot.
    """

    def __init__(
        self,
        n_candidates: int = 24,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        prior_weight: float = 1.0,
        multivariate: bool = True,
        split_index: Literal["policy", "lcb"] = "policy",
        categorical_distance_func: dict[str, CategoricalDistanceFunc] | None = None,
    ):
        if split_index not in ("policy", "lcb"):
            raise ValueError(f"split_index must be 'policy' or 'lcb', got {split_index!r}")
        self.n_candidates = n_candidates
        self.gamma_func = gamma_func or default_gamma
        self.weights_func = weights_func or default_weights
        self.prior_weight = prior_weight
        self.multivariate = multivariate
        self.split_index = split_index
        self.categorical_distance_func = categorical_distance_func

    def suggest(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        score: Callable[[ArmKey], float],
    ) -> ArmConfig:
        if not rewarded_arms:
            return self.space.sample(self.rng)

        good, bad = split_good_bad(
            rewarded_arms, self._rank(state, score), self.gamma_func
        )
        candidate = tpe_suggest(
            good_configs=[key_to_config(k, self.space.names) for k, _ in good],
            bad_configs=[key_to_config(k, self.space.names) for k, _ in bad],
            param_names=self.space.names,
            distributions=self.space.distributions,
            n_candidates=self.n_candidates,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            multivariate=self.multivariate,
            weights_func=self.weights_func,
            categorical_distance_func=self.categorical_distance_func,
        )
        if candidate is None:
            return self.space.sample(self.rng)
        return candidate

    def _rank(
        self, state: CurrentState, score: Callable[[ArmKey], float]
    ) -> Callable[[ArmKey], float]:
        if self.split_index == "lcb":
            return lambda key: lcb(
                state.arms[key].mean_reward,
                state.arms[key].nb_rewarded,
                state.nb_steps,
            )
        return score


class IMOSSTPE(IMABO):
    """IMOSS paired with the TPE oracle."""

    def __init__(
        self,
        search_space,
        *,
        beta: float = 0.5,
        seed: int | None = None,
        **oracle: Any,
    ):
        super().__init__(search_space, IMOSS(beta=beta), TPEOracle(**oracle), seed=seed)
