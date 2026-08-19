from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.memory import ArmStats, CurrentState, key_to_config
from imabo.types import ArmConfig, ArmKey

# A ``(name, current) -> value`` rule for the mutated coordinate.
ValueSampler = Callable[[str, Any], Any]


def best_config(
    rewarded_arms: list[tuple[ArmKey, Any]], param_names: list[str]
) -> ArmConfig:
    """The arm with the highest empirical mean -- the incumbent to perturb."""
    key = max(rewarded_arms, key=lambda kv: kv[1].mean_reward)[0]
    return key_to_config(key, param_names)


def mutate_value(
    name: str,
    current: Any,
    distributions: dict[str, BaseDistribution],
    rng: random.Random,
    value_sampler: ValueSampler | None = None,
) -> Any:
    """A new value for parameter ``name``, different from ``current``.

    Drawn uniformly from the parameter's own domain, exactly as
    :meth:`imabo.search_space.SearchSpace.sample_value` would draw it
    (log-uniform for log params), except that on a finite discrete domain
    (categorical, integer) ``current`` is excluded -- the remaining values stay
    equiprobable. Excluding it matters because a mutant equal to its parent is
    not a new arm: an explore step would spend its pull re-pulling a known arm
    instead of opening a new one.

    ``value_sampler`` replaces the uniform draw with any ``(name, current) ->
    value`` rule (see :func:`local_value_sampler`), which is then responsible for
    avoiding ``current`` itself.
    """
    if value_sampler is not None:
        return value_sampler(name, current)

    dist = distributions[name]
    if isinstance(dist, CategoricalDistribution):
        choices = [c for c in dist.choices if c != current]
        return rng.choice(choices) if choices else current
    if isinstance(dist, IntDistribution):
        if dist.high <= dist.low:
            return current
        # Draw over a range one short of the domain, then shift past the current
        # value: uniform over {low..high} \ {current}.
        value = rng.randint(dist.low, dist.high - 1)
        return value + 1 if value >= current else value
    if isinstance(dist, FloatDistribution):
        if dist.log:
            return math.exp(rng.uniform(math.log(dist.low), math.log(dist.high)))
        return rng.uniform(dist.low, dist.high)
    raise TypeError(f"unsupported distribution for mutation: {type(dist)!r}")


def local_value_sampler(
    distributions: dict[str, BaseDistribution],
    rng: random.Random,
    scale: float = 0.1,
) -> ValueSampler:
    """A ``(name, current) -> value`` sampler that takes a LOCAL step.

    :func:`mutate_value` resamples a coordinate from its whole domain, which on a
    finite axis is a genuine neighbour (5-10 levels) but on a continuous one is
    not a mutation at all: measured on the 2-D HPO boxes, a uniform redraw lands
    a mean 0.25 of the axis' full log-range from the parent -- 1.25 decades on a
    5-decade axis -- and only 19% of draws fall within 10% of it. A pool built
    that way gives a surrogate nothing near the incumbent to exploit.

    This instead perturbs the current value by a Gaussian step of ``scale`` times
    the axis' width, in log space for log-scaled parameters, clipped to the
    domain. At ``scale=0.1`` the mean distance falls to 0.079 of the range and
    69% of mutants land within 10% of the parent. Categorical axes have no
    metric, so they fall back to :func:`mutate_value`'s uniform-excluding-current
    draw -- which makes this argument a no-op on a purely categorical space, and
    bit-identical to leaving it unset. Integer axes round and are forced off
    ``current`` so the mutant is still a new configuration.
    """

    def sample(name: str, current: Any) -> Any:
        dist = distributions[name]
        if isinstance(dist, CategoricalDistribution):
            return mutate_value(name, current, distributions, rng)
        if isinstance(dist, IntDistribution):
            if dist.high <= dist.low:
                return current
            step = rng.gauss(0.0, scale * (dist.high - dist.low))
            value = int(round(current + step))
            if value == current:
                value = current + (1 if rng.random() < 0.5 else -1)
            return min(max(value, dist.low), dist.high)
        if isinstance(dist, FloatDistribution):
            if dist.log:
                lo, hi = math.log(dist.low), math.log(dist.high)
                value = math.log(current) + rng.gauss(0.0, scale * (hi - lo))
                # Clamp in the ORIGINAL space, not just in log space: exp(log(low))
                # round-trips to a float a hair below `low` (1e-5 -> 9.9999...e-06),
                # and a clipped step lands on the boundary often, where a uniform
                # draw essentially never does. Downstream validators reject it.
                value = math.exp(min(max(value, lo), hi))
            else:
                value = current + rng.gauss(0.0, scale * (dist.high - dist.low))
            return min(max(value, dist.low), dist.high)
        raise TypeError(f"unsupported distribution for mutation: {type(dist)!r}")

    return sample


class CandidatePool:
    """The pool of configurations a reward model ranks.

    ``"uniform"`` draws the whole pool uniformly -- the model then ranks a global
    random pool, which in a large space is the binding constraint: none of its
    members is near a good arm to begin with. ``"mutation"`` keeps a
    ``uniform_frac`` share uniform (so the pool never loses access to unexplored
    regions) and makes the rest single-coordinate mutants of the incumbent.
    """

    def __init__(
        self,
        n: int = 100,
        source: str = "mutation",
        uniform_frac: float = 0.1,
        scale: float | None = 0.1,
        filter_open: bool = True,
    ):
        if source not in ("uniform", "mutation"):
            raise ValueError(f"source must be 'uniform' or 'mutation', got {source!r}")
        if not 0.0 <= uniform_frac <= 1.0:
            raise ValueError(f"uniform_frac must be in [0, 1], got {uniform_frac}")
        self.n = n
        self.source = source
        self.uniform_frac = uniform_frac
        self.scale = scale
        self.filter_open = filter_open

    def build(
        self,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        space,
        rng: random.Random,
    ) -> list[ArmConfig]:
        """Build the ``n`` candidates the model will rank.

        Falls back to the uniform pool while no arm has a reward.
        """
        if self.source == "uniform" or not rewarded_arms:
            return [space.sample(rng) for _ in range(self.n)]

        n_uniform = min(self.n, int(round(self.uniform_frac * self.n)))
        candidates = [space.sample(rng) for _ in range(n_uniform)]
        parent = best_config(rewarded_arms, space.names)
        value_sampler = (
            local_value_sampler(space.distributions, rng, self.scale)
            if self.scale is not None
            else None
        )
        for _ in range(self.n - n_uniform):
            candidates.append(self._mutate(parent, space, rng, value_sampler))
        return candidates

    def drop_duplicates_and_open(
        self,
        candidates: list[ArmConfig],
        state: CurrentState,
        space,
        rng: random.Random,
    ) -> list[ArmConfig]:
        """Deduplicate the pool and drop candidates that are already open arms.

        Deduplicate before scoring: a mutation pool draws each candidate
        independently, so on a finite space collisions are common. Scoring
        duplicates wastes rows and fills the shortlist with repeats of one config
        instead of distinct runners-up.

        Then drop candidates that are already open arms. The pool exists so the
        model can pick the next arm to OPEN; a candidate already in memory cannot
        do that. This is not rescuing an exhausted pool -- the pool almost always
        holds novel candidates -- it OVERRIDES the acquisition, which on a finite
        grid reliably ranks an already-open neighbour of the incumbent above any
        unopened candidate, stalling ``|arms|`` below the ``t**beta`` target.
        """
        deduped: dict[ArmKey, ArmConfig] = {}
        for candidate in candidates:
            deduped.setdefault(space.encode(candidate), candidate)

        if not self.filter_open:
            return list(deduped.values())

        novel = [c for key, c in deduped.items() if key not in state.arms]
        if novel:
            return novel

        redrawn: dict[ArmKey, ArmConfig] = {}
        for _ in range(self.n):
            candidate = space.sample(rng)
            redrawn.setdefault(space.encode(candidate), candidate)
        return [c for key, c in redrawn.items() if key not in state.arms] or list(
            deduped.values()
        )

    def _mutate(
        self,
        config: ArmConfig,
        space,
        rng: random.Random,
        value_sampler: ValueSampler | None,
    ) -> ArmConfig:
        """Resample one uniformly-chosen parameter of ``config``.

        Every other coordinate is inherited from the parent arm; the new value
        comes from :func:`mutate_value` -- a uniform draw over that parameter's
        domain excluding the parent's own value, or ``value_sampler``'s local step
        when ``scale`` is set.
        """
        name = rng.choice(space.names)
        return {
            **config,
            name: mutate_value(name, config[name], space.distributions, rng, value_sampler),
        }


def to_frame(space, configs: list[ArmConfig]) -> Any:
    """Build a DataFrame from configs, tagging categorical columns.

    Columns follow ``space.names`` order and categorical params are cast to
    pandas ``category`` dtype so the model treats them as categorical features.
    """
    import pandas as pd

    df = pd.DataFrame(configs, columns=space.names)
    for name in space.names:
        if space.is_categorical(name):
            df[name] = df[name].astype("category")
    return df


def categorical_indices(space) -> list[int]:
    """0-based indices of categorical columns, in ``space.names`` order."""
    return [i for i, name in enumerate(space.names) if space.is_categorical(name)]


def rank(scores: np.ndarray, keep: int) -> list[int]:
    """Indices of the ``keep`` highest-scoring candidates, best first.

    ``argsort(...)[::-1]`` rather than ``argsort(-scores)``: the two break ties in
    opposite orders, and a reward model routinely predicts the same value for
    several candidates on a discrete grid.
    """
    return list(np.argsort(scores)[::-1][:keep])
