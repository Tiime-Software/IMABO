"""Mutation primitives shared by the two IMOSS explore oracles.

Both :class:`imabo.coord_ucb.IMABOCoordUCB` and
:class:`imabo.tabpfn_optimizer.IMABOTabPFN` propose new arms by perturbing an
incumbent, so the "which arm" and "which value" decisions live here rather than
being implemented twice.

Only the rules the two winning configurations use are kept. The alternatives
that were measured and rejected -- softmax / last-proposal / MOSS parent
selection, UCB value bandits, multi-coordinate mutation -- are on the
`oracles-archive` branch.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.types import ArmConfig, ArmKey

# A ``(name, current) -> value`` rule for the mutated coordinate.
ValueSampler = Callable[[str, Any], Any]


def best_config(
    rewarded_arms: list[tuple[ArmKey, Any]], param_names: list[str]
) -> ArmConfig:
    """The arm with the highest empirical mean -- the incumbent to perturb.

    Note this is NOT the arm the exploit phase is pulling: MOSS ranks by mean
    *plus* an exploration bonus, and the two argmaxes agreed only 58% of the time
    on the RF grid. Mutating the incumbent rather than the MOSS pick, or rather
    than a softmax draw over the population, was the single largest effect
    measured in either oracle (-26% for the surrogate-free one).
    """
    from imabo.memory import key_to_config

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
    :meth:`imabo.optimizer.IMABO.generate_random_config` would draw it
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
