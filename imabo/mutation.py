"""Population-mutation proposal helpers.

The pieces a "propose a new arm by perturbing a good existing one" oracle needs,
factored out of any one optimizer so the variants share exactly the same
mechanics: pick a *parent* arm from the rewarded population
(:func:`parent_probabilities`), pick a *coordinate*, and pick a new *value* for
it (:func:`mutate_value`, optionally driven by
:func:`tpe_value_sampler`).

Used by :class:`imabo.tabpfn_optimizer.IMABOTabPFN`'s mutation candidate pools
(where a surrogate then ranks many such mutants) and by
:class:`imabo.coord_ucb.IMABOCoordUCB` (where a single mutant is proposed
directly, with the coordinate chosen by a bandit).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Literal

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.tpe import default_weights, univariate_tpe_sampler
from imabo.types import ArmConfig

ValueSampler = Callable[[str, Any], Any]

#: Which configuration a mutation oracle perturbs. ``"best"`` is the incumbent
#: hill-climb, ``"softmax"`` keeps the whole population alive, and the other two
#: are the local walks it is worth distinguishing from "best" -- see
#: :func:`parent_config`.
ParentRule = Literal["best", "softmax", "last_proposal", "moss"]


def parent_config(
    optimizer: Any,
    rule: ParentRule,
    rewarded_arms: list[tuple[Any, Any]],
    state: Any,
    nb_pending_total: int = 0,
    nb_rewarded_total: int = 0,
    temperature: float = 1.0,
    last_proposal: ArmConfig | None = None,
) -> ArmConfig:
    """The configuration a mutation oracle should perturb, per ``rule``.

    * ``"best"``: the arm with the highest empirical mean -- an incumbent
      hill-climb. Note this is NOT the same as the arm the bandit is pulling:
      MOSS ranks by mean *plus* an exploration bonus, and the two argmaxes agreed
      only 58% of the time on the RF grid.
    * ``"softmax"``: sampled from the whole rewarded population with probability
      ``softmax(mean_reward / T)``, ``T = temperature * std(mean_rewards)``
      (:func:`parent_probabilities`).
    * ``"last_proposal"``: the configuration this oracle proposed last, whether or
      not it turned out well -- a random walk rather than a hill-climb, since a
      single-coordinate mutation of a good arm is usually worse than it. Falls back
      to ``"best"`` on the first call.
    * ``"moss"``: the arm the exploit phase would pull right now, i.e. the argmax
      of the MOSS-anytime index (:meth:`imabo.optimizer.IMABO.suggest_existing`).

    Takes the ``optimizer`` because three of the four rules need its state --
    ``rng``, ``param_names``, the MOSS index -- and both mutation oracles
    (:class:`imabo.tabpfn_optimizer.IMABOTabPFN` and
    :class:`imabo.coord_ucb.IMABOCoordUCB`) must implement them identically.
    """
    from imabo.memory import key_to_config

    if rule == "softmax":
        scores = np.array([s.mean_reward for _, s in rewarded_arms], dtype=float)
        probs = parent_probabilities(scores, temperature)
        index = optimizer.rng.choices(
            range(len(rewarded_arms)), weights=probs.tolist(), k=1
        )[0]
        return key_to_config(rewarded_arms[index][0], optimizer.param_names)

    if rule == "last_proposal" and last_proposal is not None:
        return dict(last_proposal)

    if rule == "moss":
        return optimizer.suggest_existing(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )

    best_key, _ = max(rewarded_arms, key=lambda t: t[1].mean_reward)
    return key_to_config(best_key, optimizer.param_names)


def coordinate_importance(
    rewarded_arms: list[tuple[Any, Any]],
    param_names: list[str],
    min_rewards: int = 1,
) -> dict[str, float]:
    """Per-coordinate effect size, estimated from the arm population.

    For each coordinate, the mean ``|difference in mean reward|`` over all pairs of
    observed arms that differ in *exactly* that coordinate -- the empirical
    counterpart of "move this axis alone and see how much the reward moves", which
    is what a single-coordinate mutation experiences. Arms with fewer than
    ``min_rewards`` observations are dropped, since a one-pull mean carries the
    full observation noise into the difference.

    Two reasons this is a better importance readout than the coordinate bandit's
    own statistics:

    * **Magnitude, not sign.** A bandit credited with ``mean(child) -
      mean(parent)`` learns *expected gain*, and mutating an already-good arm on
      its MOST important axis is what hurts the most -- so that statistic ranks the
      important axes LAST. Correct for control (avoid breaking what matters),
      backwards for importance.
    * **Off-policy.** It uses every pair in the population, not only the
      parent-child pairs the oracle happened to create, which is one or two orders
      of magnitude more evidence at the same budget.

    Cost is O(K*d) to group plus the pairs inside each group, so this is a
    diagnostic to call when the readout is wanted, not per explore step.
    """
    from imabo.memory import key_to_config

    arms = [
        (key, stats)
        for key, stats in rewarded_arms
        if stats.nb_rewarded >= min_rewards
    ]
    importance: dict[str, float] = {}
    for i, name in enumerate(param_names):
        # Group by every coordinate EXCEPT i: within a group, arms differ only in i.
        groups: dict[tuple, list[float]] = {}
        for key, stats in arms:
            rest = tuple(v for j, v in enumerate(key) if j != i)
            groups.setdefault(rest, []).append(float(stats.mean_reward))
        deltas = []
        for values in groups.values():
            for a in range(len(values)):
                for b in range(a + 1, len(values)):
                    deltas.append(abs(values[a] - values[b]))
        importance[name] = float(np.mean(deltas)) if deltas else 0.0
    del key_to_config  # (kept out of the hot path; grouping works on keys directly)
    return importance


def axis_values(dist, n_points: int) -> list[Any]:
    """Enumerate the finite value set of one axis, for value-bandit oracles.

    A bandit over the values of a parameter needs an explicit finite arm set, so
    it "cannot admit a value outside the declared list" -- a genuine limitation
    against an oracle that proposes into a continuous space. Shared by Hier-MAB's
    low level (:class:`experiments.baselines.hier_mab.HierMAB`) and by
    :class:`imabo.coord_ucb.IMABOCoordUCB`'s ``value_rule="ucb"``, so both draw
    values from exactly the same set.

    Categorical axes (what ``RFTabularFiniteBenchmark`` hands us) are used
    verbatim, so on the RF tabular grid the arm sets are exactly the benchmark's
    own discretisation and nothing is invented. Continuous and integer axes are
    discretised on an evenly spaced (or log-spaced) grid so these oracles can
    also run on the LR/SVM and HotpotQA spaces.
    """
    if isinstance(dist, CategoricalDistribution):
        return list(dist.choices)
    if isinstance(dist, IntDistribution):
        lo, hi = int(dist.low), int(dist.high)
        if hi - lo + 1 <= n_points:
            return list(range(lo, hi + 1))
        if dist.log:
            raw = np.exp(np.linspace(math.log(max(lo, 1)), math.log(hi), n_points))
        else:
            raw = np.linspace(lo, hi, n_points)
        return sorted({int(round(v)) for v in raw})
    if isinstance(dist, FloatDistribution):
        # Plain Python floats, not np.float64: suggested configs get written
        # to JSON checkpoints by the callers (e.g. hotpotqa_experiment).
        if dist.log:
            # np.geomspace, not exp(linspace(log lo, log hi)): geomspace pins
            # both endpoints exactly, while the exp/log roundtrip lands a hair
            # outside the bounds (e.g. 9.9999...e-06 for lo=1e-05), which
            # strict validators like ConfigSpace reject.
            return [float(v) for v in np.geomspace(dist.low, dist.high, n_points)]
        return [float(v) for v in np.linspace(dist.low, dist.high, n_points)]
    raise TypeError(f"unsupported distribution for an axis grid: {type(dist)!r}")


def parent_probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over arm scores at a temperature scaled to their dispersion.

    The temperature is ``temperature * std(scores)``, which makes the softmax
    logits z-scores: selection pressure is governed by how many standard
    deviations apart the arms are, never by the reward scale (0-1 accuracies and
    0-100 scores behave identically), and it stays comparable over a run as the
    population's spread shrinks or grows -- a fixed temperature would instead
    drift from near-uniform early (rewards still spread out) to near-greedy late
    (all surviving arms close together).

    ``temperature -> 0`` is greedy (all weight on the best arm), ``1.0`` gives one
    e-fold of weight per standard deviation of reward, and large values flatten to
    uniform. A degenerate population (a single arm, or every arm at the same
    score, so ``std == 0``) gets the uniform distribution, which is what the
    softmax converges to there anyway.
    """
    n = scores.size
    sigma = float(np.std(scores))
    if n == 0 or sigma <= 0.0 or not np.isfinite(sigma):
        return np.full(n, 1.0 / n) if n else scores
    # Subtracting the max keeps exp() from overflowing; it cancels in the
    # normalization, so the distribution is unchanged.
    weights = np.exp((scores - scores.max()) / (temperature * sigma))
    return weights / weights.sum()


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
    value`` rule (see :func:`tpe_value_sampler`), which is then responsible for
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
    domain. Categorical axes have no metric, so they fall back to
    :func:`mutate_value`'s uniform-excluding-current draw. Integer axes round and
    are forced off ``current`` so the mutant is still a new configuration.
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


def tpe_value_sampler(
    good_configs: list[ArmConfig],
    bad_configs: list[ArmConfig],
    distributions: dict[str, BaseDistribution],
    n_candidates: int,
    rng: np.random.RandomState,
    prior_weight: float = 1.0,
    weights_func: Callable = default_weights,
    pick: Literal["ei_argmax", "sample"] = "ei_argmax",
) -> ValueSampler:
    """Build a ``(name, current) -> value`` sampler backed by a univariate TPE.

    Per call, ``n_candidates`` FRESH values are drawn from the 1-D ``l`` density of
    the coordinate being mutated, and ``pick`` decides which one is returned:

    * ``"ei_argmax"`` (default): the value maximizing ``log l - log g``, i.e. the
      classical univariate-TPE proposal restricted to one parameter. Note this is
      DETERMINISTIC given the fitted densities -- every draw contains the mode and
      the mode maximizes the ratio -- so a whole mutation pool built this way gets
      the same value on each coordinate, collapsing 90 mutants to ``d`` distinct
      configurations (measured: 4 of 90 on the RF grid). Correct for a single
      proposal per step, useless for filling a pool.
    * ``"sample"``: one of the draws chosen uniformly, i.e. a sample from ``l``
      rather than its EI-argmax. Trades proposal quality per candidate for pool
      diversity, which is what a surrogate ranking the pool actually needs.

    Either way ``current`` is skipped when some other draw differs from it, so the
    mutation yields a different config; if every draw equals it, one is returned
    unchanged. The *fitted Parzen pair* is cached per coordinate
    (:func:`imabo.tpe.univariate_tpe_sampler`), so fitting is paid once per
    parameter however many values are drawn.
    """
    fitted: dict[str, Callable[[int, np.random.RandomState], list]] = {}

    def sample(name: str, current: Any) -> Any:
        if name not in fitted:
            fitted[name] = univariate_tpe_sampler(
                good_configs=good_configs,
                bad_configs=bad_configs,
                name=name,
                distribution=distributions[name],
                prior_weight=prior_weight,
                weights_func=weights_func,
            )
        values = fitted[name](n_candidates, rng)
        order = (
            range(len(values))
            if pick == "ei_argmax"
            else rng.permutation(len(values))
        )
        for i in order:
            if values[i] != current:
                return values[i]
        return values[0]

    return sample


