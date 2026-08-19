import math
from typing import Any, Callable

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
)

try:
    from optuna.samplers._tpe.parzen_estimator import (
        _ParzenEstimator,
        _ParzenEstimatorParameters,
    )
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "imabo's TPE oracles use Optuna's Parzen estimator, which lives in the "
        "private module optuna.samplers._tpe.parzen_estimator and has moved in this "
        "Optuna version. Pin a tested release (see pyproject.toml) or open an issue."
    ) from error

from imabo.memory import ArmStats
from imabo.types import ArmConfig, ArmKey, OptunaConfigs

CategoricalDistanceFunc = Callable[[Any, Any], float]


def default_gamma(x: int) -> int:
    """Default gamma function: top 30% of arms are considered 'good'. Similar to Optuna."""
    return int(math.ceil(0.3 * x))


def hyperopt_default_gamma(x: int) -> int:
    """Hyperopt-style gamma function (alternative to :func:`default_gamma`)."""
    return min(int(math.ceil(0.25 * math.sqrt(x))), 25)


def default_weights(x: int) -> np.ndarray:
    """Default weight function for the Parzen estimator. Similar to Optuna."""
    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


def numeric_l1_distance(a: float, b: float) -> float:
    """Absolute difference -- the natural distance for a categorical parameter
    whose choices are actually ordered numbers (e.g. a discretized max_depth
    grid represented as a ``CategoricalDistribution``, see
    ``RFTabularFiniteBenchmark.get_search_space``). Suggested default distance
    function for numeric-valued categorical arms; see
    :func:`adaptive_categorical_distance_func`.
    """
    return abs(a - b)


def adaptive_categorical_distance_func(
    distributions: dict[str, BaseDistribution],
) -> dict[str, CategoricalDistanceFunc]:
    """Auto-derive a ``categorical_distance_func`` dict from the search space.

    Optuna's Parzen estimator treats every categorical parameter as unordered
    by default -- a hard one-hot kernel that puts weight only on the exact
    observed choice (see ``_calculate_categorical_distributions`` in
    ``optuna.samplers._tpe.parzen_estimator``). That is the wrong prior
    whenever the "choices" are actually ordered numbers, which is common in
    this project: finite-arm benchmarks coarsen a numeric hyperparameter grid
    into ``CategoricalDistribution`` choices purely because the arm set is
    finite and precomputed (see ``RFTabularFiniteBenchmark.get_search_space``,
    whose ``max_depth``/``max_features``/``min_samples_leaf``/
    ``min_samples_split`` are all numeric "choices"). For those, a choice close
    to a good/bad observation should borrow some of its density instead of
    being treated as unrelated as one at the opposite end of the range.

    Picks up every categorical parameter whose choices are all real numbers
    (``bool`` excluded -- True/False have no ordering) and assigns it
    :func:`numeric_l1_distance`; string/mixed-type categoricals are left out of
    the dict, which keeps Optuna's one-hot default for them.
    """
    return {
        name: numeric_l1_distance
        for name, dist in distributions.items()
        if isinstance(dist, CategoricalDistribution)
        and dist.choices
        and all(
            isinstance(c, (int, float)) and not isinstance(c, bool)
            for c in dist.choices
        )
    }


def configs_to_optuna(
    configs: list[ArmConfig],
    param_names: list[str],
    distributions: dict[str, BaseDistribution],
) -> OptunaConfigs:
    """Convert list of configurations to Optuna configs format"""
    if not configs:
        return {name: np.array([]) for name in param_names}

    result: OptunaConfigs = {}
    for name in param_names:
        values = [distributions[name].to_internal_repr(c[name]) for c in configs]
        result[name] = np.array(values)
    return result


def optuna_to_configs(
    optuna_configs: OptunaConfigs,
    param_names: list[str],
    distributions: dict[str, BaseDistribution],
) -> list[ArmConfig]:
    """Convert Optuna configs format back to list of arm configs"""
    if not optuna_configs or len(optuna_configs[param_names[0]]) == 0:
        return []

    n_samples = len(optuna_configs[param_names[0]])
    configs: list[ArmConfig] = []
    for i in range(n_samples):
        config: ArmConfig = {}
        for name in param_names:
            config[name] = distributions[name].to_external_repr(
                optuna_configs[name][i]
            )
        configs.append(config)
    return configs



def fit_parzen(
    observations: OptunaConfigs,
    distributions: dict[str, BaseDistribution],
    prior_weight: float,
    weights_func: Callable,
    multivariate: bool,
    categorical_distance_func: dict[str, CategoricalDistanceFunc] | None,
) -> _ParzenEstimator:
    """Fit one Parzen density over ``observations``.

    The single place imabo touches Optuna's private ``_ParzenEstimator``. Keeping the
    three call sites behind one function means a future Optuna release that moves or
    changes it is a one-function fix rather than a hunt; ``tests/test_optuna.py``
    pins both the symbol and its numerical output so such a change fails in CI
    rather than in a user's run.
    """
    return _ParzenEstimator(
        observations=observations,
        search_space=distributions,
        parameters=_ParzenEstimatorParameters(
            prior_weight=prior_weight,
            consider_magic_clip=True,
            consider_endpoints=True,
            weights=weights_func,
            multivariate=multivariate,
            categorical_distance_func=categorical_distance_func,
        ),
    )


def tpe_suggest(
    good_configs: list[ArmConfig],
    bad_configs: list[ArmConfig],
    param_names: list[str],
    distributions: dict[str, BaseDistribution],
    n_candidates: int,
    rng: np.random.RandomState,
    prior_weight: float = 1.0,
    multivariate: bool = True,
    weights_func: Callable = default_weights,
    categorical_distance_func: dict[str, CategoricalDistanceFunc] | None = None,
) -> ArmConfig | None:
    """Suggest a new configuration using the TPE oracle.

    Fits Parzen estimators on 'good' and 'bad' sets, samples candidates
    from the good estimator, and returns the one maximizing l(x)/g(x).

    Args:
        good_configs: Configurations in the 'good' (top quantile) set.
        bad_configs: Configurations in the 'bad' set.
        param_names: Sorted list of parameter names.
        distributions: Optuna distribution objects per parameter.
        n_candidates: Number of EI candidates to sample.
        rng: Numpy RandomState for reproducibility.
        prior_weight: Prior weight for the Parzen estimator.
        multivariate: True (default) fits ONE joint Parzen mixture over the
            whole space -- each component is a past observation, so sampling
            preserves co-occurrence between parameters (including categorical
            ones). False fits an independent 1-D estimator per parameter and
            samples each coordinate on its own -- classical univariate TPE, a
            fully factored proposal. (Passing the flag into Optuna's
            _ParzenEstimator, as an earlier version did, only switches the
            numerical bandwidth heuristic and does NOT factor the mixture --
            on an all-categorical space it changed nothing at all.)
        weights_func: Function to compute observation weights.
        categorical_distance_func: Per-parameter distance function fed to the
            Parzen estimator for categorical parameters (see
            :func:`adaptive_categorical_distance_func`). ``None`` (default)
            derives it automatically from ``distributions`` -- an
            absolute-difference distance for every categorical whose choices
            are all numeric, Optuna's plain one-hot treatment for the rest.
            Pass ``{}`` to force one-hot for every categorical parameter, or a
            custom dict to override specific parameters.

    Returns:
        Best candidate configuration, or None if sampling fails.
    """
    good_obs = configs_to_optuna(good_configs, param_names, distributions)
    bad_obs = configs_to_optuna(bad_configs, param_names, distributions)

    if categorical_distance_func is None:
        categorical_distance_func = adaptive_categorical_distance_func(distributions)

    def fit(observations: OptunaConfigs, space) -> _ParzenEstimator:
        return fit_parzen(
            observations,
            space,
            prior_weight,
            weights_func,
            multivariate,
            categorical_distance_func,
        )

    if multivariate:
        parzen_l = fit(good_obs, distributions)
        parzen_g = fit(bad_obs, distributions)

        candidates_dict = parzen_l.sample(rng, n_candidates)
        candidates = optuna_to_configs(candidates_dict, param_names, distributions)

        if not candidates:
            return None

        candidates_obs = configs_to_optuna(candidates, param_names, distributions)
        log_l = parzen_l.log_pdf(candidates_obs)
        log_g = parzen_g.log_pdf(candidates_obs)
    else:
        # Univariate TPE: an independent 1-D estimator pair per parameter,
        # each coordinate sampled from its own l-density, EI scored as the
        # sum of per-dimension log ratios (the log of a product density).
        candidates_dict: OptunaConfigs = {}
        log_l = np.zeros(n_candidates)
        log_g = np.zeros(n_candidates)
        for name in param_names:
            dist = {name: distributions[name]}
            parzen_l = fit({name: good_obs[name]}, dist)
            parzen_g = fit({name: bad_obs[name]}, dist)
            samples = parzen_l.sample(rng, n_candidates)
            candidates_dict[name] = samples[name]
            log_l += parzen_l.log_pdf(samples)
            log_g += parzen_g.log_pdf(samples)

        candidates = optuna_to_configs(candidates_dict, param_names, distributions)
        if not candidates:
            return None

    ei_scores = log_l - log_g
    best_idx = int(np.argmax(ei_scores))
    return candidates[best_idx]


def _as_python(value):
    """Numpy scalar -> plain Python scalar, other values untouched.

    Optuna's ``to_external_repr`` hands back ``np.float64`` for float
    parameters. Proposed configs end up in JSON checkpoints written by the
    experiment scripts, and ``json.dump`` cannot serialize numpy scalars, so the
    proposal helpers below hand out plain Python numbers.
    """
    return value.item() if isinstance(value, np.generic) else value


def univariate_tpe_sampler(
    good_configs: list[ArmConfig],
    bad_configs: list[ArmConfig],
    name: str,
    distribution: BaseDistribution,
    prior_weight: float = 1.0,
    weights_func: Callable = default_weights,
    categorical_distance_func: dict[str, CategoricalDistanceFunc] | None = None,
) -> Callable[[int, np.random.RandomState], list]:
    """Fit the 1-D Parzen pair for ``name`` once; return a re-usable sampler.

    The returned callable takes ``(n_candidates, rng)`` and gives that many FRESH
    draws from ``l``, EI-ranked -- so two calls return different candidate values
    from the same fitted densities. That split matters for a caller that needs
    many independent values from one coordinate (a mutation pool): the fit is the
    expensive part and is paid once, while reusing a single ranked *list* would
    hand every candidate the same top-ranked value.

    ``categorical_distance_func``: see :func:`tpe_suggest`. ``None`` (default)
    derives it for ``name`` alone via :func:`adaptive_categorical_distance_func`.
    """
    search_space = {name: distribution}
    if categorical_distance_func is None:
        categorical_distance_func = adaptive_categorical_distance_func(search_space)

    parzen_l = fit_parzen(
        configs_to_optuna(good_configs, [name], search_space),
        search_space,
        prior_weight,
        weights_func,
        multivariate=False,
        categorical_distance_func=categorical_distance_func,
    )
    parzen_g = fit_parzen(
        configs_to_optuna(bad_configs, [name], search_space),
        search_space,
        prior_weight,
        weights_func,
        multivariate=False,
        categorical_distance_func=categorical_distance_func,
    )

    def draw(n_candidates: int, rng: np.random.RandomState) -> list:
        samples = parzen_l.sample(rng, n_candidates)
        ei = parzen_l.log_pdf(samples) - parzen_g.log_pdf(samples)
        order = np.argsort(ei)[::-1]
        return [
            _as_python(distribution.to_external_repr(samples[name][i])) for i in order
        ]

    return draw


def univariate_tpe_values(
    good_configs: list[ArmConfig],
    bad_configs: list[ArmConfig],
    name: str,
    distribution: BaseDistribution,
    n_candidates: int,
    rng: np.random.RandomState,
    prior_weight: float = 1.0,
    weights_func: Callable = default_weights,
    categorical_distance_func: dict[str, CategoricalDistanceFunc] | None = None,
) -> list:
    """EI-ranked candidate values for ONE parameter (classical univariate TPE).

    The single-coordinate counterpart of :func:`tpe_suggest`: fit the 1-D Parzen
    pair ``l``/``g`` for ``name`` alone over the good/bad arms, sample
    ``n_candidates`` values from ``l``, and return them in the parameter's
    external representation, sorted by descending ``log l(x) - log g(x)``. The
    caller picks -- ``[0]`` reproduces what univariate ``tpe_suggest`` would
    choose for this coordinate, while the rest of the ranking lets a caller skip
    a value it must not return (e.g. a mutation operator rejecting the parent's
    current value).

    ``categorical_distance_func``: see :func:`tpe_suggest`.
    """
    return univariate_tpe_sampler(
        good_configs,
        bad_configs,
        name,
        distribution,
        prior_weight,
        weights_func,
        categorical_distance_func,
    )(n_candidates, rng)


def split_good_bad(
    rewarded_arms: list[tuple[ArmKey, ArmStats]],
    score: Callable[[ArmKey], float],
    gamma_func: Callable[[int], int],
) -> tuple[list[tuple[ArmKey, ArmStats]], list[tuple[ArmKey, ArmStats]]]:
    """Split rewarded arms into 'good' and 'bad' sets, ranked by ``score``.

    A free function rather than a method: both the global TPE oracle and the
    coordinate one fit their densities on this same split, and neither is a
    special case of the other.
    """
    sorted_arms = sorted(rewarded_arms, key=lambda arm: score(arm[0]), reverse=True)
    n_good = max(1, min(gamma_func(len(sorted_arms)), len(sorted_arms) - 1))
    return sorted_arms[:n_good], sorted_arms[n_good:]
