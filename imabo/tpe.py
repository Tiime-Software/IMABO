"""TPE (Tree-structured Parzen Estimator) oracle for IMABO."""

import math
from typing import Callable

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    _ParzenEstimatorParameters,
)

from imabo.types import ArmConfig, OptunaConfigs


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


def create_search_space(
    search_space_specs: dict,
) -> tuple[dict[str, BaseDistribution], dict[str, str]]:
    """Create Optuna search space from domain"""
    distributions: dict[str, BaseDistribution] = {}
    param_types: dict[str, str] = {}

    for name, spec in search_space_specs.items():
        lower = spec.get("lower")
        upper = spec.get("upper")
        if lower is not None and upper is not None:
            if spec.get("int", False):
                distributions[name] = IntDistribution(
                    low=lower, high=upper, log=spec.get("log", False)
                )
                param_types[name] = "integer"
            else:
                distributions[name] = FloatDistribution(
                    low=lower, high=upper, log=spec.get("log", False)
                )
                param_types[name] = "continuous"
        elif spec.get("choices"):
            distributions[name] = CategoricalDistribution(choices=spec["choices"])
            param_types[name] = "categorical"
        else:
            raise ValueError(f"Invalid parameter specification for '{name}': {spec}")

    return distributions, param_types


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

    Returns:
        Best candidate configuration, or None if sampling fails.
    """
    good_obs = configs_to_optuna(good_configs, param_names, distributions)
    bad_obs = configs_to_optuna(bad_configs, param_names, distributions)

    parzen_params = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=True,
        consider_endpoints=True,
        weights=weights_func,
        multivariate=multivariate,
        categorical_distance_func={},
    )

    if multivariate:
        parzen_l = _ParzenEstimator(
            observations=good_obs,
            search_space=distributions,
            parameters=parzen_params,
        )
        parzen_g = _ParzenEstimator(
            observations=bad_obs,
            search_space=distributions,
            parameters=parzen_params,
        )

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
            parzen_l = _ParzenEstimator(
                observations={name: good_obs[name]},
                search_space=dist,
                parameters=parzen_params,
            )
            parzen_g = _ParzenEstimator(
                observations={name: bad_obs[name]},
                search_space=dist,
                parameters=parzen_params,
            )
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
