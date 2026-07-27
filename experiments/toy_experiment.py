"""
Toy-function experiment: compare IMABO against StoSOO, HOO-T, and Stroquool.

Usage (from repo root):
    python -m experiments.toy_experiment
"""

from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from experiments.utils.stats import (
    calculate_statistics,
    save_iterations_to_csv,
    save_results_to_csv,
)
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5


class Algorithm(Enum):
    IMABO = "IMABO"
    STOSOO = "StoSOO"
    HOO_T = "HOO-T"
    STROQUOOL = "Stroquool"


class RegretData(TypedDict):
    regrets: list[float]
    simple_regrets: float


def _rescale_to_search_space(z: np.ndarray, search_space: dict) -> np.ndarray:
    """Map a point in [0,1]^dim (as produced by the tree baselines) onto the
    actual per-dimension [lower, upper] bounds declared in ``search_space``."""
    bounds = np.array([[v["lower"], v["upper"]] for v in search_space.values()])
    lower, upper = bounds[:, 0], bounds[:, 1]
    return lower + (upper - lower) * np.asarray(z)


def run_optimization(
    function_name: str,
    dim: int,
    beta: float = 0.5,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, RegretData]:
    obj_func = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj_func.get_function_by_name(function_name)
    func_noiseless = obj_func.get_function_by_name(function_name, noise=False)
    fmax = obj_func.get_theoretical_max(function_name)
    search_space = obj_func.get_search_space(function_name)

    if func is None:
        raise ValueError(f"Unknown function: {function_name}")

    imabo_opt = IMABO(
        search_space=search_space,
        seed=seed,
        multivariate=False,
        beta=beta,
    )
    stosoo_opt = TimedOptimizer(stosoo, n_iterations, dim)
    hoo_t_opt = TimedOptimizer(hoo_t, n_iterations, dim, rho=0.4, nu1=10.0)
    stroquool_opt = TimedOptimizer(stroquool, n_iterations, dim)

    # (enum, optimizer, is_xarm_style)
    opt_config = [
        (Algorithm.IMABO, imabo_opt, False),
        (Algorithm.STOSOO, stosoo_opt, True),
        (Algorithm.HOO_T, hoo_t_opt, True),
        (Algorithm.STROQUOOL, stroquool_opt, True),
    ]

    regrets: dict[str, RegretData] = {
        algo.value: {"regrets": [], "simple_regrets": float("inf")}
        for algo in Algorithm
    }

    for algo, opt, is_xarm in opt_config:
        opt_name = algo.value
        for _ in tqdm(range(n_iterations), desc=f"Running {opt_name}", leave=False):
            if is_xarm and opt.done:
                continue
            z = opt.suggest()
            x = _rescale_to_search_space(z, search_space) if is_xarm else z
            noiseless = func_noiseless(x) / dim
            y = noiseless + obj_func.noise_rng.normal(0, obj_func.noise_std)
            regret = fmax - noiseless
            if is_xarm:
                opt.observe(z, y)
            else:
                opt.observe(y)
            regrets[opt_name]["regrets"].append(regret)

        if is_xarm:
            best_x = _rescale_to_search_space(opt.suggest_best(), search_space)
        else:
            best_x = opt.best_x
        regrets[opt_name]["simple_regrets"] = fmax - func_noiseless(best_x) / dim

    return regrets


def run_multiple_experiments(
    function_name: str,
    dim: int,
    n_iterations: int = 1000,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = 8,
    beta: float = 0.5,
) -> list[dict[str, RegretData]]:
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_optimization)(
            function_name,
            dim,
            beta,
            n_iterations,
            base_seed + i * 1000,
        )
        for i in range(n_runs)
    )


if __name__ == "__main__":
    function_name = ["sin1", "garland", "rastrigin"]
    dim = 4
    n_runs = 20
    base_seed = 42
    for fn in function_name:
        print(f"Running {fn}...")
        test_cases = [
            (fn, dim, 1000),
            (fn, dim, 3000),
            (fn, dim, 5000),
            (fn, dim, 10000),
        ]

        algorithms_names = [algo.value for algo in Algorithm]
        n_evals = [tc[2] for tc in test_cases]
        keys = [f"{fn}_{d}D_{n}" for fn, d, n in test_cases]
        results_dict = {}

        for i, (fn, d, n_iter) in enumerate(tqdm(test_cases, desc="Test cases")):
            all_results = run_multiple_experiments(
                fn, d, n_iter, n_runs=n_runs, base_seed=base_seed, beta=BETA
            )
            results_dict[keys[i]] = calculate_statistics(all_results)
        save_results_to_csv(results_dict, fn, exp_type="toy", result_dir=RESULT_DIR)
        save_iterations_to_csv(results_dict, fn, exp_type="toy", result_dir=RESULT_DIR)
