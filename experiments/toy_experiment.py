"""
Toy-function experiment: compare IMABO against StoSOO, HOO-T, and Stroquool.

Usage (from repo root):
    python -m experiments.toy_experiment
"""

import json
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.baselines.hier_mab import HierMAB
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from experiments.utils.stats import (
    calculate_statistics,
    save_iterations_to_csv,
    save_results_to_csv,
)
from imabo import IMABO, IMABOCoordUCB, IMABOTabPFN
from imabo.tabpfn_optimizer import load_tabpfn

RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
# Per-(function, budget, seed, algorithm) checkpoints: reruns skip finished
# work and adding an algorithm later never recomputes the others.
CKPT_DIR = RESULT_DIR / "toy_runs"
CKPT_DIR.mkdir(exist_ok=True)

BETA = 0.5


class Algorithm(Enum):
    IMABO = "IMABO"
    STOSOO = "StoSOO"
    HOO_T = "HOO-T"
    STROQUOOL = "Stroquool"
    # Factored baseline (AutoRAG-HP) on an 11-point grid per axis
    # (linspace(lower, upper, 11) per coordinate via HierMAB.axis_values).
    HIER_MAB_11 = "Hier-MAB-11"
    # The two tuned explore oracles -- see winning_configs.pdf.
    IMOSS_MUTATE_KLXTPE = "IMOSS-mutate-KLxTPE"
    IMOSS_TABPFN_TUNED = "IMOSS-TabPFN-tuned"


class RegretData(TypedDict):
    regrets: list[float]
    simple_regrets: float


def _rescale_to_search_space(z: np.ndarray, search_space: dict) -> np.ndarray:
    """Map a point in [0,1]^dim (as produced by the tree baselines) onto the
    actual per-dimension [lower, upper] bounds declared in ``search_space``."""
    bounds = np.array([[v["lower"], v["upper"]] for v in search_space.values()])
    lower, upper = bounds[:, 0], bounds[:, 1]
    return lower + (upper - lower) * np.asarray(z)


_TABPFN_MODEL = None


def _tabpfn_model():
    global _TABPFN_MODEL
    if _TABPFN_MODEL is None:
        _TABPFN_MODEL = load_tabpfn()
    return _TABPFN_MODEL


def _build_optimizer(
    algo: Algorithm,
    search_space: dict,
    dim: int,
    n_iterations: int,
    seed: int,
    beta: float,
):
    """(optimizer, is_xarm_style) for one algorithm. Tree-based algorithms
    suggest points in [0,1]^dim; IMABO and Hier-MAB suggest config dicts on
    the native domain."""
    if algo == Algorithm.IMABO:
        return (
            IMABO(search_space=search_space, seed=seed, multivariate=False,
                  beta=beta),
            False,
        )
    if algo == Algorithm.STOSOO:
        return TimedOptimizer(stosoo, n_iterations, dim), True
    if algo == Algorithm.HOO_T:
        return TimedOptimizer(hoo_t, n_iterations, dim, rho=0.4, nu1=10.0), True
    if algo == Algorithm.STROQUOOL:
        return TimedOptimizer(stroquool, n_iterations, dim), True
    if algo == Algorithm.IMOSS_MUTATE_KLXTPE:
        return IMABOCoordUCB(search_space=search_space, seed=seed, beta=beta), False
    if algo == Algorithm.IMOSS_TABPFN_TUNED:
        return (
            IMABOTabPFN(
                search_space=search_space,
                seed=seed,
                beta=beta,
                candidate_source="mutation",
                candidate_uniform_frac=0.1,
                mutation_scale=0.1,
                refit_every=1,
                quantile=0.975,
                tabpfn_model=_tabpfn_model(),
            ),
            False,
        )
    if algo == Algorithm.HIER_MAB_11:
        return HierMAB(search_space, n_points=11, seed=seed), False
    raise ValueError(f"unknown algorithm: {algo!r}")


def run_optimization(
    function_name: str,
    dim: int,
    beta: float = 0.5,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, RegretData]:
    """Run one seed of the experiment, checkpointed per algorithm.

    Each algorithm gets its own ObjectiveFunctions instance so its noise
    stream depends only on the seed, not on which algorithms ran before it in
    the same process -- a checkpoint-resumed run is identical to a fresh one.
    """
    regrets: dict[str, RegretData] = {}
    for algo in Algorithm:
        opt_name = algo.value
        ckpt = (
            CKPT_DIR
            / f"{function_name}_{dim}D_{n_iterations}iters_seed{seed}_{opt_name}.json"
        )
        if ckpt.exists():
            with open(ckpt) as f:
                regrets[opt_name] = json.load(f)
            continue

        obj_func = ObjectiveFunctions(dim=dim, noise_seed=seed)
        func = obj_func.get_function_by_name(function_name)
        func_noiseless = obj_func.get_function_by_name(function_name, noise=False)
        fmax = obj_func.get_theoretical_max(function_name)
        search_space = obj_func.get_search_space(function_name)
        if func is None:
            raise ValueError(f"Unknown function: {function_name}")

        opt, is_xarm = _build_optimizer(
            algo, search_space, dim, n_iterations, seed, beta
        )

        data: RegretData = {"regrets": [], "simple_regrets": float("inf")}
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
            data["regrets"].append(regret)

        if is_xarm:
            best_x = _rescale_to_search_space(opt.suggest_best(), search_space)
        else:
            best_x = opt.best_x
        data["simple_regrets"] = fmax - func_noiseless(best_x) / dim

        with open(ckpt, "w") as f:
            json.dump(data, f)
        regrets[opt_name] = data

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
