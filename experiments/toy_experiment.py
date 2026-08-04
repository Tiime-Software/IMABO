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

from experiments.baselines.hier_mab import HierMAB
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
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
    # Best surrogate-free explore oracle from the RF-tabular comparison: mutate
    # the best arm so far on a coordinate chosen by a KL-UCB bandit, with the new
    # value drawn by a univariate TPE. Unlike Hier-MAB it needs no axis grid --
    # the TPE proposes on the real interval -- so nothing is discretised here.
    IMOSS_MUTATE_KLXTPE = "IMOSS-mutate-KLxTPE"
    # ... plus an extra bandit arm proposing a whole multivariate-TPE config, so
    # the bandit decides how often to jump globally instead of hill-climbing.
    IMOSS_MUTATE_KLXTPE_GLOBAL = "IMOSS-mutate-KLxTPE-global"
    # ... and the two per-parent variants: one coordinate bandit PER PARENT
    # rather than one for the run, at both confidence widths. The credit of
    # mutating a coordinate is conditional on the arm being mutated, so these
    # condition on it instead of pooling across incumbents. On the RF grid the
    # KL one was a wash (+14 ± 23 over 30 seeds); these continuous landscapes
    # have far more distinct incumbents, which is where it should either pay
    # off or clearly fail.
    IMOSS_MUTATE_KLXTPE_PERPARENT = "IMOSS-mutate-KLxTPE-perparent"
    IMOSS_MUTATE_UCBXTPE_PERPARENT = "IMOSS-mutate-UCBxTPE-perparent"
    # The TabPFN oracle at the settings tuned across the RF grid and the 2-D HPO
    # boxes: a LOCAL mutation step (mutate_value resamples a continuous axis over
    # its whole domain, which is not a neighbour), a refit at every explore step
    # (the default 10 leaves ~89% of proposals coming off a stale shortlist), and
    # a 0.841 acquisition quantile (0.99 rewards predictive variance, which on a
    # wide continuous pool just picks the least familiar candidate). These toys
    # are 4-D and continuous -- between the two regimes where those settings were
    # tuned -- so they are the case most likely to break them.
    IMOSS_TABPFN_TUNED = "IMOSS-TabPFN-tuned"
    # The same at quantile 0.99. On the coordination-barrier landscapes 0.841
    # was the whole failure: it took the uniform share of the pool from 12% of
    # argmaxes to 0 of 183, and restoring 0.99 moved family_d2 by -313.8 +- 99.5.
    # These toys are the remaining case that ran at 0.841 and lost.
    IMOSS_TABPFN_TUNED_Q99 = "IMOSS-TabPFN-tuned-q0.99"
    IMOSS_TABPFN_TUNED_Q90 = "IMOSS-TabPFN-tuned-q0.9"
    IMOSS_TABPFN_TUNED_Q975 = "IMOSS-TabPFN-tuned-q0.975"


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
    if algo == Algorithm.HIER_MAB_11:
        return HierMAB(search_space, n_points=11, seed=seed), False
    if algo == Algorithm.IMOSS_MUTATE_KLXTPE_GLOBAL:
        return (
            IMABOCoordUCB(
                search_space=search_space,
                seed=seed,
                beta=beta,
                parent_rule="best",
                coord_rule="ucb",
                value_rule="tpe",
                credit_rule="arm_mean",
                bandit_bonus="kl",
                global_tpe_arm=True,
            ),
            False,
        )
    if algo in (Algorithm.IMOSS_TABPFN_TUNED, Algorithm.IMOSS_TABPFN_TUNED_Q99,
                Algorithm.IMOSS_TABPFN_TUNED_Q90,
                Algorithm.IMOSS_TABPFN_TUNED_Q975):
        return (
            IMABOTabPFN(
                search_space=search_space,
                seed=seed,
                beta=beta,
                candidate_source="mutation",
                parent_rule="best",
                candidate_uniform_frac=0.1,
                mutation_scale=0.1,
                refit_every=1,
                quantile={
                    Algorithm.IMOSS_TABPFN_TUNED_Q99: 0.99,
                    Algorithm.IMOSS_TABPFN_TUNED_Q90: 0.9,
                    Algorithm.IMOSS_TABPFN_TUNED_Q975: 0.975,
                }.get(algo, 0.841),
                tabpfn_model=_tabpfn_model(),
            ),
            False,
        )
    if algo == Algorithm.IMOSS_MUTATE_KLXTPE:
        return (
            IMABOCoordUCB(
                search_space=search_space,
                seed=seed,
                beta=beta,
                parent_rule="best",
                coord_rule="ucb",
                value_rule="tpe",
                credit_rule="arm_mean",
                bandit_bonus="kl",
            ),
            False,
        )
    if algo in (
        Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT,
        Algorithm.IMOSS_MUTATE_UCBXTPE_PERPARENT,
    ):
        return (
            IMABOCoordUCB(
                search_space=search_space,
                seed=seed,
                beta=beta,
                parent_rule="best",
                coord_rule="ucb",
                value_rule="tpe",
                credit_rule="arm_mean",
                bandit_bonus=(
                    "kl"
                    if algo is Algorithm.IMOSS_MUTATE_KLXTPE_PERPARENT
                    else "hoeffding"
                ),
                coord_bandit_scope="parent",
            ),
            False,
        )
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
