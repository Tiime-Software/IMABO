"""
Toy-function experiment: compare IMABO against StoSOO, HOO-T, and Stroquool.

Usage (from repo root):
    python -m experiments.toy_experiment
    python -m experiments.toy_experiment --algorithm IMOSS-mutate-KLxTPE
    python -m experiments.toy_experiment --quick  # fast smoke test
    python -m experiments.toy_experiment --plot-only  # (re)draw the figure only
"""

import argparse
import json
import time
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
from imabo import IMOSSTPE, load_tabpfn

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
            IMOSSTPE(search_space, beta=beta, seed=seed, multivariate=True),
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


def run_optimization(
    function_name: str,
    dim: int,
    beta: float = 0.5,
    n_iterations: int = 1000,
    seed: int = 42,
    algorithms: list[Algorithm] | None = None,
) -> dict[str, RegretData]:
    """Run one seed of the experiment, checkpointed per algorithm.

    Each algorithm gets its own ObjectiveFunctions instance so its noise
    stream depends only on the seed, not on which algorithms ran before it in
    the same process -- a checkpoint-resumed run is identical to a fresh one.
    """
    algorithms = list(Algorithm) if algorithms is None else algorithms

    regrets: dict[str, RegretData] = {}
    for algo in algorithms:
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
        func_noiseless = obj_func.get_function_by_name(function_name)
        fmax = obj_func.get_theoretical_max(function_name)
        search_space = obj_func.get_search_space(function_name)
        if func_noiseless is None:
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
            if function_name == "gaussian":
                # Match coordination_barrier_experiment.py's observation model
                # for this landscape exactly: a single Bernoulli draw with
                # success probability mu(x) (noiseless is already mu(x), the
                # dim scaling cancels), not additive Gaussian noise.
                y = float(obj_func.noise_rng.random() < noiseless)
            else:
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
    algorithms: list[Algorithm] | None = None,
) -> list[dict[str, RegretData]]:
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_optimization)(
            function_name,
            dim,
            beta,
            n_iterations,
            base_seed + i * 1000,
            algorithms=algorithms,
        )
        for i in range(n_runs)
    )


def make_plot(
    functions,
    columns: int = 1,
    save_fig: bool = True,
    conference: str = "arxiv",
) -> None:
    """Draw the paper's toy-function appendix figure: performance
    trajectories (simple vs. cumulative regret) for every algorithm, one
    panel per function. Reads each function's `{fn}_toy_summary.csv` (written
    by save_results_to_csv after a run), not the raw checkpoints.
    """
    # Head-less: make the plotting helper's trailing ``plt.show()`` a no-op so
    # the PDF is written without a GUI (an interactive backend would block).
    import matplotlib

    matplotlib.use("Agg")
    from experiments.utils.plots.toy_plot import plot_multiple_trajectories

    print(f"Generating toy performance-trajectories figure ({', '.join(functions)})...")
    plot_multiple_trajectories(
        functions,
        save_fig=save_fig,
        exp_type="toy",
        columns=columns,
        conference=conference,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--algorithm",
        default="all",
        choices=["all"] + [a.value for a in Algorithm],
        help="method to run (default: 'all' -- every algorithm)",
    )
    p.add_argument(
        "--functions",
        nargs="+",
        default=["sin1", "garland", "rastrigin", "gaussian"],
        help="toy objective functions to run (see ObjectiveFunctions)",
    )
    p.add_argument("--dim", type=int, default=4, help="search-space dimensionality")
    p.add_argument(
        "--n-iters",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000, 10000],
        help="iteration budgets (T) to sweep, one run per value",
    )
    p.add_argument(
        "--n-runs", type=int, default=20, help="independent seeds per algorithm"
    )
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=8, help="parallel seed workers")
    p.add_argument(
        "--plot", action="store_true", help="after running, draw the paper figure"
    )
    p.add_argument(
        "--plot-only", action="store_true", help="skip running, only (re)plot"
    )
    p.add_argument("--no-plot", action="store_true", help="run but skip plotting")
    p.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke test: T=60, 2 runs, sin1 only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.quick:
        functions, n_iters, n_runs = ["sin1"], [60], 2
    else:
        functions, n_iters, n_runs = args.functions, args.n_iters, args.n_runs

    dim = args.dim
    algorithms = (
        list(Algorithm) if args.algorithm == "all" else [Algorithm(args.algorithm)]
    )

    if not args.plot_only:
        total_tasks = len(functions) * len(n_iters)
        start = time.time()
        with tqdm(total=total_tasks, desc="function x budget", unit="task") as bar:
            for fn in functions:
                print(f"Running {fn}...")
                test_cases = [(fn, dim, n_iter) for n_iter in n_iters]
                keys = [f"{fn}_{d}D_{n}" for fn, d, n in test_cases]
                results_dict = {}

                for i, (fn_, d, n_iter) in enumerate(test_cases):
                    all_results = run_multiple_experiments(
                        fn_,
                        d,
                        n_iter,
                        n_runs=n_runs,
                        base_seed=args.base_seed,
                        n_jobs=args.n_jobs,
                        beta=BETA,
                        algorithms=algorithms,
                    )
                    results_dict[keys[i]] = calculate_statistics(all_results)
                    bar.update(1)
                    done, total = bar.n, bar.total
                    elapsed = time.time() - start
                    eta = elapsed / done * (total - done) if done else 0.0
                    bar.set_postfix_str(f"elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m")

                save_results_to_csv(
                    results_dict, fn, exp_type="toy", result_dir=RESULT_DIR
                )
                save_iterations_to_csv(
                    results_dict, fn, exp_type="toy", result_dir=RESULT_DIR
                )

    # Plot when asked (--plot/--plot-only) or by default after an "all" run.
    want_plot = not args.no_plot and (
        args.plot or args.plot_only or args.algorithm == "all"
    )
    if want_plot:
        make_plot(functions, columns=1, save_fig=True)
