"""
Main HPO experiment: IMABO vs StoSOO, HOO-T, and StroquOOL on real ML benchmarks.

Reproduces the results in Section "Real-World HPO Benchmarks" of the paper.
Benchmarks: Logistic Regression (lr) and SVM (svm) from HPOBench.
Budgets: T ∈ {1000, 3000, 5000, 10000}, 20 independent runs.

Tree-based algorithms (StoSOO, HOO-T, StroquOOL) operate on [0,1]^d coordinates.
Their suggestions are converted to actual hyperparameter values via
HPOBenchmark.array_to_config() before evaluation.

Requires the HPO benchmark server (Docker):
    docker compose up hpo-server

Usage (from repo root):
    python -m experiments.hpo_experiment
"""

import asyncio
import csv
import json
from enum import Enum
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm

from experiments.baselines.hier_mab import HierMAB
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from experiments.benchmarks.config import BENCHMARKS
from experiments.benchmarks.hpo_bench.client import (
    api_call,
    start_hpo_server,
    stop_hpo_server,
)
from experiments.benchmarks.hpo_wrapper import HPOBenchmark
from experiments.utils.stats import calculate_statistics
from imabo import IMABO, IMABOCoordUCB, IMABOTabPFN
from imabo.tabpfn_optimizer import load_tabpfn

RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
# Per-(benchmark, budget, seed, algorithm) checkpoints, so reruns skip
# finished work and adding an algorithm later never recomputes the others.
CKPT_DIR = RESULT_DIR / "hpo_continuous"
CKPT_DIR.mkdir(exist_ok=True)
BETA = 0.5


class Algorithm(Enum):
    IMABO = "IMABO"
    STOSOO = "StoSOO"
    HOO_T = "HOO-T"
    STROQUOOL = "Stroquool"
    # Hier-MAB (AutoRAG-HP) needs each axis as an explicit finite set: both
    # hyperparameters of these log-scaled 2D spaces are discretized to a
    # geometric grid -- np.geomspace(lower, upper, m) -- with m = 10 or 100
    # values per axis (HierMAB's axis_values does exactly this for log axes).
    HIER_MAB_10 = "Hier-MAB-10"
    HIER_MAB_100 = "Hier-MAB-100"
    # The two tuned explore oracles -- see winning_configs.pdf.
    IMOSS_MUTATE_KLXTPE = "IMOSS-mutate-KLxTPE"
    IMOSS_TABPFN_TUNED = "IMOSS-TabPFN-tuned"


class RegretData(TypedDict):
    regrets: list[float]
    simple_regrets: float


_TABPFN_MODEL = None


def _tabpfn_model():
    global _TABPFN_MODEL
    if _TABPFN_MODEL is None:
        _TABPFN_MODEL = load_tabpfn()
    return _TABPFN_MODEL


def build_optimizer(
    algo: Algorithm,
    benchmark_obj: HPOBenchmark,
    n_iterations: int,
    seed: int,
    beta: float,
) -> tuple:
    """(optimizer, is_tree_based) for one algorithm.

    Tree-based algorithms suggest points in [0,1]^d; IMABO and Hier-MAB
    suggest config dicts.
    """
    dim = benchmark_obj.dim
    if algo == Algorithm.IMABO:
        return (
            IMABO(
                search_space=benchmark_obj.param_specs,
                seed=seed,
                multivariate=True,
                beta=beta,
            ),
            False,
        )
    if algo == Algorithm.STOSOO:
        return TimedOptimizer(stosoo, n_iterations, dim), True
    if algo == Algorithm.HOO_T:
        return TimedOptimizer(hoo_t, n_iterations, dim, rho=0.4, nu1=10.0), True
    if algo == Algorithm.STROQUOOL:
        return TimedOptimizer(stroquool, n_iterations, dim), True
    if algo == Algorithm.IMOSS_MUTATE_KLXTPE:
        return (
            IMABOCoordUCB(
                search_space=benchmark_obj.param_specs, seed=seed, beta=beta
            ),
            False,
        )
    if algo == Algorithm.IMOSS_TABPFN_TUNED:
        return (
            IMABOTabPFN(
                search_space=benchmark_obj.param_specs,
                seed=seed,
                beta=beta,
                tabpfn_model=_tabpfn_model(),
            ),
            False,
        )
    if algo == Algorithm.HIER_MAB_10:
        return HierMAB(benchmark_obj.param_specs, n_points=10, seed=seed), False
    if algo == Algorithm.HIER_MAB_100:
        return HierMAB(benchmark_obj.param_specs, n_points=100, seed=seed), False
    raise ValueError(f"unknown algorithm: {algo!r}")


async def run_single_experiment(
    benchmark: str,
    n_iterations: int,
    seed: int = 42,
    beta: float = 0.5,
    algorithms: list[Algorithm] | None = None,
) -> dict[str, RegretData]:
    """Run one seed of the experiment, checkpointed per algorithm.

    Each algorithm gets its own ``HPOBenchmark`` instance so its reward-sample
    stream depends only on the seed (the wrapper draws the per-pull sample
    with ``random.Random(seed)``), not on which algorithms ran before it in
    the same process -- that keeps a checkpoint-resumed run identical to a
    fresh one. Server-side objective values are joblib-cached on disk keyed
    by config, so repeated pulls of a config never re-train a model.
    """
    algorithms = list(Algorithm) if algorithms is None else algorithms

    regrets: dict[str, RegretData] = {}
    for algo in algorithms:
        opt_name = algo.value
        ckpt = (
            CKPT_DIR / f"{benchmark}_{n_iterations}iters_seed{seed}_{opt_name}.json"
        )
        if ckpt.exists():
            with open(ckpt) as f:
                regrets[opt_name] = json.load(f)
            continue

        benchmark_obj = HPOBenchmark(benchmark_name=benchmark, seed=seed)
        opt, is_tree = build_optimizer(algo, benchmark_obj, n_iterations, seed, beta)

        data: RegretData = {"regrets": [], "simple_regrets": float("inf")}
        for _ in tqdm(range(n_iterations), desc=f"{benchmark}/{opt_name}", leave=False):
            if is_tree and opt.done:
                continue

            x = opt.suggest()
            # Tree-based algorithms return [0,1]^d arrays; convert to config dict.
            eval_config = benchmark_obj.array_to_config(x) if is_tree else x

            result = await benchmark_obj.eval_config(eval_config)
            reward = result.get("sample_result", float("-inf"))
            # Regret = 1 - val_acc (minimisation of error)
            data["regrets"].append(1.0 - result.get("avg_result", float("-inf")))

            if is_tree:
                opt.observe(x, reward)
            else:
                opt.observe(reward)

        best = opt.suggest_best() if is_tree else opt.best_x
        if is_tree:
            best = benchmark_obj.array_to_config(best)
        # Make sure the recommendation has been evaluated: Hier-MAB's incumbent
        # is the per-axis argmax combination, which the run may never have
        # served as a full config (get_config_value would return -inf for it).
        # A cache hit for everyone else.
        await benchmark_obj.eval_config(best)
        data["simple_regrets"] = 1.0 - benchmark_obj.get_config_value(best)

        with open(ckpt, "w") as f:
            json.dump(data, f)
        regrets[opt_name] = data

    return regrets


async def run_multiple_experiments(
    benchmark: str,
    n_iterations: int,
    n_runs: int = 20,
    base_seed: int = 42,
    beta: float = 0.5,
    algorithms: list[Algorithm] | None = None,
) -> list[dict[str, RegretData]]:
    """Run multiple independent runs of the experiment."""
    results = []
    for i in tqdm(
        range(n_runs), desc=f"  {benchmark} {n_iterations} runs", leave=False
    ):
        results.append(
            await run_single_experiment(
                benchmark,
                n_iterations,
                seed=base_seed + i,
                beta=beta,
                algorithms=algorithms,
            )
        )
    return results


def save_results_to_csv(
    results_dict: dict, benchmark: str, exp_type: str = "hpo", beta: float = BETA
) -> None:
    summary_rows = []
    for key, stats in results_dict.items():
        parts = key.split("_")
        bench_name = parts[0]
        dim = int(parts[1].replace("D", ""))
        n_iter = int(parts[2])
        for algorithm, data in stats.items():
            summary_rows.append(
                {
                    "benchmark": bench_name,
                    "dimension": dim,
                    "n_iterations": n_iter,
                    "algorithm": algorithm,
                    "simple_regret_mean": data["simple_regrets"]["mean"],
                    "simple_regret_std": data["simple_regrets"]["std"],
                    "total_regret_mean": float(data["regrets"]["sum_regrets"].mean()),
                    "total_regret_std": float(data["regrets"]["sum_regrets"].std()),
                }
            )
    path = RESULT_DIR / f"{benchmark}_{exp_type}_beta_{beta}_summary.csv"
    with open(path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary saved to {path}")


def save_iterations_to_csv(
    results_dict: dict, benchmark: str, exp_type: str = "hpo", beta: float = BETA
) -> None:
    rows = []
    for key, stats in results_dict.items():
        parts = key.split("_")
        bench_name = parts[0]
        dim = int(parts[1].replace("D", ""))
        n_iter = int(parts[2])
        for algorithm, data in stats.items():
            for i, (mean, std) in enumerate(
                zip(data["regrets"]["mean"], data["regrets"]["std"])
            ):
                rows.append(
                    {
                        "benchmark": bench_name,
                        "dimension": dim,
                        "n_iterations": n_iter,
                        "algorithm": algorithm,
                        "iteration": i + 1,
                        "regret_mean": mean,
                        "regret_std": std,
                    }
                )
    path = RESULT_DIR / f"{benchmark}_{exp_type}_beta_{beta}_iterations.csv"
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Iterations saved to {path}")


async def main():
    benchmarks_to_run = ["lr", "svm"]
    n_runs = 20
    base_seed = 42

    test_cases_per_benchmark = [
        (10000,),
        (5000,),
        (3000,),
        (1000,),
    ]

    if not await start_hpo_server():
        print("Failed to start server")
        return

    try:
        for benchmark in benchmarks_to_run:
            print(f"\n=== Benchmark: {benchmark} ===")

            if not await api_call("load", {"benchmark": benchmark}):
                print(f"Failed to load {benchmark}, skipping.")
                continue

            dim = len(BENCHMARKS[benchmark]["param_specs"])
            test_cases = [(benchmark, n) for (n,) in test_cases_per_benchmark]
            keys = [f"{b}_{dim}D_{n}" for b, n in test_cases]
            n_evals = [n for _, n in test_cases]
            results_dict = {}

            for i, (bench, n_iter) in enumerate(test_cases):
                print(f"  T={n_iter}, {n_runs} runs...")
                all_results = await run_multiple_experiments(
                    benchmark=bench,
                    n_iterations=n_iter,
                    n_runs=n_runs,
                    base_seed=base_seed,
                    beta=BETA,
                )
                if all_results:
                    results_dict[keys[i]] = calculate_statistics(all_results)

            if results_dict:
                save_results_to_csv(results_dict, benchmark, exp_type="hpo", beta=BETA)
                save_iterations_to_csv(
                    results_dict, benchmark, exp_type="hpo", beta=BETA
                )
    finally:
        await stop_hpo_server()


if __name__ == "__main__":
    # No nest_asyncio here: applying it patches the event loop in a way that
    # makes every aiohttp request fail on Python >= 3.12 ("Timeout context
    # manager should be used inside a task"), which wait_for_server's bare
    # except then misreports as "server not up". Run as a plain script there
    # are no nested event loops to enable.
    asyncio.run(main())
