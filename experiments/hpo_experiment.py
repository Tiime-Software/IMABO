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
from enum import Enum
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm

from experiments.benchmarks.config import BENCHMARKS
from experiments.benchmarks.hpo_bench.client import (
    api_call,
    start_hpo_server,
    stop_hpo_server,
)
from experiments.benchmarks.hpo_wrapper import HPOBenchmark
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from experiments.utils.stats import calculate_statistics
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


class Algorithm(Enum):
    IMABO = "IMABO"
    STOSOO = "StoSOO"
    HOO_T = "HOO-T"
    STROQUOOL = "Stroquool"


class RegretData(TypedDict):
    regrets: list[float]
    simple_regrets: float


async def run_single_experiment(
    benchmark: str,
    n_iterations: int,
    seed: int = 42,
) -> dict[str, RegretData]:
    """Run one seed of the experiment with all four algorithms."""
    benchmark_obj = HPOBenchmark(benchmark_name=benchmark, seed=seed)
    dim = benchmark_obj.dim

    imabo_opt = IMABO(
        search_space=benchmark_obj.param_specs,
        seed=seed,
        multivariate=True,
    )
    stosoo_opt = TimedOptimizer(stosoo, n_iterations, dim)
    hoo_t_opt = TimedOptimizer(hoo_t, n_iterations, dim, rho=0.4, nu1=10.0)
    stroquool_opt = TimedOptimizer(stroquool, n_iterations, dim)

    # (enum, optimizer, is_tree_based)
    # Tree-based algorithms suggest points in [0,1]^d; IMABO suggests dicts.
    opts_config = [
        (Algorithm.IMABO, imabo_opt, False),
        (Algorithm.STOSOO, stosoo_opt, True),
        (Algorithm.HOO_T, hoo_t_opt, True),
        (Algorithm.STROQUOOL, stroquool_opt, True),
    ]

    regrets: dict[str, RegretData] = {
        algo.value: {"regrets": [], "simple_regrets": float("inf")}
        for algo in Algorithm
    }

    for algo, opt, is_tree in opts_config:
        opt_name = algo.value
        for _ in tqdm(range(n_iterations), desc=f"{benchmark}/{opt_name}", leave=False):
            if is_tree and opt.done:
                continue

            x = opt.suggest()
            # Tree-based algorithms return [0,1]^d arrays; convert to config dict.
            eval_config = benchmark_obj.array_to_config(x) if is_tree else x

            result = await benchmark_obj.eval_config(eval_config)
            reward = result.get("sample_result", float("-inf"))
            # Regret = 1 - val_acc (minimisation of error)
            regrets[opt_name]["regrets"].append(
                1.0 - result.get("avg_result", float("-inf"))
            )

            if is_tree:
                opt.observe(x, reward)
            else:
                opt.observe(reward)

        best = opt.suggest_best() if is_tree else opt.best_x
        if is_tree:
            best = benchmark_obj.array_to_config(best)
        regrets[opt_name]["simple_regrets"] = 1.0 - benchmark_obj.get_config_value(best)

    return regrets


async def run_multiple_experiments(
    benchmark: str,
    n_iterations: int,
    n_runs: int = 20,
    base_seed: int = 42,
) -> list[dict[str, RegretData]]:
    """Run multiple independent runs of the experiment."""
    results = []
    for i in tqdm(
        range(n_runs), desc=f"  {benchmark} {n_iterations} runs", leave=False
    ):
        results.append(
            await run_single_experiment(benchmark, n_iterations, seed=base_seed + i)
        )
    return results


def save_results_to_csv(
    results_dict: dict, benchmark: str, exp_type: str = "hpo"
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
    path = RESULT_DIR / f"{benchmark}_{exp_type}_summary.csv"
    with open(path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary saved to {path}")


def save_iterations_to_csv(
    results_dict: dict, benchmark: str, exp_type: str = "hpo"
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
    path = RESULT_DIR / f"{benchmark}_{exp_type}_iterations.csv"
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
                )
                if all_results:
                    results_dict[keys[i]] = calculate_statistics(all_results)

            if results_dict:
                save_results_to_csv(results_dict, benchmark, exp_type="hpo")
                save_iterations_to_csv(results_dict, benchmark, exp_type="hpo")
    finally:
        await stop_hpo_server()


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(main())
