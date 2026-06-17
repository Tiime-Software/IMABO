"""
Ablation study: two sub-experiments from the paper.

1. TPE oracle impact — IMABO vs Random (use_tpe=False) across dims 1/3/5/7/10.
   Fixed budget T=3000, functions: sin1, garland, quadratic. 10 runs.

2. MOSS oracle / k impact — IMABO vs OptunaBandit with varying k values.
   Fixed dim=4, T=3000, functions: sin1, garland, rastrigin. 10 runs.

Usage (from repo root):
    python -m experiments.ablation_experiment
"""

import csv
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.baselines.optuna_bandit import OptunaBandit
from experiments.utils.stats import calculate_statistics, save_results_to_csv
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

# ── Sub-experiment 1: TPE oracle impact ───────────────────────────────────────
TPE_DIMS = [1, 3, 5, 7, 10]
TPE_FUNCTIONS = ["sin1", "garland", "quadratic"]
TPE_N_ITER = 3000
TPE_N_RUNS = 10

# ── Sub-experiment 2: MOSS / k impact ─────────────────────────────────────────
K_VALUES = [1, 10, 50, 100, 200]
K_DIM = 4
K_FUNCTIONS = ["sin1", "garland", "rastrigin"]
K_N_ITER = 3000
K_N_RUNS = 10


# ── Runners ───────────────────────────────────────────────────────────────────


def run_single(
    function_name: str,
    dim: int,
    n_iterations: int,
    seed: int,
    optimizer_type: Literal["random_suggest", "imabo", "optuna"],
    k: int = 1,
) -> dict:
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)
    func_noiseless = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    search_space = obj.get_search_space(function_name)

    if optimizer_type == "random_suggest":
        opt = IMABO(
            search_space=search_space, seed=seed, multivariate=False, use_tpe=False
        )
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = lambda: opt.best_x  # noqa: E731
    elif optimizer_type == "imabo":
        opt = IMABO(search_space=search_space, seed=seed, multivariate=False)
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = lambda: opt.best_x  # noqa: E731
    else:  # optuna
        opt = OptunaBandit(search_space=search_space, k=k, seed=seed)
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = opt.suggest_best

    regrets = []
    for _ in tqdm(
        range(n_iterations), desc=f"  {function_name} {dim}D runs", leave=False
    ):
        x = suggest_fn()
        y = func(x)
        regrets.append(fmax - func_noiseless(x) / dim)
        observe_fn(y)

    best_x = best_x_fn()
    simple_regret = (
        fmax - func_noiseless(best_x) / dim if best_x is not None else float("inf")
    )
    return {"regrets": regrets, "simple_regrets": simple_regret}


def run_experiment(
    function_name: str,
    dim: int,
    optimizers: list[str],
    n_iterations: int,
    n_runs: int,
    base_seed: int = 42,
    k: int = 1,
) -> list[dict]:
    all_results = []
    for i in tqdm(range(n_runs), desc=f"  {function_name} {dim}D runs", leave=False):
        seed = base_seed + i * 1000
        all_results.append(
            {
                opt_name: run_single(
                    function_name, dim, n_iterations, seed, opt_name, k
                )
                for opt_name in optimizers
            }
        )
    return all_results


# ── Sub-experiment 1 ──────────────────────────────────────────────────────────


def run_tpe_ablation(
    n_runs: int = TPE_N_RUNS,
    base_seed: int = 42,
):
    """IMABO vs Random across dimensions, one plot per function."""
    algorithms = ["Random", "IMABO"]
    opt_types = ["random_suggest", "imabo"]

    for fn in TPE_FUNCTIONS:
        print(f"\n[TPE ablation] {fn}")
        results_dict: dict = {}

        for dim in TPE_DIMS:
            key = f"{fn}_{dim}D_{TPE_N_ITER}"
            raw = run_experiment(fn, dim, opt_types, TPE_N_ITER, n_runs, base_seed)
            # rename keys to display names
            renamed = [{algorithms[j]: r[opt_types[j]] for j in range(2)} for r in raw]
            results_dict[key] = calculate_statistics(renamed)

        save_results_to_csv(
            results_dict, fn, exp_type="tpe_ablation", result_dir=RESULT_DIR
        )


# ── Sub-experiment 2 ──────────────────────────────────────────────────────────


def run_k_ablation(
    n_runs: int = K_N_RUNS,
    base_seed: int = 42,
    save_fig: bool = False,
):
    """IMABO vs OptunaBandit with varying k, one plot per function."""
    for fn in K_FUNCTIONS:
        print(f"\n[k ablation] {fn}")
        results: dict = {}

        # IMABO (run once, shared across k comparisons)
        raw_imabo = run_experiment(fn, K_DIM, ["imabo"], K_N_ITER, n_runs, base_seed)
        imabo_stats = calculate_statistics([{"IMABO": r["imabo"]} for r in raw_imabo])
        results["IMABO"] = imabo_stats["IMABO"]

        for k in tqdm(K_VALUES, desc=f"{fn} k-values", leave=True):
            raw_k = run_experiment(
                fn, K_DIM, ["optuna"], K_N_ITER, n_runs, base_seed + k, k=k
            )
            k_stats = calculate_statistics([{"optuna": r["optuna"]} for r in raw_k])
            results[f"Optuna k={k}"] = k_stats["optuna"]

        _save_k_results(results, fn)


def _save_k_results(results: dict, function_name: str) -> None:
    # Summary (one row per algorithm)
    summary_rows = []
    iter_rows = []

    for alg, data in results.items():
        k_val = None if alg == "IMABO" else int(alg.split("k=")[1])
        summary_rows.append(
            {
                "function": function_name,
                "dimension": K_DIM,
                "n_iterations": K_N_ITER,
                "algorithm": alg,
                "k": k_val,
                "simple_regret_mean": data["simple_regrets"]["mean"],
                "simple_regret_std": data["simple_regrets"]["std"],
            }
        )
        for i, (mean, std) in enumerate(
            zip(data["regrets"]["mean"], data["regrets"]["std"])
        ):
            iter_rows.append(
                {
                    "function": function_name,
                    "dimension": K_DIM,
                    "n_iterations": K_N_ITER,
                    "algorithm": alg,
                    "k": k_val,
                    "iteration": i + 1,
                    "regret_mean": mean,
                    "regret_std": std,
                }
            )

    summary_path = RESULT_DIR / f"k_ablation_{function_name}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved {summary_path}")

    # Save iterations CSV
    iter_rows = []
    for alg, data in results.items():
        k_value = None if alg == "IMABO" else int(alg.split("k=")[1])
        regrets_mean = data["regrets"]["mean"]
        regrets_std = data["regrets"]["std"]
        for i, (mean, std) in enumerate(zip(regrets_mean, regrets_std)):
            iter_rows.append(
                {
                    "function": function_name,
                    "dimension": K_DIM,
                    "n_iterations": K_N_ITER,
                    "algorithm": alg,
                    "k": k_value,
                    "regret_mean": mean,
                    "regret_std": std,
                    "iteration": i + 1,
                }
            )
    iter_path = RESULT_DIR / f"k_ablation_{function_name}_{K_N_ITER}_iterations.csv"
    with open(iter_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=iter_rows[0].keys())
        writer.writeheader()
        writer.writerows(iter_rows)
    print(f"  Saved {iter_path}")


if __name__ == "__main__":
    print("=== Sub-experiment 1: TPE oracle impact ===")
    run_tpe_ablation()

    print("\n=== Sub-experiment 2: MOSS oracle / k impact ===")
    run_k_ablation()
