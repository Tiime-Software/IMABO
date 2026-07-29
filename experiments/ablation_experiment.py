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

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.optuna_bandit import OptunaBandit
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.utils.stats import (
    calculate_statistics,
    save_iterations_to_csv,
    save_results_to_csv,
)
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results" / "ablation_experiment"
RESULT_DIR.mkdir(exist_ok=True)

# ── Sub-experiment 1: TPE oracle impact ───────────────────────────────────────
TPE_DIMS = [1, 3, 5, 7, 10]
TPE_FUNCTIONS = ["sin1", "garland", "rastrigin"]
TPE_N_ITER = 5000
TPE_N_RUNS = 10

# ── Sub-experiment 2: MOSS / k impact ─────────────────────────────────────────
K_VALUES = [1, 10, 50, 70, 100, 200]
K_DIM = 4
K_FUNCTIONS = ["sin1", "garland", "rastrigin"]
K_N_ITER = 5000
K_N_RUNS = 10
BETA = 0.5

# ── Runners ───────────────────────────────────────────────────────────────────


def run_single(
    function_name: str,
    dim: int,
    n_iterations: int,
    seed: int,
    optimizer_type: Literal["random_suggest", "imabo", "tpe"],
    k: int = 1,
    beta: float = BETA,
) -> dict:
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed, noise_std=0.5)
    func = obj.get_function_by_name(function_name)  # built-in noise
    func_noiseless = obj.get_function_by_name(function_name, noise=False)  # noiseless
    fmax = obj.get_theoretical_max(function_name)
    search_space = obj.get_search_space(function_name)

    rng = np.random.default_rng(seed)

    if optimizer_type == "random_suggest":
        opt = IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=False,
            use_tpe=False,
            beta=beta,
        )
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = lambda: opt.best_x  # noqa: E731
    elif optimizer_type == "imabo":
        opt = IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=False,
            beta=beta,
        )
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = lambda: opt.best_x  # noqa: E731
    else:  # tpe
        opt = OptunaBandit(search_space=search_space, k=k, seed=seed)
        suggest_fn = opt.suggest
        observe_fn = opt.observe
        best_x_fn = opt.suggest_best

    regrets = []
    for _ in tqdm(
        range(n_iterations), desc=f"  {function_name} {dim}D runs", leave=False
    ):
        x = suggest_fn()
        noiseless = func_noiseless(x) / dim
        y = noiseless + obj.noise_rng.normal(0, obj.noise_std)
        regrets.append(fmax - noiseless)
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
    beta: float = BETA,
    n_jobs: int = 8,
) -> list[dict]:
    def _one_run(i: int) -> dict:
        seed = base_seed + i * 1000
        return {
            opt_name: run_single(
                function_name, dim, n_iterations, seed, opt_name, k, beta
            )
            for opt_name in optimizers
        }

    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_one_run)(i) for i in range(n_runs)
    )


# ── Sub-experiment 1 ──────────────────────────────────────────────────────────


def run_tpe_ablation(
    n_runs: int = TPE_N_RUNS,
    base_seed: int = 42,
    beta: float = BETA,
) -> dict:
    """I-MOSS-TPE vs I-MOSS across dimensions, one plot per function."""
    algorithms = ["I-MOSS", "I-MOSS-TPE"]
    opt_types = ["random_suggest", "imabo"]
    exp_type = f"tpe_ablation_beta_{beta}"

    all_results_dict: dict = {}
    for fn in TPE_FUNCTIONS:
        print(f"\n[TPE ablation] {fn} (beta={beta})")
        results_dict: dict = {}

        for dim in TPE_DIMS:
            key = f"{fn}_{dim}D_{TPE_N_ITER}"
            raw = run_experiment(
                fn, dim, opt_types, TPE_N_ITER, n_runs, base_seed, beta=beta
            )
            # rename keys to display names
            renamed = [{algorithms[j]: r[opt_types[j]] for j in range(2)} for r in raw]
            results_dict[key] = calculate_statistics(renamed)

        save_results_to_csv(results_dict, fn, exp_type=exp_type, result_dir=RESULT_DIR)
        save_iterations_to_csv(
            results_dict, fn, exp_type=exp_type, result_dir=RESULT_DIR
        )
        all_results_dict[fn] = results_dict

    return all_results_dict


# ── Sub-experiment 2 ──────────────────────────────────────────────────────────


def run_k_ablation(
    n_runs: int = K_N_RUNS,
    base_seed: int = 42,
    beta: float = BETA,
) -> dict:
    """I-MOSS-TPE vs TPE with varying k, one plot per function."""
    all_results: dict = {}
    for fn in K_FUNCTIONS:
        print(f"\n[k ablation] {fn} (beta={beta})")
        results: dict = {}

        # I-MOSS-TPE (run once, shared across k comparisons)
        raw_imabo = run_experiment(
            fn,
            K_DIM,
            ["imabo"],
            K_N_ITER,
            n_runs,
            base_seed,
            beta=beta,
        )
        imabo_stats = calculate_statistics(
            [{"I-MOSS-TPE": r["imabo"]} for r in raw_imabo]
        )
        results["I-MOSS-TPE"] = imabo_stats["I-MOSS-TPE"]

        for k in tqdm(K_VALUES, desc=f"{fn} k-values", leave=True):
            raw_k = run_experiment(
                fn,
                K_DIM,
                ["tpe"],
                K_N_ITER,
                n_runs,
                base_seed + k,
                k=k,
                beta=beta,
            )
            k_stats = calculate_statistics([{"tpe": r["tpe"]} for r in raw_k])
            results[f"TPE k={k}"] = k_stats["tpe"]

        _save_k_results(results, fn, beta=beta)
        all_results[fn] = results

    return all_results


def _save_k_results(results: dict, function_name: str, beta: float = BETA) -> None:
    # Summary (one row per algorithm)
    summary_rows = []
    iter_rows = []

    for alg, data in results.items():
        k_val = None if alg == "I-MOSS-TPE" else int(alg.split("k=")[1])
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

    summary_path = RESULT_DIR / f"k_ablation_{function_name}_beta_{beta}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved {summary_path}")

    iter_path = (
        RESULT_DIR / f"k_ablation_{function_name}_{K_N_ITER}_beta_{beta}_iterations.csv"
    )
    with open(iter_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=iter_rows[0].keys())
        writer.writeheader()
        writer.writerows(iter_rows)
    print(f"  Saved {iter_path}")


if __name__ == "__main__":
    # print("=== Sub-experiment 1: TPE oracle impact ===")
    # run_tpe_ablation()

    print("\n=== Sub-experiment 2: MOSS oracle / k impact ===")
    run_k_ablation()
