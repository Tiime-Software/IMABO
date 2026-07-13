"""
Sweep experiments.ablation_experiment's beta the same way
experiments.test_exp.imoss_beta_sweep sweeps IMABO's beta against MOSS-AIR:
parallel runs via joblib, printed summary table, plus a beta-comparison plot
built straight from the in-memory results (no CSV round-trip -- see
experiments.test_exp.ablation_beta_plot for plots built from
ablation_experiment.py's own saved CSVs instead).

Usage (from repo root):
    python -m experiments.test_exp.ablation_beta_compare
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from experiments.ablation_experiment import (
    K_DIM,
    K_FUNCTIONS,
    K_N_ITER,
    K_N_RUNS,
    K_VALUES,
    TPE_DIMS,
    TPE_FUNCTIONS,
    TPE_N_ITER,
    TPE_N_RUNS,
    run_single,
)
from experiments.utils.plots.plot_configs import (
    create_figure_legend,
    display_name,
    get_algorithm_color,
    save_figure,
)

RESULT_DIR = Path(__file__).parents[2] / "results"

# ── Fixed run config ──────────────────────────────────────────────────────────
BETAS = [0.5, 0.6, 0.7, 0.8]
N_TPE_RUNS = TPE_N_RUNS
N_K_RUNS = K_N_RUNS
BASE_SEED = 42
N_JOBS = 8
TPE_PLOT_DIM = 10  # dimension used for the TPE-ablation cumulative regret plot
TPE_ALGORITHMS = ["I-MOSS", "I-MOSS-TPE"]
SERIES_MARKERS = ["o", "^", "s", "D", "v", "p", "*", "X"]
SAVE_FIG = True


def one_tpe_run(fn, dim, n_iter, seed, betas):
    row = {}
    for beta in betas:
        row[("I-MOSS", beta)] = run_single(
            fn, dim, n_iter, seed, "random_suggest", beta=beta
        )
        row[("I-MOSS-TPE", beta)] = run_single(
            fn, dim, n_iter, seed, "imabo", beta=beta
        )
    return row


def sweep_tpe_ablation(
    dims=TPE_DIMS,
    functions=TPE_FUNCTIONS,
    n_iter=TPE_N_ITER,
    n_runs=N_TPE_RUNS,
    betas=BETAS,
    base_seed=BASE_SEED,
):
    """I-MOSS / I-MOSS-TPE across beta, one printed table per (function, dim).

    Returns {function: {dim: {(algo, beta): {"simple_regret_mean",
    "simple_regret_std", "regret_mean" (per-iteration array)}}}} for the
    plot_tpe_* functions below.
    """
    labels = [(algo, beta) for algo in TPE_ALGORITHMS for beta in betas]
    results = {}
    for fn in functions:
        results[fn] = {}
        for dim in dims:
            runs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(one_tpe_run)(fn, dim, n_iter, base_seed + r * 1000, betas)
                for r in range(n_runs)
            )
            print(f"\n[{fn} {dim}D]")
            dim_result = {}
            for algo, beta in labels:
                simple = np.array([r[(algo, beta)]["simple_regrets"] for r in runs])
                regrets = np.array([r[(algo, beta)]["regrets"] for r in runs])
                dim_result[(algo, beta)] = {
                    "simple_regret_mean": simple.mean(),
                    "simple_regret_std": simple.std(),
                    "regret_mean": regrets.mean(axis=0),
                }
                print(
                    f"   {algo:12s} β={beta:<4} simple={simple.mean():.4f}  "
                    f"cumreg={regrets.sum(axis=1).mean():8.1f}"
                )
            results[fn][dim] = dim_result
    return results


def one_k_run(fn, dim, n_iter, run_idx, base_seed, betas, k_values):
    row = {}
    imabo_seed = base_seed + run_idx * 1000
    for beta in betas:
        row[("I-MOSS", beta)] = run_single(
            fn, dim, n_iter, imabo_seed, "random_suggest", beta=beta
        )
        row[("I-MOSS-TPE", beta)] = run_single(
            fn, dim, n_iter, imabo_seed, "imabo", beta=beta
        )
    for k in k_values:
        tpe_seed = base_seed + k + run_idx * 1000
        row[("TPE", k)] = run_single(fn, dim, n_iter, tpe_seed, "tpe", k=k)
    return row


def sweep_k_ablation(
    functions=K_FUNCTIONS,
    dim=K_DIM,
    n_iter=K_N_ITER,
    n_runs=N_K_RUNS,
    betas=BETAS,
    k_values=K_VALUES,
    base_seed=BASE_SEED,
):
    """I-MOSS / I-MOSS-TPE across beta (+ TPE-k reference, which is beta-invariant),
    one table per function, all at the k-ablation's fixed K_DIM.

    Returns {function: {(algo, beta) | ("TPE", k): {"simple_regret_mean",
    "simple_regret_std", "regret_mean" (per-iteration array)}}} for the
    plot_k_* functions below.
    """
    results = {}
    for fn in functions:
        runs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(one_k_run)(fn, dim, n_iter, r, base_seed, betas, k_values)
            for r in range(n_runs)
        )
        print(f"\n[{fn} {dim}D]")
        fn_result = {}
        for algo in TPE_ALGORITHMS:
            for beta in betas:
                simple = np.array([r[(algo, beta)]["simple_regrets"] for r in runs])
                regrets = np.array([r[(algo, beta)]["regrets"] for r in runs])
                fn_result[(algo, beta)] = {
                    "simple_regret_mean": simple.mean(),
                    "simple_regret_std": simple.std(),
                    "regret_mean": regrets.mean(axis=0),
                }
                print(
                    f"   {algo:12s} β={beta:<4} simple={simple.mean():.4f}  "
                    f"cumreg={regrets.sum(axis=1).mean():8.1f}"
                )
        for k in k_values:
            simple = np.array([r[("TPE", k)]["simple_regrets"] for r in runs])
            regrets = np.array([r[("TPE", k)]["regrets"] for r in runs])
            fn_result[("TPE", k)] = {
                "simple_regret_mean": simple.mean(),
                "simple_regret_std": simple.std(),
                "regret_mean": regrets.mean(axis=0),
            }
            print(
                f"   {'TPE k=' + str(k):12s} {'(β-invariant)':8} "
                f"simple={simple.mean():.4f}  cumreg={regrets.sum(axis=1).mean():8.1f}"
            )
        results[fn] = fn_result
    return results


# ── Plots (built straight from the in-memory sweep results above, no CSVs) ───


def plot_tpe_simple_regret(results, betas=BETAS, save_fig=SAVE_FIG):
    """Simple regret vs dimension, one subplot per function, one line per
    (algorithm, beta) pair."""
    functions = list(results.keys())
    fig, axes = plt.subplots(1, len(functions), figsize=(18, 5), squeeze=False)
    axes = axes[0]

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        dims = sorted(results[fn].keys())
        for series_idx, (algo, beta) in enumerate(
            (algo, beta) for algo in TPE_ALGORITHMS for beta in betas
        ):
            means = [
                results[fn][dim][(algo, beta)]["simple_regret_mean"] for dim in dims
            ]
            stds = [results[fn][dim][(algo, beta)]["simple_regret_std"] for dim in dims]
            ax.errorbar(
                dims,
                means,
                yerr=stds,
                color=get_algorithm_color(series_idx),
                marker=SERIES_MARKERS[series_idx % len(SERIES_MARKERS)],
                markersize=8,
                linewidth=2.0,
                capsize=5,
                capthick=2,
                label=f"{display_name(algo)} (β={beta})",
            )
        if idx == 0:
            ax.set_ylabel("Simple Regret", fontweight="bold", fontsize=22)
        ax.set_xlabel("Dimension", fontweight="bold", fontsize=22)
        ax.set_title(fn.capitalize(), fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=4, bbox_y=1.16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR / f"beta_compare_tpe_ablation_simple_regret_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


def plot_tpe_cumulative_regret(
    results, betas=BETAS, dim=TPE_PLOT_DIM, save_fig=SAVE_FIG
):
    """Cumulative regret vs iteration at a fixed dimension, one subplot per function,
    one line per (algorithm, beta) pair."""
    functions = list(results.keys())
    fig, axes = plt.subplots(1, len(functions), figsize=(18, 5), squeeze=False)
    axes = axes[0]

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        for series_idx, (algo, beta) in enumerate(
            (algo, beta) for algo in TPE_ALGORITHMS for beta in betas
        ):
            cumulative = np.cumsum(results[fn][dim][(algo, beta)]["regret_mean"])
            iterations = np.arange(1, len(cumulative) + 1)
            ax.plot(
                iterations,
                cumulative,
                color=get_algorithm_color(series_idx),
                marker=SERIES_MARKERS[series_idx % len(SERIES_MARKERS)],
                markevery=max(1, len(iterations) // 6),
                markersize=8,
                linewidth=2.0,
                label=f"{display_name(algo)} (β={beta})",
            )
        if idx == 1:
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=22)
        if idx == 0:
            ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)
        ax.set_title(f"{fn.capitalize()} ({dim}D)", fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=4, bbox_y=1.16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR
            / f"beta_compare_tpe_ablation_cumulative_regret_{dim}D_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


def plot_k_simple_regret(results, betas=BETAS, k_values=K_VALUES, save_fig=SAVE_FIG):
    """Simple regret vs (I-MOSS, I-MOSS-TPE, TPE k=...), one subplot per function, one
    line per beta -- I-MOSS/I-MOSS-TPE vary with beta, TPE-k doesn't."""
    functions = list(results.keys())
    fig, axes = plt.subplots(1, len(functions), figsize=(18, 5), squeeze=False)
    axes = axes[0]

    sorted_k = sorted(k_values)
    x_categories = [display_name(a) for a in TPE_ALGORITHMS] + [
        f"TPE-{k}" for k in sorted_k
    ]
    x = np.arange(len(x_categories))

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        for beta_idx, beta in enumerate(betas):
            means, stds = [], []
            for algo in TPE_ALGORITHMS:
                stat = results[fn][(algo, beta)]
                means.append(stat["simple_regret_mean"])
                stds.append(stat["simple_regret_std"])
            for k in sorted_k:
                stat = results[fn][("TPE", k)]
                means.append(stat["simple_regret_mean"])
                stds.append(stat["simple_regret_std"])
            ax.errorbar(
                x,
                means,
                yerr=stds,
                color=get_algorithm_color(beta_idx),
                marker="o",
                markersize=7,
                linewidth=2.5,
                capsize=4,
                label=f"β={beta}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_categories, rotation=45, ha="right")
        ax.set_title(fn.capitalize(), fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Simple Regret", fontweight="bold")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=len(betas), bbox_y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR / f"beta_compare_k_ablation_simple_regret_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_k_cumulative_regret(results, betas=BETAS, save_fig=SAVE_FIG):
    """Cumulative regret vs iteration at K_DIM, one subplot per function, one line
    per (algorithm, beta) pair. TPE-k baselines are beta-invariant, so they're left
    out here -- see plot_k_simple_regret for how they compare at the final budget."""
    functions = list(results.keys())
    fig, axes = plt.subplots(1, len(functions), figsize=(18, 5), squeeze=False)
    axes = axes[0]

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        for series_idx, (algo, beta) in enumerate(
            (algo, beta) for algo in TPE_ALGORITHMS for beta in betas
        ):
            cumulative = np.cumsum(results[fn][(algo, beta)]["regret_mean"])
            iterations = np.arange(1, len(cumulative) + 1)
            ax.plot(
                iterations,
                cumulative,
                color=get_algorithm_color(series_idx),
                marker=SERIES_MARKERS[series_idx % len(SERIES_MARKERS)],
                markevery=max(1, len(iterations) // 6),
                markersize=8,
                linewidth=2.0,
                label=f"{display_name(algo)} (β={beta})",
            )
        if idx == 1:
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=22)
        if idx == 0:
            ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)
        ax.set_title(f"{fn.capitalize()} ({K_DIM}D)", fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=4, bbox_y=1.16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR / f"beta_compare_k_ablation_cumulative_regret_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


def main():
    print("=== TPE-ablation beta sweep (I-MOSS vs I-MOSS-TPE) ===")
    tpe_results = sweep_tpe_ablation()
    plot_tpe_simple_regret(tpe_results)
    plot_tpe_cumulative_regret(tpe_results)

    print("\n=== k-ablation beta sweep (I-MOSS vs I-MOSS-TPE vs TPE-k) ===")
    k_results = sweep_k_ablation()
    plot_k_simple_regret(k_results)
    plot_k_cumulative_regret(k_results)


if __name__ == "__main__":
    main()
