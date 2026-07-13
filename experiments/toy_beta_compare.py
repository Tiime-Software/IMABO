"""
Sweep beta for experiments.toy_experiment's IMABO run, the same way
experiments.test_exp.ablation_beta_compare sweeps beta for
experiments.ablation_experiment: parallel runs via joblib, printed summary
table, plus a beta-comparison plot built straight from the in-memory results
(no CSV round-trip).

toy_experiment.py itself sweeps iteration budget (not dimension) at a fixed
dim=4, so that's the axis swept here too.

Usage (from repo root):
    python -m experiments.test_exp.toy_beta_compare
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.utils.plots.plot_configs import (
    RESEARCH_COLORS,
    confidence_ellipse,
    create_figure_legend,
    get_algorithm_color,
    save_figure,
)
from imabo import IMABO

RESULT_DIR = Path(__file__).parents[2] / "results"

# ── Fixed run config ──────────────────────────────────────────────────────────
BETAS = [0.5, 0.6, 0.7, 0.8]
FUNCTIONS = ["sin1", "garland", "rastrigin"]
DIM = 4
N_ITERS = [1000, 3000, 5000, 10000]
PLOT_N_ITER = 3000  # budget used for the cumulative-regret plot
N_RUNS = 20
BASE_SEED = 42
N_JOBS = 8
SAVE_FIG = True


def run_imabo(function_name, dim, n_iterations, seed, beta):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)
    func_noiseless = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    search_space = obj.get_search_space(function_name)

    opt = IMABO(
        search_space=search_space,
        seed=seed,
        multivariate=True,
        beta=beta,
        use_tpe=False,
    )

    regrets = []
    for _ in range(n_iterations):
        x = opt.suggest()
        y = func(x)
        regrets.append(fmax - func_noiseless(x) / dim)
        opt.observe(y)

    simple_regret = fmax - func_noiseless(opt.best_x) / dim
    return {"regrets": regrets, "simple_regrets": simple_regret}


def one_run(fn, dim, n_iter, seed, betas):
    return {beta: run_imabo(fn, dim, n_iter, seed, beta) for beta in betas}


def sweep_toy_beta(
    functions=FUNCTIONS,
    dim=DIM,
    n_iters=N_ITERS,
    n_runs=N_RUNS,
    betas=BETAS,
    base_seed=BASE_SEED,
):
    """IMABO across beta, one printed table per (function, iteration budget).

    Returns {function: {n_iter: {beta: {"simple_regret_mean", "simple_regret_std",
    "regret_mean" (per-iteration array), "simple_regrets_raw", "sum_regrets_raw"
    (per-run arrays, for plot_trajectories' confidence ellipses -- "sum_regrets_raw"
    is normalized by n_iter, matching calculate_statistics/toy_plot.py's convention,
    so it sits on the same 0-1 scale as simple regret)}}}} for the plot_* functions
    below.
    """
    results = {}
    for fn in functions:
        results[fn] = {}
        for n_iter in n_iters:
            runs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000, betas)
                for r in range(n_runs)
            )
            print(f"\n[{fn} {n_iter} iters]")
            iter_result = {}
            for beta in betas:
                simple = np.array([r[beta]["simple_regrets"] for r in runs])
                regrets = np.array([r[beta]["regrets"] for r in runs])
                iter_result[beta] = {
                    "simple_regret_mean": simple.mean(),
                    "simple_regret_std": simple.std(),
                    "regret_mean": regrets.mean(axis=0),
                    "simple_regrets_raw": simple,
                    "sum_regrets_raw": regrets.sum(axis=1) / regrets.shape[1],
                }
                print(
                    f"   β={beta:<4} simple={simple.mean():.4f}  "
                    f"cumreg={regrets.sum(axis=1).mean():8.1f}"
                )
            results[fn][n_iter] = iter_result
    return results


def plot_simple_regret(results, betas=BETAS, save_fig=SAVE_FIG):
    """Simple regret vs iteration budget, one subplot per function, one line per beta."""
    functions = list(results.keys())
    fig, axes = plt.subplots(
        1, len(functions), figsize=(6 * len(functions), 5), squeeze=False
    )
    axes = axes[0]

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        n_iters = sorted(results[fn].keys())
        for beta_idx, beta in enumerate(betas):
            means = [results[fn][n][beta]["simple_regret_mean"] for n in n_iters]
            stds = [results[fn][n][beta]["simple_regret_std"] for n in n_iters]
            ax.errorbar(
                n_iters,
                means,
                yerr=stds,
                color=get_algorithm_color(beta_idx),
                marker="o",
                markersize=7,
                linewidth=2.5,
                capsize=4,
                label=f"β={beta}",
            )
        ax.set_xlabel("Iterations")
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
            RESULT_DIR / f"toy_beta_compare_simple_regret_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_cumulative_regret(results, betas=BETAS, n_iter=PLOT_N_ITER, save_fig=SAVE_FIG):
    """Cumulative regret vs iteration at a fixed budget, one subplot per function, one
    line per beta."""
    functions = list(results.keys())
    fig, axes = plt.subplots(
        1, len(functions), figsize=(6 * len(functions), 5), squeeze=False
    )
    axes = axes[0]

    for idx, fn in enumerate(functions):
        ax = axes[idx]
        for beta_idx, beta in enumerate(betas):
            cumulative = np.cumsum(results[fn][n_iter][beta]["regret_mean"])
            iterations = np.arange(1, len(cumulative) + 1)
            ax.plot(
                iterations,
                cumulative,
                color=get_algorithm_color(beta_idx),
                marker="o",
                markevery=max(1, len(iterations) // 6),
                markersize=7,
                linewidth=2.5,
                label=f"β={beta}",
            )
        ax.set_xlabel("Iteration")
        ax.set_title(f"{fn.capitalize()} ({n_iter} iters)", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Cumulative Regret", fontweight="bold")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=len(betas), bbox_y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR / f"toy_beta_compare_cumulative_regret_{n_iter}_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_trajectories(results, betas=BETAS, save_fig=SAVE_FIG):
    """Simple regret vs cumulative regret trajectory across iteration budgets, one
    point per (beta, budget) with a confidence ellipse over the raw per-run scatter --
    mirrors experiments/utils/plots/toy_plot.py's plot_multiple_trajectories, with
    beta standing in for "algorithm"."""
    functions = list(results.keys())
    fig, axes = plt.subplots(
        1, len(functions), figsize=(8 * len(functions), 8), squeeze=False
    )
    axes = axes[0]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for ax_idx, fn in enumerate(functions):
        ax = axes[ax_idx]
        n_iters = sorted(results[fn].keys())

        trajectories = {}
        all_xs, all_ys = [], []
        for beta in betas:
            xs = [results[fn][n][beta]["simple_regret_mean"] for n in n_iters]
            ys = [results[fn][n][beta]["sum_regrets_raw"].mean() for n in n_iters]
            trajectories[beta] = (xs, ys)
            all_xs.extend(xs)
            all_ys.extend(ys)

        x_mid = (min(all_xs) + max(all_xs)) / 2
        y_mid = (min(all_ys) + max(all_ys)) / 2
        ax.axhline(
            y=y_mid,
            color=RESEARCH_COLORS["neutral"],
            linestyle=":",
            alpha=0.4,
            linewidth=1,
        )
        ax.axvline(
            x=x_mid,
            color=RESEARCH_COLORS["neutral"],
            linestyle=":",
            alpha=0.4,
            linewidth=1,
        )

        for beta_idx, beta in enumerate(betas):
            color = get_algorithm_color(beta_idx)
            xs, ys = trajectories[beta]

            for n in n_iters:
                simple_raw = results[fn][n][beta]["simple_regrets_raw"]
                sum_raw = results[fn][n][beta]["sum_regrets_raw"]
                if len(simple_raw) > 2:
                    confidence_ellipse(
                        simple_raw,
                        sum_raw,
                        ax,
                        n_std=1.0,
                        alpha=0.15,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1.5,
                        zorder=2,
                    )
                    ax.scatter(
                        simple_raw,
                        sum_raw,
                        c=color,
                        alpha=0.25,
                        s=15,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=3,
                    )

            for j in range(len(xs) - 1):
                alpha = 0.4 + 0.2 * (j / (len(xs) - 1))
                ax.plot(
                    [xs[j], xs[j + 1]],
                    [ys[j], ys[j + 1]],
                    color=color,
                    linewidth=3,
                    alpha=alpha,
                )

            sizes = np.linspace(60, 120, len(xs))
            ax.scatter(
                xs,
                ys,
                c=color,
                s=sizes,
                marker=markers[beta_idx % len(markers)],
                alpha=0.8,
                edgecolors="white",
                linewidths=2,
                label=f"β={beta}" if ax_idx == 0 else "",
                zorder=5,
            )

        ax.set_xlabel("Simple Regret", fontweight="bold", fontsize=22)
        if ax_idx == 0:
            ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)
        ax.set_title(fn.capitalize(), fontweight="bold", fontsize=24, pad=15)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(
        fig, handles, labels, ncol=len(betas), bbox_y=1.05, fontsize=20
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if save_fig:
        suffix = "_vs_".join(str(b) for b in betas)
        save_figure(
            RESULT_DIR / f"toy_beta_compare_trajectories_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def main():
    print("=== toy_experiment beta sweep (IMABO) ===")
    results = sweep_toy_beta()
    plot_simple_regret(results)
    plot_cumulative_regret(results)
    plot_trajectories(results)


if __name__ == "__main__":
    main()
