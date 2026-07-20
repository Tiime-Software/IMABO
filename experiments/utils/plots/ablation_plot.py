"""
Plots for ablation study results (experiments/ablation_experiment.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    create_figure_legend,
    display_name,
    get_algorithm_color,
    save_figure,
)

from experiments.ablation_experiment import K_N_ITER, BETA


RESULTS_DIR = Path(__file__).parents[3] / "results" / "ablation_experiment"


def _format_k_ablation_label(algo, k=None):
    if str(algo).upper() == "I-MOSS-TPE" or (k is not None and pd.isna(k)):
        return display_name("I-MOSS-TPE")
    if k is not None and not pd.isna(k):
        return f"TPE-{int(k)}"
    if "k=" in str(algo):
        return f"TPE-{int(str(algo).split('k=')[1])}"
    return str(algo)


def _sort_k_ablation_algorithms(algorithms):
    def sort_key(algo):
        if str(algo).upper() == "I-MOSS-TPE":
            return (-1, 0)
        if "k=" in str(algo):
            return (0, int(str(algo).split("k=")[1]))
        return (1, 0)

    return sorted(algorithms, key=sort_key)


def plot_regret_vs_dimension_tpe_ablation(save_fig=False, beta=0.8):
    """
    Plot average regret (mean regret per iteration, averaged over runs) as a
    function of search-space dimension, for I-MOSS vs I-MOSS-TPE.
    Each function (sin1, garland, rastrigin) is a subplot.
    """
    functions = ["sin1", "garland", "rastrigin"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, func in enumerate(functions):
        ax = axes[idx]

        csv_file = RESULTS_DIR / f"{func}_tpe_ablation_beta_{beta}_summary.csv"
        df = pd.read_csv(csv_file)

        algorithms = sorted(df["algorithm"].unique())

        for algo_idx, algo in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algo].sort_values("dimension")

            dims = algo_data["dimension"].values
            mean_regrets = algo_data["total_regret_mean"].values
            std_regrets = algo_data["total_regret_std"].values

            ax.errorbar(
                dims,
                mean_regrets,
                yerr=std_regrets,
                color=get_algorithm_color(algo_idx),
                label=display_name(algo),
                marker="o",
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
            )

        ax.set_xticks(sorted(df["dimension"].unique()))
        if idx == 1:  # Middle subplot
            ax.set_xlabel(
                "Dimension", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        if idx == 0:  # Left subplot
            ax.set_ylabel(
                "Average Regret",
                fontweight="bold",
                fontsize=adaptive_label_fontsize(ax),
            )

        ax.set_title(f"{func.capitalize()}", fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=2, bbox_y=1.10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        save_figure(
            RESULTS_DIR / f"tpe_ablation_regret_vs_dimension_beta_{beta}.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


def plot_cumulative_regrets_k_experiment(save_fig=False, beta=0.5):
    """
    Plot cumulative regrets for different k values across sin1, guirland, and quadratic functions.
    Each function is a subplot with curves for I-MOSS-TPE and Optuna with different k values.
    """
    functions = ["sin1", "garland", "rastrigin"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, func in enumerate(functions):
        ax = axes[idx]

        # Read iterations CSV
        csv_file = (
            RESULTS_DIR / f"k_ablation_{func}_{K_N_ITER}_beta_{beta}_iterations.csv"
        )
        df = pd.read_csv(csv_file)

        algorithms = _sort_k_ablation_algorithms(df["algorithm"].unique())

        for algo_idx, algo in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algo]

            iterations = algo_data["iteration"].values
            mean_regrets = algo_data["regret_mean"].values
            std_regrets = algo_data["regret_std"].values

            # cumulative_mean = np.cumsum(mean_regrets) / np.arange(
            #     1, len(mean_regrets) + 1
            # )
            cumulative_mean = np.cumsum(mean_regrets)

            color = get_algorithm_color(algo_idx)
            label = _format_k_ablation_label(algo)

            # Plot line with markers
            # Use different markers for each algorithm
            markers = ["o", "^", "s", "D", "v", "p"]
            marker_idx = algo_idx
            ax.plot(
                iterations,
                cumulative_mean,
                color=color,
                label=label,
                marker=markers[marker_idx % len(markers)],
                markevery=len(iterations) // 6,
                linewidth=2.0,
                markersize=8,
            )

        # Only show xlabel on middle subplot and ylabel on left subplot
        if idx == 1:  # Middle subplot
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        if idx == 0:  # Left subplot
            ax.set_ylabel(
                "Cumulative Regret",
                fontweight="bold",
                fontsize=adaptive_label_fontsize(ax),
            )

        ax.set_title(f"{func.capitalize()}", fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    # Create common legend at top center
    handles, labels = axes[0].get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=6, bbox_y=1.10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        save_figure(
            RESULTS_DIR
            / "paper_plots"
            / f"k_experiment_cumulative_regrets_beta_{beta}.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


def plot_simple_regret_k_experiment(save_fig=False, beta=0.5):
    """
    Plot simple regret for different k values across sin1, guirland, and quadratic functions.
    Each function is a subplot showing final simple regret with error bars as line plot.
    """
    functions = ["sin1", "garland", "rastrigin"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, func in enumerate(functions):
        ax = axes[idx]

        # Read summary CSV
        csv_file = RESULTS_DIR / f"k_ablation_{func}_beta_{beta}_summary.csv"
        df = pd.read_csv(csv_file)

        # Get algorithms and their data
        algorithms = df["algorithm"].values
        k_values = df["k"].values
        simple_regret_mean = df["simple_regret_mean"].values
        simple_regret_std = df["simple_regret_std"].values

        # Create x-axis positions
        x_positions = np.arange(len(algorithms))

        x_labels = [
            _format_k_ablation_label(algo, k) for algo, k in zip(algorithms, k_values)
        ]

        # Plot line with error bars
        ax.errorbar(
            x_positions,
            simple_regret_mean,
            yerr=simple_regret_std,
            color=get_algorithm_color(0),
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=22)
        if idx == 0:
            ax.set_ylabel(
                "Simple Regret", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(f"{func.capitalize()}", fontweight="bold", fontsize=22)
        ax.tick_params(axis="y", which="major", labelsize=22)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        save_figure(
            RESULTS_DIR / "k_ablation_simple_regrets.pdf",
            dpi=300,
            bbox_inches="tight",
            mkdir=False,
            verbose=False,
        )
    plt.show()


if __name__ == "__main__":
    # Generate all plots
    save_fig = True
    beta = BETA
    print("Generating cumulative regrets plot...")
    plot_cumulative_regrets_k_experiment(save_fig=save_fig, beta=beta)

    # print("Generating simple regret plot...")
    # plot_simple_regret_k_experiment(save_fig=save_fig, beta=beta)

    # print("Generating regret vs dimension (TPE ablation) plot...")
    # plot_regret_vs_dimension_tpe_ablation(save_fig=save_fig, beta=beta)

    print("All plots generated successfully!")
