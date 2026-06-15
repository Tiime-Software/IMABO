import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils.plot_functions import (
    ALGORITHM_COLORS,
)


RESULTS_DIR = Path(__file__).parents[2] / "results"


def _format_k_ablation_label(algo, k=None):
    if str(algo).upper() == "IMABO" or (k is not None and pd.isna(k)):
        return "IMABO"
    if k is not None and not pd.isna(k):
        return f"Optuna {int(k)}"
    if "k=" in str(algo):
        return f"Optuna {int(str(algo).split('k=')[1])}"
    return str(algo)


def _sort_k_ablation_algorithms(algorithms):
    def sort_key(algo):
        if str(algo).upper() == "IMABO":
            return (-1, 0)
        if "k=" in str(algo):
            return (0, int(str(algo).split("k=")[1]))
        return (1, 0)

    return sorted(algorithms, key=sort_key)


def plot_cumulative_regrets_k_experiment(save_fig=False):
    """
    Plot cumulative regrets for different k values across sin1, guirland, and quadratic functions.
    Each function is a subplot with curves for IMABO and Optuna with different k values.
    """
    functions = ["sin1", "garland", "rastrigin"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, func in enumerate(functions):
        ax = axes[idx]

        # Read iterations CSV
        csv_file = RESULTS_DIR / f"k_ablation_{func}_3000_iterations.csv"
        df = pd.read_csv(csv_file)

        algorithms = _sort_k_ablation_algorithms(df["algorithm"].unique())

        for algo_idx, algo in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algo]

            iterations = algo_data["iteration"].values
            mean_regrets = algo_data["regret_mean"].values
            std_regrets = algo_data["regret_std"].values

            # Calculate cumulative mean regret
            # cumulative_mean = np.cumsum(mean_regrets)
            cumulative_mean = np.cumsum(mean_regrets) / np.arange(
                1, len(mean_regrets) + 1
            )

            color = ALGORITHM_COLORS[algo_idx % len(ALGORITHM_COLORS)]
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
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=22)
        if idx == 0:  # Left subplot
            ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)

        ax.set_title(f"{func.capitalize()}", fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

    # Create common legend at top center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=6,
        # fontsize=25,
        frameon=False,
        prop={"size": 22, "family": "serif", "weight": "bold"},
    )

    # plt.suptitle(
    #     "Cumulative Regret Comparison Across Different k Values",
    #     fontsize=20,
    #     fontweight="bold",
    #     y=1.08,
    # )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        plt.savefig(
            RESULTS_DIR / "k_experiment_cumulative_regrets.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_simple_regret_k_experiment(save_fig=False):
    """
    Plot simple regret for different k values across sin1, guirland, and quadratic functions.
    Each function is a subplot showing final simple regret with error bars as line plot.
    """
    functions = ["sin1", "garland", "rastrigin"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, func in enumerate(functions):
        ax = axes[idx]

        # Read summary CSV
        csv_file = RESULTS_DIR / f"k_ablation_{func}_summary.csv"
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
            color=ALGORITHM_COLORS[0],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=22)
        if idx == 0:
            ax.set_ylabel("Simple Regret", fontweight="bold", fontsize=22)
        # ax.set_xlabel("Setting", fontweight="bold", fontsize=18)
        ax.set_title(f"{func.capitalize()}", fontweight="bold", fontsize=22)
        ax.tick_params(axis="y", which="major", labelsize=22)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        # ax.set_yscale("log")

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            RESULTS_DIR / "k_ablation_simple_regrets.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    # Generate both plots
    print("Generating cumulative regrets plot...")
    plot_cumulative_regrets_k_experiment(save_fig=True)

    print("Generating simple regret plot...")
    plot_simple_regret_k_experiment(save_fig=True)

    print("All plots generated successfully!")
