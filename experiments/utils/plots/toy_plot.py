"""
Plots for toy benchmark experiment results (experiments/toy_experiment.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    confidence_ellipse,
    create_figure_legend,
    display_name,
    get_algorithm_color,
    save_figure,
    set_research_style,
    RESEARCH_COLORS,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"

set_research_style()


def plot_multiple_trajectories(benchmarks, save_fig=False, exp_type="toy"):
    """
    Plot performance trajectories with confidence ellipses for multiple toy benchmarks.

    Args:
        benchmarks: List of benchmark names (e.g., ['sin1', 'garland', 'rastrigin'])
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
    """
    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(10 * n_benchmarks, 8))

    if n_benchmarks == 1:
        axes = [axes]

    for ax_idx, benchmark in enumerate(benchmarks):
        ax = axes[ax_idx]

        csv_file = RESULTS_DIR / f"{benchmark}_{exp_type}_summary.csv"
        df = pd.read_csv(csv_file)

        has_raw_data = "simple_regrets" in df.columns and "sum_regrets" in df.columns

        algorithms = df["algorithm"].unique().tolist()

        base_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
        markers = [base_markers[i % len(base_markers)] for i in range(len(algorithms))]

        all_xs, all_ys = [], []
        trajectories = {}

        for i, algorithm in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algorithm].sort_values("n_iterations")
            simple_regrets = algo_data["simple_regret_mean"].values
            total_regrets = algo_data["total_regret_mean"].values
            trajectories[algorithm] = (simple_regrets, total_regrets)
            all_xs.extend(simple_regrets)
            all_ys.extend(total_regrets)

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

        for i, algorithm in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algorithm].sort_values("n_iterations")
            xs = algo_data["simple_regret_mean"].values
            ys = algo_data["total_regret_mean"].values
            color = get_algorithm_color(i)

            if has_raw_data:
                for j, row in algo_data.iterrows():
                    simple_regrets_raw = np.fromstring(
                        row["simple_regrets"].strip("[]"), sep=" "
                    )
                    sum_regrets_raw = np.fromstring(
                        row["sum_regrets"].strip("[]"), sep=" "
                    )
                    if len(simple_regrets_raw) > 2 and len(sum_regrets_raw) > 2:
                        confidence_ellipse(
                            simple_regrets_raw,
                            sum_regrets_raw,
                            ax,
                            n_std=1.0,
                            alpha=0.15,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=1.5,
                            zorder=2,
                        )
                        ax.scatter(
                            simple_regrets_raw,
                            sum_regrets_raw,
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
                marker=markers[i],
                alpha=0.8,
                edgecolors="white",
                linewidths=2,
                label=display_name(algorithm) if ax_idx == 0 else "",
                zorder=5,
            )

        label_fs = adaptive_label_fontsize(ax)
        # Only show xlabel on the middle subplot and ylabel on the left
        # subplot, matching ablation_plot.py's convention (one shared label
        # per axis instead of repeating it under/beside every panel).
        if ax_idx == n_benchmarks // 2:
            ax.set_xlabel("Simple Regret", fontweight="bold", fontsize=35)
        if ax_idx == 0:
            ax.set_ylabel(
                "Normalized Cumulative Regret", fontweight="bold", fontsize=30
            )
        ax.set_title(benchmark.upper(), fontweight="bold", fontsize=24, pad=15)
        ax.tick_params(axis="both", which="major", labelsize=30)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[i],
            color=get_algorithm_color(i),
            markerfacecolor=get_algorithm_color(i),
            markeredgecolor="white",
            markeredgewidth=1.5,
            markersize=16,
            linestyle="",
        )
        for i in range(len(algorithms))
    ]
    legend_labels = [display_name(algorithm) for algorithm in algorithms]
    create_figure_legend(
        fig, legend_handles, legend_labels, ncol=len(algorithms), bbox_y=1.05
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        filename = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{filename}_trajectories_comparison_{exp_type}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    benchmarks = ["sin1", "garland", "rastrigin"]
    exp_type = "toy"
    print(f"Generating multiple trajectories comparison for {benchmarks}...")
    plot_multiple_trajectories(benchmarks=benchmarks, save_fig=True, exp_type=exp_type)
