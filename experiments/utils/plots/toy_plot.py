"""
Plots for toy benchmark experiment results (experiments/toy_experiment.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.utils.plots.plot_configs import (
    RESEARCH_COLORS,
    confidence_ellipse,
    display_name,
    get_algorithm_color,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"


set_research_style()

# This figure lives in the APPENDIX, where we don't need exact main-paper
# print size -- so, like ablation_plot.py / delayed_feedback_plot.py's
# _appendix_style / _appendix_figsize, it follows the paper_style CONVENTION
# (Wong colors, shared top legend via create_figure_legend, style_axis look)
# but overrides paper_style's tiny print fonts/lines (~6.5-9pt, sized for
# shrink-to-column at final print) with larger, readable values.
_APPENDIX_PANEL_W_IN = 4.2  # per-benchmark panel width
_APPENDIX_PANEL_H_IN = 3.4  # per-panel height (before legend headroom)


def _appendix_style(columns: int = 1, max_legend_single_row: int = 3):
    return paper_style(
        "aaai",
        columns=columns,
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        legend_fontsize=16,
        linewidth=2.2,
        markersize=8,
        max_legend_single_row=max_legend_single_row,
    )


def _appendix_figsize(n_panels: int, n_legend_rows: int = 1):
    return (
        _APPENDIX_PANEL_W_IN * n_panels,
        _APPENDIX_PANEL_H_IN + 0.55 * n_legend_rows,
    )


def _appendix_legend_rect_top(n_legend_rows: int) -> float:
    fig_h = _APPENDIX_PANEL_H_IN + 0.55 * n_legend_rows
    return 1.0 - (0.55 * n_legend_rows) / fig_h


def plot_multiple_trajectories(benchmarks, save_fig=False, exp_type="toy", columns=1):
    """
    Plot performance trajectories with confidence ellipses for multiple toy
    benchmarks -- appendix figure, styled like
    delayed_feedback_plot.plot_cumulative_regret_grid /
    ablation_plot.plot_cumulative_regrets_k_experiment: paper_style
    CONVENTION (colors, spines, shared top legend) at larger, appendix-
    readable font/line sizes rather than the tiny main-paper print pt.

    Args:
        benchmarks: List of benchmark names (e.g., ['sin1', 'garland', 'rastrigin'])
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        columns: 1 (single text column) or 2 (full text width / AAAI
            `figure*`) -- see plot_configs.paper_figure_width_in.
    """
    n_benchmarks = len(benchmarks)

    dfs = {
        benchmark: pd.read_csv(RESULTS_DIR / f"{benchmark}_{exp_type}_summary.csv")
        for benchmark in benchmarks
    }
    algorithms = dfs[benchmarks[0]]["algorithm"].unique().tolist()
    base_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    markers = [base_markers[i % len(base_markers)] for i in range(len(algorithms))]

    # Force the legend onto one row -- at this panel width (n * 4.2in) even
    # several entries fit comfortably on one line.
    style = _appendix_style(columns, max_legend_single_row=len(algorithms))
    n_rows = style.n_legend_rows(len(algorithms))

    fig, axes = plt.subplots(
        1, n_benchmarks, figsize=_appendix_figsize(n_benchmarks, n_rows)
    )
    if n_benchmarks == 1:
        axes = [axes]

    for ax_idx, benchmark in enumerate(benchmarks):
        ax = axes[ax_idx]
        df = dfs[benchmark]

        has_raw_data = "simple_regrets" in df.columns and "sum_regrets" in df.columns

        all_xs, all_ys = [], []
        trajectories = {}

        for algorithm in algorithms:
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
                            linewidth=1.2,
                            zorder=2,
                        )
                        ax.scatter(
                            simple_regrets_raw,
                            sum_regrets_raw,
                            c=color,
                            alpha=0.25,
                            s=10,
                            edgecolors="white",
                            linewidths=0.4,
                            zorder=3,
                        )

            for j in range(len(xs) - 1):
                alpha = 0.4 + 0.2 * (j / (len(xs) - 1))
                ax.plot(
                    [xs[j], xs[j + 1]],
                    [ys[j], ys[j + 1]],
                    color=color,
                    linewidth=style.linewidth,
                    alpha=alpha,
                )

            sizes = np.linspace(30, 60, len(xs))
            ax.scatter(
                xs,
                ys,
                c=color,
                s=sizes,
                marker=markers[i],
                alpha=0.8,
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
            )

        # Only show xlabel on the middle subplot and ylabel on the left
        # subplot, matching ablation_plot.py's convention (one shared label
        # per axis instead of repeating it under/beside every panel).
        if ax_idx == n_benchmarks // 2:
            ax.set_xlabel(
                "Simple Regret", fontweight="bold", fontsize=style.label_fontsize
            )
        if ax_idx == 0:
            ax.set_ylabel(
                "Online Avg. Regret",
                fontweight="bold",
                fontsize=style.label_fontsize,
            )
        ax.set_title(
            benchmark.capitalize(),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[i],
            color=get_algorithm_color(i),
            markerfacecolor=get_algorithm_color(i),
            markeredgecolor="white",
            markeredgewidth=1.2,
            markersize=style.markersize,
            linestyle="",
        )
        for i in range(len(algorithms))
    ]
    legend_labels = [display_name(algorithm) for algorithm in algorithms]
    style.legend(fig, legend_handles, legend_labels, n_labels=len(algorithms))

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        filename = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{filename}_trajectories_comparison_{exp_type}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    benchmarks = ["sin1", "garland", "rastrigin"]
    exp_type = "toy"
    print(f"Generating multiple trajectories comparison for {benchmarks}...")
    plot_multiple_trajectories(benchmarks=benchmarks, save_fig=True, exp_type=exp_type)
