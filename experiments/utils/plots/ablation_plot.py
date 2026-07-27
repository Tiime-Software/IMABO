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
    paper_style,
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


_K_ABLATION_MARKERS = ["o", "^", "s", "D", "v", "p", "<"]

_APPENDIX_PANEL_W_IN = 4.2  # per-function panel width
_APPENDIX_PANEL_H_IN = 3.4  # per-panel height (before legend headroom)


def _appendix_style(columns: int = 1, max_legend_single_row: int = 3):
    return paper_style(
        "aaai",
        columns=columns,
        markevery_divisor=6,
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


def plot_cumulative_regrets_k_experiment(save_fig=False, beta=0.5, columns=1):
    """
    Cumulative regret over iterations for I-MOSS-TPE vs TPE-k variants, one
    panel per function (sin1, garland, rastrigin) side by side sharing one
    top legend -- appendix figure, styled like
    delayed_feedback_plot.plot_cumulative_regret_grid: paper_style
    CONVENTION (colors, spines, shared top legend) at larger, appendix-
    readable font/line sizes rather than the tiny main-paper print pt.
    """
    functions = ["sin1", "garland", "rastrigin"]

    dfs = {
        func: pd.read_csv(
            RESULTS_DIR / f"k_ablation_{func}_{K_N_ITER}_beta_{beta}_iterations.csv"
        )
        for func in functions
    }
    algorithms = _sort_k_ablation_algorithms(dfs[functions[0]]["algorithm"].unique())

    # Force the legend onto one row -- at this panel width (n * 4.2in) even
    # 7 entries fit comfortably on one line.
    style = _appendix_style(columns, max_legend_single_row=len(algorithms))

    n = len(functions)
    n_rows = style.n_legend_rows(len(algorithms))
    fig, axes = plt.subplots(1, n, figsize=_appendix_figsize(n, n_rows))

    seen: dict = {}
    for idx, func in enumerate(functions):
        ax = axes[idx]
        df = dfs[func]

        for algo_idx, algo in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algo]

            iterations = algo_data["iteration"].values
            mean_regrets = algo_data["regret_mean"].values
            cumulative_mean = np.cumsum(mean_regrets)

            label = _format_k_ablation_label(algo)
            (line,) = ax.plot(
                iterations,
                cumulative_mean,
                color=get_algorithm_color(algo_idx),
                label=label,
                marker=_K_ABLATION_MARKERS[algo_idx % len(_K_ABLATION_MARKERS)],
                markevery=style.markevery(len(iterations)),
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
            seen.setdefault(label, line)

        if idx == n // 2:  # middle panel
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            func.capitalize(), fontweight="bold", fontsize=style.title_fontsize, pad=4
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Cumulative Regret", fontweight="bold", fontsize=style.label_fontsize
    )

    ordered_labels = [_format_k_ablation_label(a) for a in algorithms]
    style.legend(
        fig,
        [seen[label] for label in ordered_labels],
        ordered_labels,
        n_labels=len(algorithms),
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        save_figure(
            RESULTS_DIR
            / "paper_plots"
            / f"k_experiment_cumulative_regrets_beta_{beta}.pdf",
            bbox_inches="tight",
            parents=True,
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
    plot_cumulative_regrets_k_experiment(save_fig=save_fig, beta=beta, columns=1)
