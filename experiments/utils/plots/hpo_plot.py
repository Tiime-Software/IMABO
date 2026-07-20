"""
Plots for real HPO benchmark experiment results (experiments/hpo_experiment.py).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.hpo_experiment import BETA
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

# Apply research style
set_research_style()


def plot_performance_trajectories(
    benchmark="lr", save_fig=False, exp_type="hpo", beta=BETA
):
    """
    Plot performance trajectories showing simple regret vs cumulative regret (total regret).
    Each algorithm gets a line connecting points for different evaluation budgets.

    Args:
        benchmark: Name of the benchmark (e.g., 'lr', 'svm', 'rf')
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        beta: Beta value used for the run, for filename
    """
    # Read summary CSV
    csv_file = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_summary.csv"
    df = pd.read_csv(csv_file)

    # Check if we have raw data arrays for confidence ellipses
    has_raw_data = "simple_regrets" in df.columns and "sum_regrets" in df.columns

    # Get unique algorithms and iterations
    algorithms = df["algorithm"].unique().tolist()
    n_evals = sorted(df["n_iterations"].unique())

    # Markers for each algorithm
    base_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    markers = [base_markers[i % len(base_markers)] for i in range(len(algorithms))]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all data points and build trajectories
    all_xs, all_ys = [], []
    trajectories = {}

    for i, algorithm in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algorithm].sort_values("n_iterations")

        simple_regrets = algo_data["simple_regret_mean"].values
        total_regrets = algo_data["total_regret_mean"].values

        trajectories[algorithm] = (simple_regrets, total_regrets)
        all_xs.extend(simple_regrets)
        all_ys.extend(total_regrets)

    # Calculate median lines for quadrants
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # Draw median lines
    ax.axhline(
        y=y_mid, color=RESEARCH_COLORS["neutral"], linestyle=":", alpha=0.4, linewidth=1
    )
    ax.axvline(
        x=x_mid, color=RESEARCH_COLORS["neutral"], linestyle=":", alpha=0.4, linewidth=1
    )

    # Plot trajectories for each algorithm
    for i, algorithm in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algorithm].sort_values("n_iterations")

        xs = algo_data["simple_regret_mean"].values
        ys = algo_data["total_regret_mean"].values

        color = get_algorithm_color(i)

        # Draw confidence ellipses if raw data available
        if has_raw_data:
            for j, row in algo_data.iterrows():
                # Parse array strings to numpy arrays
                simple_regrets_raw = np.fromstring(
                    row["simple_regrets"].strip("[]"), sep=" "
                )
                sum_regrets_raw = np.fromstring(row["sum_regrets"].strip("[]"), sep=" ")
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

                    # Draw individual run points
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

        # Draw trajectory lines with increasing alpha
        for j in range(len(xs) - 1):
            alpha = 0.4 + 0.2 * (j / (len(xs) - 1))
            ax.plot(
                [xs[j], xs[j + 1]],
                [ys[j], ys[j + 1]],
                color=color,
                linewidth=3,
                alpha=alpha,
            )

        # Draw scatter points with increasing size
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
            label=display_name(algorithm),
            zorder=5,
        )

        # Add arrows showing direction of trajectory
        for j in range(len(xs) - 1):
            arrow_alpha = 0.3 + 0.2 * (j / (len(xs) - 1))
            ax.annotate(
                "",
                xy=(xs[j + 1], ys[j + 1]),  # Arrow points TO this point
                xytext=(xs[j], ys[j]),  # Arrow starts FROM this point
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=arrow_alpha,
                    lw=2,
                    shrinkA=8,
                    shrinkB=8,
                ),
                zorder=4,
            )

        # Annotate middle points
        for j, (x, y, n_eval) in enumerate(zip(xs[1:-1], ys[1:-1], n_evals[1:-1])):
            ax.annotate(
                f"{n_eval}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.7,
                ha="right",
                va="bottom",
            )

    # Labels and title
    ax.set_xlabel(
        "Simple Regret (Best Point Found)", fontsize=adaptive_label_fontsize(ax)
    )
    ax.set_ylabel(
        "Cumulative Regret (Total Regret)", fontsize=adaptive_label_fontsize(ax)
    )

    # Calculate final ranking
    final_scores = []
    for algorithm in algorithms:
        xs, ys = trajectories[algorithm]
        final_simple = xs[-1]
        final_cumulative = ys[-1]
        score = (final_simple / x_max) + (final_cumulative / y_max)
        final_scores.append((algorithm, score))

    final_scores.sort(key=lambda x: x[1])

    # Create ordered legend
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for alg, _ in final_scores:
        for handle, label in zip(handles, labels):
            if label == display_name(alg):
                ordered_handles.append(handle)
                ordered_labels.append(
                    f"{display_name(alg)} (Rank #{final_scores.index((alg, _)) + 1})"
                )
                break

    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        # loc="upper right",
        fontsize=16,
        title="Algorithm Performance\n(by final combined score)",
        title_fontsize=16,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)

    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=20)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_performance_trajectories_{exp_type}_beta_{beta}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_cumulative_regret_over_iterations(
    benchmark="lr", n_iterations=10000, save_fig=False, exp_type="hpo", beta=BETA
):
    """
    Plot cumulative regret over iterations for all algorithms at a specific iteration budget.

    Args:
        benchmark: Name of the benchmark (e.g., 'lr', 'svm', 'rf')
        n_iterations: Number of iterations to plot (default: 10000)
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        beta: Beta value used for the run, for filename
    """
    # Read iterations CSV
    csv_file = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_iterations.csv"
    df = pd.read_csv(csv_file)

    # Filter for the specific iteration count
    df = df[df["n_iterations"] == n_iterations]

    if df.empty:
        print(f"No data found for {n_iterations} iterations")
        return

    # Get unique algorithms
    algorithms = df["algorithm"].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Markers for each algorithm
    base_markers = ["o", "^", "s", "D", "v", "p"]

    for i, algo in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algo]

        iterations = algo_data["iteration"].values
        mean_regrets = algo_data["regret_mean"].values
        std_regrets = algo_data["regret_std"].values

        # Calculate cumulative mean regret
        cumulative_mean = np.cumsum(
            mean_regrets
        )  # / np.arange(1, len(mean_regrets) + 1)

        color = get_algorithm_color(i)
        marker = base_markers[i % len(base_markers)]

        # Plot line with markers
        ax.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=display_name(algo),
            marker=marker,
            markevery=len(iterations) // 8,
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax))
    ax.set_ylabel(
        "Cumulative Regret", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
    )
    ax.set_title(
        f"Cumulative Regret over {n_iterations} Iterations - {benchmark.upper()}",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(
        loc="best",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_cumulative_regret_{n_iterations}iters_{exp_type}_beta_{beta}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_simple_regret_vs_iterations(
    benchmark="lr", save_fig=False, exp_type="hpo", beta=BETA
):
    """
    Plot simple regret for all algorithms across different iteration budgets.
    Inspired by plot_simple_regret_k_experiment from ablation_plot.py

    Args:
        benchmark: Name of the benchmark (e.g., 'lr', 'svm', 'rf')
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        beta: Beta value used for the run, for filename
    """
    # Read summary CSV
    csv_file = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_summary.csv"
    df = pd.read_csv(csv_file)

    # Get unique algorithms
    algorithms = df["algorithm"].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Markers for each algorithm
    base_markers = ["o", "^", "s", "D", "v", "p"]

    for i, algo in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algo].sort_values("n_iterations")

        n_iters = algo_data["n_iterations"].values
        simple_regret_mean = algo_data["simple_regret_mean"].values
        simple_regret_std = algo_data["simple_regret_std"].values

        color = get_algorithm_color(i)
        marker = base_markers[i % len(base_markers)]

        # Plot line with error bars
        ax.errorbar(
            n_iters,
            simple_regret_mean,
            yerr=simple_regret_std,
            color=color,
            label=display_name(algo),
            marker=marker,
            markersize=8,
            linewidth=2.5,
            capsize=5,
            capthick=2,
        )

    ax.set_xlabel(
        "Number of Iterations", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
    )
    ax.set_ylabel(
        "Simple Regret", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
    )
    ax.set_title(
        f"Simple Regret vs Iteration Budget - {benchmark.upper()}",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(
        loc="best",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_simple_regret_vs_iterations_{exp_type}_beta_{beta}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_combined_regrets(
    benchmark="lr", n_iterations=10000, save_fig=False, exp_type="hpo", beta=BETA
):
    """
    Plot cumulative regret and simple regret as subplots with common legend.

    Args:
        benchmark: Name of the benchmark (e.g., 'lr', 'svm', 'rf')
        n_iterations: Number of iterations for cumulative regret plot
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        beta: Beta value used for the run, for filename
    """
    # Read both CSVs
    iterations_csv = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_iterations.csv"
    summary_csv = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_summary.csv"

    df_iterations = pd.read_csv(iterations_csv)
    df_summary = pd.read_csv(summary_csv)

    # Filter iterations data for specific iteration count
    df_iterations = df_iterations[df_iterations["n_iterations"] == n_iterations]

    if df_iterations.empty:
        print(f"No data found for {n_iterations} iterations")
        return

    # Get unique algorithms
    algorithms = df_iterations["algorithm"].unique()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Markers for each algorithm
    base_markers = ["o", "^", "s", "D", "v", "p"]

    # --- LEFT PLOT: Cumulative Regret over Iterations ---
    for i, algo in enumerate(algorithms):
        algo_data = df_iterations[df_iterations["algorithm"] == algo]

        iterations = algo_data["iteration"].values
        mean_regrets = algo_data["regret_mean"].values

        # Calculate cumulative mean regret
        cumulative_mean = np.cumsum(
            mean_regrets
        )  # / np.arange(1, len(mean_regrets) + 1)

        color = get_algorithm_color(i)
        marker = base_markers[i % len(base_markers)]

        # Plot line with markers
        ax1.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=display_name(algo),
            marker=marker,
            markevery=len(iterations) // 8,
            linewidth=2.0,
            markersize=8,
        )

    ax1.set_xlabel(
        "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax1)
    )
    ax1.set_ylabel(
        "Cumulative Regret", fontweight="bold", fontsize=adaptive_label_fontsize(ax1)
    )
    ax1.tick_params(axis="both", which="major", labelsize=20)
    ax1.set_axisbelow(True)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- RIGHT PLOT: Simple Regret vs Iterations ---
    for i, algo in enumerate(algorithms):
        algo_data = df_summary[df_summary["algorithm"] == algo].sort_values(
            "n_iterations"
        )

        n_iters = algo_data["n_iterations"].values
        simple_regret_mean = algo_data["simple_regret_mean"].values
        simple_regret_std = algo_data["simple_regret_std"].values

        color = get_algorithm_color(i)
        marker = base_markers[i % len(base_markers)]

        # Plot line with error bars
        ax2.errorbar(
            n_iters,
            simple_regret_mean,
            yerr=simple_regret_std,
            color=color,
            label=display_name(algo),
            marker=marker,
            markersize=8,
            linewidth=2.0,
            capsize=5,
            capthick=2,
        )

    ax2.set_xlabel(
        "Evaluation Budget", fontweight="bold", fontsize=adaptive_label_fontsize(ax2)
    )
    ax2.set_ylabel(
        "Simple Regret", fontweight="bold", fontsize=adaptive_label_fontsize(ax2)
    )
    ax2.set_xticks(n_iters)
    ax2.tick_params(axis="both", which="major", labelsize=20)
    ax2.set_axisbelow(True)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Create common legend
    handles, labels = ax1.get_legend_handles_labels()
    create_figure_legend(fig, handles, labels, ncol=len(algorithms))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_combined_regrets_{exp_type}_beta_{beta}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_combined_regrets_grid(
    benchmarks=("lr", "svm"),
    n_iterations=10000,
    save_fig=False,
    exp_type="hpo",
    beta=BETA,
):
    """
    Plot cumulative regret and simple regret for multiple benchmarks in a single
    wide, short grid (len(benchmarks) rows x 2 cols: Cumulative | Simple Regret)
    with one common legend. Replaces the previous per-benchmark 1x2 figures
    (plot_combined_regrets) so that e.g. LR and SVM results fit in one compact
    figure instead of two.

    Args:
        benchmarks: Iterable of benchmark names (e.g. ('lr', 'svm'))
        n_iterations: Number of iterations for the cumulative-regret column
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
        beta: Beta value used for the run, for filename
    """
    n_bench = len(benchmarks)
    fig, axes = plt.subplots(n_bench, 2, figsize=(14, 3.0 * n_bench))
    if n_bench == 1:
        axes = axes.reshape(1, 2)

    base_markers = ["o", "^", "s", "D", "v", "p"]
    legend_handles, legend_labels = None, None

    for row, benchmark in enumerate(benchmarks):
        iterations_csv = (
            RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_iterations.csv"
        )
        summary_csv = RESULTS_DIR / f"{benchmark}_{exp_type}_beta_{beta}_summary.csv"

        df_iterations = pd.read_csv(iterations_csv)
        df_summary = pd.read_csv(summary_csv)
        df_iterations = df_iterations[df_iterations["n_iterations"] == n_iterations]

        if df_iterations.empty:
            print(f"No data found for {benchmark} at {n_iterations} iterations")
            continue

        algorithms = df_iterations["algorithm"].unique()
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]

        # --- LEFT COL: Cumulative Regret over Iterations ---
        for i, algo in enumerate(algorithms):
            algo_data = df_iterations[df_iterations["algorithm"] == algo]
            iterations = algo_data["iteration"].values
            mean_regrets = algo_data["regret_mean"].values
            cumulative_mean = np.cumsum(mean_regrets)

            color = get_algorithm_color(i)
            marker = base_markers[i % len(base_markers)]

            ax_left.plot(
                iterations,
                cumulative_mean,
                color=color,
                label=display_name(algo),
                marker=marker,
                markevery=len(iterations) // 8,
                linewidth=1.8,
                markersize=6,
            )

        ax_left.set_ylabel(
            f"{benchmark.upper()}\nCumulative Regret",
            fontweight="bold",
            fontsize=20,
        )
        if row == n_bench - 1:
            ax_left.set_xlabel("Iteration", fontweight="bold", fontsize=20)
        ax_left.tick_params(axis="both", which="major", labelsize=15)
        ax_left.set_axisbelow(True)
        ax_left.grid(True, alpha=0.3)
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)

        # --- RIGHT COL: Simple Regret vs Evaluation Budget ---
        for i, algo in enumerate(algorithms):
            algo_data = df_summary[df_summary["algorithm"] == algo].sort_values(
                "n_iterations"
            )
            n_iters = algo_data["n_iterations"].values
            simple_regret_mean = algo_data["simple_regret_mean"].values
            simple_regret_std = algo_data["simple_regret_std"].values

            color = get_algorithm_color(i)
            marker = base_markers[i % len(base_markers)]

            ax_right.errorbar(
                n_iters,
                simple_regret_mean,
                yerr=simple_regret_std,
                color=color,
                label=display_name(algo),
                marker=marker,
                markersize=6,
                linewidth=1.8,
                capsize=4,
                capthick=1.5,
            )

        ax_right.set_ylabel(
            "Simple Regret",
            fontweight="bold",
            fontsize=20,
        )
        if row == n_bench - 1:
            ax_right.set_xlabel(
                "Evaluation Budget",
                fontweight="bold",
                fontsize=20,
            )
        ax_right.set_xticks(n_iters)
        ax_right.tick_params(axis="both", which="major", labelsize=15)
        ax_right.set_axisbelow(True)
        ax_right.grid(True, alpha=0.3)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax_left.get_legend_handles_labels()

    # bbox_y/rect tightened vs. the shared defaults: this figure is short
    # (3in per row) and wide, so create_figure_legend's default bbox_y=1.08
    # leaves a disproportionately large absolute gap above the top row.
    create_figure_legend(
        fig, legend_handles, legend_labels, ncol=len(legend_labels), bbox_y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.2, w_pad=4.0)

    if save_fig:
        bench_tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{bench_tag}_combined_regrets_grid_{exp_type}_beta_{beta}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    print("Generating combined regrets grid plot for LR and SVM...")
    plot_combined_regrets_grid(
        benchmarks=("lr", "svm"), n_iterations=10000, save_fig=True, exp_type="hpo"
    )
