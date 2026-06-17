import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from experiments.utils.plot_configs import (
    set_research_style,
    get_algorithm_color,
    RESEARCH_COLORS,
)

RESULTS_DIR = Path(__file__).parents[2] / "results"

set_research_style()


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = np.corrcoef(x, y)[0, 1]
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_multiple_trajectories(benchmarks, save_fig=False, exp_type="toy"):
    """
    Plot performance trajectories with confidence ellipses for multiple toy benchmarks.

    Args:
        benchmarks: List of benchmark names (e.g., ['sin1', 'garland', 'rastrigin'])
        save_fig: Whether to save the figure
        exp_type: Experiment type for filename
    """
    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(8 * n_benchmarks, 8))

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

        ax.axhline(y=y_mid, color=RESEARCH_COLORS["neutral"], linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(x=x_mid, color=RESEARCH_COLORS["neutral"], linestyle=":", alpha=0.4, linewidth=1)

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
                            simple_regrets_raw, sum_regrets_raw, ax,
                            n_std=1.0, alpha=0.15, facecolor=color,
                            edgecolor=color, linewidth=1.5, zorder=2,
                        )
                        ax.scatter(
                            simple_regrets_raw, sum_regrets_raw,
                            c=color, alpha=0.25, s=15,
                            edgecolors="white", linewidths=0.5, zorder=3,
                        )

            for j in range(len(xs) - 1):
                alpha = 0.4 + 0.2 * (j / (len(xs) - 1))
                ax.plot(
                    [xs[j], xs[j + 1]], [ys[j], ys[j + 1]],
                    color=color, linewidth=3, alpha=alpha,
                )

            sizes = np.linspace(60, 120, len(xs))
            ax.scatter(
                xs, ys, c=color, s=sizes, marker=markers[i],
                alpha=0.8, edgecolors="white", linewidths=2,
                label=algorithm if ax_idx == 0 else "",
                zorder=5,
            )

        ax.set_xlabel("Simple Regret", fontweight="bold", fontsize=22)
        if ax_idx == 0:
            ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)
        ax.set_title(benchmark.upper(), fontweight="bold", fontsize=24, pad=15)
        ax.tick_params(axis="both", which="major", labelsize=30)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(algorithms),
        frameon=False,
        prop={"size": 30, "family": "serif", "weight": "bold"},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        filename = "_".join(benchmarks)
        plt.savefig(
            output_dir / f"{filename}_trajectories_comparison_{exp_type}.pdf",
            bbox_inches="tight",
        )
        print(f"Saved to {output_dir / f'{filename}_trajectories_comparison_{exp_type}.pdf'}")

    plt.show()


if __name__ == "__main__":
    benchmarks = ["sin1", "garland", "rastrigin"]
    exp_type = "toy"
    print(f"Generating multiple trajectories comparison for {benchmarks}...")
    plot_multiple_trajectories(benchmarks=benchmarks, save_fig=True, exp_type=exp_type)
