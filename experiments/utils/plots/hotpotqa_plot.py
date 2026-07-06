"""Plot functions for HotpotQA multi-run experiment results."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from experiments.utils.plots.plot_configs import set_research_style, get_algorithm_color

RESULTS_DIR = Path(__file__).parents[3] / "results" / "hotpotqa"

set_research_style()


def plot_hotpotqa_results(
    algorithms: list[str],
    n_samples: int,
    n_runs: int,
    save_fig: bool = False,
) -> None:
    """
    Side-by-side plot: cumulative regret over iterations (left) and
    simple regret bar chart (right), one series per algorithm.

    Args:
        algorithms: Algorithm names to compare, e.g. ["IMABO", "Random"]
        n_samples: Number of optimization steps used in the experiment
        n_runs: Number of runs used in the experiment
        save_fig: Whether to save the figure as PDF
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    base_markers = ["o", "^", "s", "D", "v", "P", "X", "*"]
    rng = np.random.default_rng(0)

    x_pos = np.arange(len(algorithms))

    for i, algo in enumerate(algorithms):
        stem = f"{algo}_hotpotqa_{n_samples}samples_{n_runs}runs"
        iter_csv = RESULTS_DIR / f"{stem}_iterations.csv"
        summary_csv = RESULTS_DIR / f"{stem}_summary.csv"

        df_iter = pd.read_csv(iter_csv)
        df_summary = pd.read_csv(summary_csv)

        # --- LEFT: cumulative regret over iterations ---
        iterations = df_iter["iteration"].values
        mean_regrets = df_iter["regret_mean"].values
        std_regrets = df_iter["regret_std"].values

        cumulative_mean = np.cumsum(mean_regrets) / np.arange(1, len(mean_regrets) + 1)
        # cumulative_mean = np.cumsum(mean_regrets)
        cumulative_std = np.sqrt(np.cumsum(std_regrets**2)) / iterations

        color = get_algorithm_color(i)
        ax1.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=algo,
            marker=base_markers[i % len(base_markers)],
            markevery=max(1, len(iterations) // 8),
            linewidth=2.0,
            markersize=7,
        )
        ax1.fill_between(
            iterations,
            cumulative_mean - cumulative_std,
            cumulative_mean + cumulative_std,
            color=color,
            alpha=0.15,
        )

        # --- RIGHT: simple regret bar chart with individual run points ---
        row = df_summary.iloc[0]
        mean = row["simple_regret_mean"]
        std = row["simple_regret_std"]
        raw = np.fromstring(row["simple_regrets"].strip("[]"), sep=" ")

        ax2.bar(
            x_pos[i],
            mean,
            yerr=std,
            color=color,
            alpha=0.7,
            capsize=6,
            width=0.5,
            error_kw={"elinewidth": 2, "ecolor": color},
        )
        ax2.scatter(
            x_pos[i] + rng.uniform(-0.08, 0.08, len(raw)),
            raw,
            color=color,
            zorder=5,
            s=40,
            edgecolors="white",
            linewidths=0.8,
        )

    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Cumulative Regret", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_axisbelow(True)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms, fontsize=13)
    ax2.set_ylabel("Simple Regret", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.set_axisbelow(True)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(algorithms),
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"hotpotqa_results_{n_samples}samples.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


def _shorten_model(name: str) -> str:
    return name.split("/")[-1].split("-")[0]


def plot_config_analysis(
    algorithms: list[str],
    n_samples: int,
    n_runs: int,
    save_fig: bool = False,
) -> None:
    """
    Per-run config analysis.

    Layout: (n_algo * n_runs) data rows + 1 table row.
    Each data row = one run: [reward_dist, model, prompt_template, top_k, temperature].
    Final row: best-config-per-run table, one panel per algorithm.
    """
    # Load and preprocess
    algo_data: dict[str, list[dict]] = {}
    for algo in algorithms:
        path = (
            RESULTS_DIR / f"{algo}_hotpotqa_multi_{n_samples}samples_{n_runs}runs.json"
        )
        with open(path) as f:
            all_results = json.load(f)
        processed = []
        for run in all_results:
            df = pd.DataFrame(run["configs"])
            df["model"] = df["model"].apply(_shorten_model)
            bc = dict(run["best_config"])
            bc["model"] = _shorten_model(bc["model"])
            bc["temperature"] = round(bc["temperature"], 2)
            processed.append(
                {
                    "df_cfg": df,
                    "rewards": [1.0 - r for r in run["regrets"]],
                    "best_config": bc,
                }
            )
        algo_data[algo] = processed

    n_algo = len(algorithms)
    # retrieval is now fixed (dense) and no longer searched; prompt_template
    # replaces it as the fourth searched HP.
    hp_order = ["model", "prompt_template", "top_k", "temperature"]
    n_cols = 5  # reward + 4 HPs
    total_data_rows = n_algo * n_runs

    fig = plt.figure(figsize=(20, 3.5 * total_data_rows + 3))
    gs = gridspec.GridSpec(
        total_data_rows + 1, n_cols, figure=fig, hspace=0.8, wspace=0.45
    )

    for a, algo in enumerate(algorithms):
        color = get_algorithm_color(a)
        for r, run_data in enumerate(algo_data[algo]):
            row = a * n_runs + r
            df = run_data["df_cfg"]
            rewards = run_data["rewards"]
            run_label = f"Run {r}" if n_algo == 1 else f"{algo} · Run {r}"

            # Col 0: reward distribution
            ax = fig.add_subplot(gs[row, 0])
            ax.hist(
                rewards,
                bins=20,
                color=color,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel("Reward", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{run_label} — reward", fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Cols 1-4: HP distributions
            for col, hp in enumerate(hp_order, start=1):
                ax = fig.add_subplot(gs[row, col])
                if hp == "temperature":
                    bins = np.arange(0.0, 1.05, 0.1)
                    ax.hist(
                        df[hp].values,
                        bins=bins,
                        color=color,
                        alpha=0.75,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.set_xlabel("temperature", fontsize=10)
                else:
                    cats = sorted(df[hp].unique())
                    vc = df[hp].value_counts()
                    counts = [vc.get(c, 0) for c in cats]
                    ax.bar(
                        np.arange(len(cats)),
                        counts,
                        color=color,
                        alpha=0.75,
                        width=0.6,
                    )
                    ax.set_xticks(np.arange(len(cats)))
                    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
                    ax.set_xlabel(hp, fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_title(f"{run_label} — {hp}", fontsize=11, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    # Bottom row: best config table, one panel per algorithm
    table_row = total_data_rows
    span = n_cols // n_algo
    hp_display = ["model", "prompt_template", "top_k", "temperature"]
    for i, algo in enumerate(algorithms):
        col_start = i * span
        col_end = col_start + span if i < n_algo - 1 else n_cols
        ax = fig.add_subplot(gs[table_row, col_start:col_end])
        ax.axis("off")

        cell_data = [
            [f"Run {r}"]
            + [str(run_data["best_config"].get(hp, "")) for hp in hp_display]
            for r, run_data in enumerate(algo_data[algo])
        ]
        col_labels = ["Run"] + hp_display
        table = ax.table(
            cellText=cell_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (trow, tcol), cell in table.get_celld().items():
            if trow == 0:
                cell.set_facecolor(get_algorithm_color(i))
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f9f9f9" if trow % 2 == 0 else "white")
        ax.set_title(
            f"{algo} — best config per run",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"hotpotqa_config_analysis_{n_samples}samples.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


if __name__ == "__main__":
    # Labels must match algo_label() in hotpotqa_experiment.py (file stems).
    algorithms = ["IMABO", "Random"]  # , "IMABO-noTPE", "Optuna", "Optuna-k5"]
    n_samples = 2000
    n_runs = 5
    plot_hotpotqa_results(
        algorithms=algorithms,
        n_samples=n_samples,
        n_runs=n_runs,
        save_fig=False,
    )
    plot_config_analysis(
        algorithms=algorithms,
        n_samples=n_samples,
        n_runs=n_runs,
        save_fig=False,
    )
