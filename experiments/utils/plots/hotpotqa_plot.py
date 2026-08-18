"""Plot functions for HotpotQA multi-run experiment results."""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    algorithm_style,
    create_figure_legend,
    get_algorithm_color,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results" / "hotpotqa"

ALGO_DISPLAY_NAMES = {
    "IMABO": "IMOSS-TPE",
    "IMABO-noTPE": "IMOSS",
    "IMOSS-TABFM": "IMOSS-TabFM",
    "IMOSS-TABPFN": "IMOSS-TabPFN",
    "IMOSS-TABPFN-untuned": "IMOSS-TabPFN (untuned)",
    "UCB-AIR": "UCB-AIR",
}

_IMABO_FAMILY = ["IMABO", "IMABO-noTPE"]

# Below this physical width, plot_hotpotqa_results' two panels stack
# vertically instead of sitting side by side (see the `stacked` comment
# there). Only AAAI's single-column placement (3.3in) falls under it;
# AAAI's full text width (7.0in) and every single-column venue's one text
# column (NeurIPS 5.5in, arXiv 6.5in) clear it comfortably.
_STACK_BELOW_WIDTH_IN = 4.5


def display_label(algo: str) -> str:
    return ALGO_DISPLAY_NAMES.get(algo, algo)


set_research_style()


def plot_hotpotqa_results(
    algorithms: list[str],
    n_samples: int,
    n_runs: int,
    save_fig: bool = False,
    fig_name: str | None = None,
    display_overrides: dict[str, str] | None = None,
    dirs: dict[str, Path] | None = None,
    columns: int = 2,
    conference: str = "aaai",
) -> None:
    """
    Side-by-side plot: cumulative regret over iterations (left) and
    simple regret bar chart (right), one series per algorithm.

    Args:
        algorithms: Algorithm names to compare, e.g. ["IMABO", "Random"]
        n_samples: Number of optimization steps used in the experiment
        n_runs: Number of runs used in the experiment
        save_fig: Whether to save the figure as PDF
        fig_name: Filename to save under (default: hotpotqa_results_{n_samples}samples.pdf)
        display_overrides: Per-call legend label overrides, merged over
            ALGO_DISPLAY_NAMES (e.g. to label beta-sweep variants distinctly
            without touching the shared display-name map).
        dirs: Per-algorithm results directory override (default RESULTS_DIR),
            for when a series' CSVs live in a different subfolder.
        columns: 1 (fits a single column) or 2 (spans both columns, a LaTeX
            `figure*`; only valid for two-column venues like AAAI) -- see
            plot_configs.paper_figure_width_in. Side-by-side vs. stacked
            panel layout is decided from the resulting physical width, not
            this count directly (see _STACK_BELOW_WIDTH_IN).
        conference: Which conference's column widths to use (default "aaai").
    """
    names = {**ALGO_DISPLAY_NAMES, **(display_overrides or {})}
    label_of = lambda algo: names.get(algo, algo)
    dir_of = lambda algo: (dirs or {}).get(algo, RESULTS_DIR)
    # Legend forced onto a single row (ncol = n algorithms, see the direct
    # create_figure_legend call below); a 1-row legend also needs only a thin
    # top band, which pulls it close to the top panel. Font size is left at
    # PaperStyle's shared default (PAPER_LEGEND_FONTSIZE) so this legend reads
    # at the same size as every other paper figure's, e.g. hpo_plot's.
    style = paper_style(conference=conference, columns=columns)
    algo_tick_fontsize = style.tick_fontsize - 1.0
    n_legend_rows = 1

    legend_band_in = 0.15 * n_legend_rows + 0.03
    # Layout adapts to available width, not the raw `columns` count: AAAI's
    # single-column placement (3.3in) is too narrow to fit two panels side by
    # side (each would be ~half a column), so those stack vertically instead --
    # each panel then spans the full column width. Taller, but that's the
    # natural shape for a single-column figure, and the extra height is what
    # makes the closely-spaced online-regret curves legible. Single-column
    # VENUES (NeurIPS 5.5in, arXiv 6.5in) only ever have columns=1 -- but
    # their one text column is as wide as AAAI's two-column figure*, so it
    # fits the side-by-side layout just fine.
    stacked = style.width_in < _STACK_BELOW_WIDTH_IN
    if stacked:
        panels_in = 2 * (0.62 * style.width_in)
        gridspec_kw = None
        nrows, ncols = 2, 1
    else:
        panels_in = 0.32 * style.width_in
        gridspec_kw = None
        nrows, ncols = 1, 2
    height_in = panels_in + legend_band_in
    legend_top = panels_in / height_in  # axes fill exactly the panel region
    fig, (ax1, ax2) = plt.subplots(
        nrows, ncols, figsize=(style.width_in, height_in), gridspec_kw=gridspec_kw
    )

    x_pos = np.arange(len(algorithms))
    simple_regret_ymax = 0.5
    bar_hatches = ["", "///", "...", "xxx", "\\\\\\", "||"]

    for i, algo in enumerate(algorithms):
        stem = f"{algo}_hotpotqa_{n_samples}samples_{n_runs}runs"
        iter_csv = dir_of(algo) / f"{stem}_iterations.csv"
        summary_csv = dir_of(algo) / f"{stem}_summary.csv"

        df_iter = pd.read_csv(iter_csv)
        df_summary = pd.read_csv(summary_csv)

        # --- LEFT: cumulative regret over iterations ---
        iterations = df_iter["iteration"].values
        mean_regrets = df_iter["regret_mean"].values
        std_regrets = df_iter["regret_std"].values

        cumulative_mean = np.cumsum(mean_regrets) / np.arange(1, len(mean_regrets) + 1)
        cumulative_std = np.sqrt(np.cumsum(std_regrets**2)) / iterations

        color, marker = algorithm_style(algo)
        ax1.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=label_of(algo),
            marker=marker,
            markevery=style.markevery(len(iterations)),
            linewidth=style.linewidth,
            markersize=style.markersize,
        )
        ax1.fill_between(
            iterations,
            cumulative_mean - cumulative_std,
            cumulative_mean + cumulative_std,
            color=color,
            alpha=style.band_alpha,
        )

        # --- RIGHT: simple regret bar chart ---
        row = df_summary.iloc[0]
        mean = row["simple_regret_mean"]
        std = row["simple_regret_std"]
        raw = np.fromstring(row["simple_regrets"].strip("[]"), sep=" ")

        ax2.bar(
            x_pos[i],
            mean,
            yerr=std,
            color=color,
            hatch=bar_hatches[i % len(bar_hatches)],
            edgecolor="black",
            linewidth=0.5,
            capsize=style.capsize,
            width=0.6,
            error_kw={"elinewidth": 1.0, "ecolor": "black", "capthick": style.capthick},
        )
        simple_regret_ymax = max(simple_regret_ymax, mean + std, raw.max())

    ax1.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
    ax1.set_ylabel(
        "Online Avg. Regret", fontweight="bold", fontsize=style.label_fontsize
    )
    style.style_axis(ax1)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [label_of(algo) for algo in algorithms],
        fontsize=algo_tick_fontsize,
        rotation=30,
        ha="right",
    )
    ax2.set_ylabel("Simple Regret", fontweight="bold", fontsize=style.label_fontsize)
    ax2.set_ylim(0.5, simple_regret_ymax + 0.02)
    style.style_axis(ax2, grid_axis="y")

    handles, labels = ax1.get_legend_handles_labels()
    create_figure_legend(
        fig,
        handles,
        labels,
        ncol=len(algorithms),  # single row
        bbox_y=1.0,
        fontsize=style.legend_fontsize,
    )
    plt.tight_layout(
        rect=[0, 0, 1, legend_top], h_pad=1.0 if stacked else None, w_pad=1.0
    )

    if save_fig:
        name = fig_name or f"hotpotqa_results_{n_samples}samples.pdf"
        path = RESULTS_DIR / "paper_plots" / name
        save_figure(path, bbox_inches="tight", parents=True)

    plt.close()


def _shorten_model(name: str) -> str:
    return name.split("/")[-1].split("-")[0]


def plot_config_analysis(
    algorithms: list[str],
    n_samples: int,
    n_runs: int,
    save_fig: bool = False,
    dirs: dict[str, Path] | None = None,
) -> None:
    """
    Per-run config analysis.

    Layout: (n_algo * n_runs) data rows + 1 table row.
    Each data row = one run: [reward_dist, model, prompt_template, top_k, temperature].
    Final row: best-config-per-run table, one panel per algorithm.

    Args:
        dirs: Per-algorithm results directory override (default RESULTS_DIR),
            for when a series' JSON lives in a different subfolder.
    """
    dir_of = lambda algo: (dirs or {}).get(algo, RESULTS_DIR)

    # Load and preprocess
    algo_data: dict[str, list[dict]] = {}
    for algo in algorithms:
        path = (
            dir_of(algo) / f"{algo}_hotpotqa_multi_{n_samples}samples_{n_runs}runs.json"
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
            run_label = (
                f"Run {r}" if n_algo == 1 else f"{display_label(algo)} · Run {r}"
            )

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
            ax.set_xlabel("Reward", fontsize=adaptive_label_fontsize(ax))
            ax.set_ylabel("Count", fontsize=adaptive_label_fontsize(ax))
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
                    ax.set_xlabel("temperature", fontsize=adaptive_label_fontsize(ax))
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
                    ax.set_xlabel(hp, fontsize=adaptive_label_fontsize(ax))
                ax.set_ylabel("Count", fontsize=adaptive_label_fontsize(ax))
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
            f"{display_label(algo)} — best config per run",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    if save_fig:
        path = (
            RESULTS_DIR
            / "paper_plots"
            / f"hotpotqa_config_analysis_{n_samples}samples.pdf"
        )
        save_figure(path, bbox_inches="tight", parents=True)

    plt.close()


if __name__ == "__main__":
    beta_08_dir = RESULTS_DIR / "beta_0.8"
    algorithms = [
        "IMOSS-TPE-beta0.5",
        "IMOSS-TABPFN-beta0.5",
        "IMOSS-mutate-KLxTPE-beta0.5",
        "UCB-AIR-beta0.5",
        "Random",
        "Hier-MAB",
    ]
    dirs = None
    display_overrides = {
        "IMABO-beta0.5": "IMOSS-TPE",
        "IMOSS-TABFM-beta0.5": "IMOSS-TabFM",
        "UCB-AIR-beta0.5": "UCB-AIR",
    }
    n_samples = 5000
    n_runs = 5
    save_fig = True
    plot_hotpotqa_results(
        algorithms=algorithms,
        n_samples=n_samples,
        n_runs=n_runs,
        save_fig=save_fig,
        dirs=dirs,
        display_overrides=display_overrides,
        fig_name=f"hotpotqa_imabo_family_beta_compare_{n_samples}samples.pdf",
        columns=1,
        conference="aaai",
    )
