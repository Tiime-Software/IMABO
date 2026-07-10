"""
Plots for the finite-armed RF tabular HPO experiment
(experiments/hpo_finite_experiment.py).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.utils.plots.plot_configs import set_research_style, get_algorithm_color

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "hpo_finite"

set_research_style()

# plot_configs' palette has a pale yellow (#F0E442, index 5) that's nearly
# invisible against a white background -- push it to the end of the rotation
# instead of skipping it (so it only appears once >6 algorithms are plotted).
_COLOR_ORDER = [0, 1, 2, 3, 4, 6, 7, 5]


def _algo_color(i: int) -> str:
    return get_algorithm_color(_COLOR_ORDER[i % len(_COLOR_ORDER)])


def _load_all(benchmark: str, exp_type: str, suffix: str) -> pd.DataFrame:
    """Concat every per-algorithm CSV (hpo_finite_experiment.py writes one
    file per algorithm, e.g. rf9952_imoss_tabfm_hpo_finite_summary.csv)."""
    pattern = f"{benchmark}_*_{exp_type}_{suffix}.csv"
    paths = sorted(DATA_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {DATA_DIR} -- run "
            f"experiments/hpo_finite_experiment.py for at least one algorithm first."
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# hpo_finite_experiment.py's algo_slug() is lossy (both " " and "-" -> "_"),
# so slugs are mapped back to display labels explicitly rather than guessed.
_PRETTY_LABELS = {
    "imoss_tpe": "IMOSS-TPE",
    "imoss": "IMOSS",
    "random_search": "Random Search",
    "imoss_tabfm": "IMOSS-TabFM",
    "ucb_air": "UCB-AIR",
    "moss_air": "MOSS-AIR",
}


def _load_best_rewards(benchmark: str, n_iterations: int | None) -> tuple[dict, int]:
    """Read best_reward from every per-run JSON checkpoint, grouped by algorithm.

    hpo_finite_experiment.py logs `best_config`/`best_reward` per run, but only
    inside the per-run JSON checkpoints -- the aggregated CSVs (used by the
    other plots here) only carry regrets/simple_regrets, so this reads the
    raw run files directly instead.
    """
    run_pattern = re.compile(rf"{re.escape(benchmark)}_(.+)_(\d+)iters_run(\d+)\.json$")

    if n_iterations is None:
        available = {
            int(m.group(2))
            for p in DATA_DIR.glob(f"{benchmark}_*_*iters_run*.json")
            if (m := run_pattern.match(p.name))
        }
        if not available:
            raise FileNotFoundError(
                f"No run checkpoints found in {DATA_DIR} -- run "
                f"experiments/hpo_finite_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    rewards_by_algo = defaultdict(list)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        if data.get("best_reward") is None:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        rewards_by_algo[label].append(data["best_reward"])

    if not rewards_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with a logged best_reward for T={n_iterations} in "
            f"{DATA_DIR} -- re-run experiments/hpo_finite_experiment.py (best_reward "
            f"logging only applies to runs executed after it was added)."
        )
    return rewards_by_algo, n_iterations


def _detect_break(all_vals: np.ndarray) -> tuple[float, float] | None:
    """Find the largest gap in sorted values; return it if it exceeds 15% of
    the total range (a sign of rare severe outliers vs. an otherwise tight
    cluster), else None."""
    all_vals = np.sort(all_vals)
    span = all_vals[-1] - all_vals[0]
    if span <= 0 or len(all_vals) < 2:
        return None
    gaps = np.diff(all_vals)
    gap_idx = int(np.argmax(gaps))
    if gaps[gap_idx] > 0.15 * span:
        return (all_vals[gap_idx], all_vals[gap_idx + 1])
    return None


def _broken_axis_figure(all_vals: np.ndarray, figsize: tuple[float, float]):
    """Return (fig, axes, split) -- either a single normal axis (split=None),
    or two stacked axes with a broken y-axis around the largest gap in
    all_vals (see _detect_break), break marks already drawn. axes[-1] is
    always the bottom (x-labeled) axis; ylims are already set on both.
    """
    split = _detect_break(all_vals)

    if split is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, [ax], None

    low_max, high_min = split
    gap = high_min - low_max
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )
    ax_top.set_ylim(
        high_min - 0.05 * gap,
        all_vals.max() + 0.08 * ((all_vals.max() - high_min) or gap),
    )
    ax_bot.set_ylim(
        all_vals.min() - 0.08 * ((low_max - all_vals.min()) or gap),
        low_max + 0.05 * gap,
    )

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    d = 0.010
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return fig, [ax_top, ax_bot], split


def _load_true_optimum():
    try:
        from experiments.benchmarks.tabular_finite import RFTabularFiniteBenchmark

        return RFTabularFiniteBenchmark().max_value
    except Exception:
        return None


def plot_best_reward_per_run(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Scatter each run's best-config reward per algorithm (higher is better).

    One point per run (jittered horizontally so overlapping runs stay
    visible), with a bar at the per-algorithm mean. A dashed reference line
    marks the true optimum (bench.max_value) when the benchmark can be loaded.

    When a few severe outlier runs sit far below an otherwise tight cluster
    (the largest gap in the combined data exceeds 15% of the total range),
    the y-axis is automatically broken around that gap (see
    _broken_axis_figure) -- otherwise a plain single-panel plot is used.
    """
    rewards_by_algo, n_iterations = _load_best_rewards(benchmark, n_iterations)
    algorithms = sorted(rewards_by_algo.keys())
    max_value = _load_true_optimum()

    all_vals = np.concatenate([rewards_by_algo[a] for a in algorithms])
    rng = np.random.default_rng(0)

    def draw(ax):
        for i, algo in enumerate(algorithms):
            rewards = rewards_by_algo[algo]
            color = _algo_color(i)
            x = i + rng.uniform(-0.15, 0.15, size=len(rewards))
            ax.scatter(
                x, rewards, color=color, alpha=0.7, edgecolors="black",
                linewidths=0.5, s=60, zorder=3,
            )
            ax.hlines(np.mean(rewards), i - 0.25, i + 0.25, color=color, linewidth=3, zorder=4)
        if max_value is not None:
            ax.axhline(
                max_value, color="black", linestyle="--", linewidth=1.5,
                alpha=0.6, label=f"True optimum ({max_value:.4f})", zorder=2,
            )
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["right"].set_visible(False)

    title = f"Best Reward per Run - {benchmark.upper()} (T={n_iterations})"
    figsize = (10, 6) if _detect_break(all_vals) is None else (10, 7)
    fig, axes, split = _broken_axis_figure(all_vals, figsize)

    for ax in axes:
        draw(ax)
    top_ax, bottom_ax = axes[0], axes[-1]
    if split is not None:
        top_ax.spines["top"].set_visible(True)
    else:
        top_ax.spines["top"].set_visible(False)

    bottom_ax.set_xticks(range(len(algorithms)))
    bottom_ax.set_xticklabels(algorithms, rotation=20, ha="right")
    top_ax.set_ylabel("Best Config Reward", fontweight="bold", fontsize=14)
    top_ax.set_title(title, fontweight="bold", fontsize=16, pad=15)
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=12)
    if max_value is not None:
        top_ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)

    plt.tight_layout()

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        out_path = (
            output_dir / f"{benchmark}_best_reward_{n_iterations}iters_{exp_type}.pdf"
        )
        plt.savefig(out_path)
        print(f"Saved to {out_path}")

    plt.show()


def plot_noise_comparison(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Compare each algorithm's best-reward spread under Bernoulli (noisy,
    default hpo_finite_experiment.py behavior) vs noiseless (reward = f(x)
    directly, noise=False) rewards -- checks how much of the run-to-run
    spread/outliers (e.g. IMOSS-TPE's occasional bad runs) comes from reward
    noise itself rather than the search-space/optimizer mechanics.

    Requires experiments/hpo_finite_experiment.py to have been run for both
    noise=True (default) and noise=False for the algorithms being compared.
    """
    noisy, n_iterations = _load_best_rewards(benchmark, n_iterations)
    noiseless, _ = _load_best_rewards(f"{benchmark}noiseless", n_iterations)

    algorithms = sorted(set(noisy) & set(noiseless))
    if not algorithms:
        raise ValueError(
            "No algorithm has runs under both noise=True and noise=False -- "
            "run experiments/hpo_finite_experiment.py for both first."
        )

    max_value = _load_true_optimum()
    all_vals = np.concatenate(
        [noisy[a] for a in algorithms] + [noiseless[a] for a in algorithms]
    )
    split = _detect_break(all_vals)
    rng = np.random.default_rng(0)

    def draw(ax, data_by_algo, optimum_label=False):
        for i, algo in enumerate(algorithms):
            rewards = data_by_algo[algo]
            color = _algo_color(i)
            x = i + rng.uniform(-0.15, 0.15, size=len(rewards))
            ax.scatter(
                x, rewards, color=color, alpha=0.75, edgecolors="black",
                linewidths=0.5, s=55, zorder=3,
            )
            ax.hlines(np.mean(rewards), i - 0.25, i + 0.25, color=color, linewidth=3, zorder=4)
        if max_value is not None:
            ax.axhline(
                max_value, color="black", linestyle="--", linewidth=1.5, alpha=0.6,
                zorder=2, label=f"True optimum ({max_value:.4f})" if optimum_label else None,
            )
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3)

    def finish_column(ax, subtitle):
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=20, ha="right")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_title(subtitle, fontsize=14, fontweight="bold")

    title = f"Bernoulli vs Noiseless Reward - {benchmark.upper()} (T={n_iterations})"

    if split is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        draw(ax_l, noisy)
        draw(ax_r, noiseless, optimum_label=True)
        finish_column(ax_l, "Bernoulli (noisy)")
        finish_column(ax_r, "Noiseless")
        ax_l.spines["top"].set_visible(False)
        ax_r.spines["top"].set_visible(False)
        ax_l.spines["right"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.tick_params(left=False)
        ax_l.set_ylabel("Best Config Reward", fontweight="bold", fontsize=14)
        ax_r.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)
    else:
        low_max, high_min = split
        gap = high_min - low_max
        fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(
            2, 2, figsize=(14, 7), sharex="col",
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06, "wspace": 0.06},
        )
        draw(ax_tl, noisy)
        draw(ax_tr, noiseless, optimum_label=True)
        draw(ax_bl, noisy)
        draw(ax_br, noiseless)

        top_ylim = (
            high_min - 0.05 * gap,
            all_vals.max() + 0.08 * ((all_vals.max() - high_min) or gap),
        )
        bot_ylim = (
            all_vals.min() - 0.08 * ((low_max - all_vals.min()) or gap),
            low_max + 0.05 * gap,
        )
        for ax in (ax_tl, ax_tr):
            ax.set_ylim(*top_ylim)
        for ax in (ax_bl, ax_br):
            ax.set_ylim(*bot_ylim)

        for ax_top, ax_bot in ((ax_tl, ax_bl), (ax_tr, ax_br)):
            ax_top.spines["bottom"].set_visible(False)
            ax_bot.spines["top"].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            d = 0.012
            kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_bot.transAxes)
            ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        finish_column(ax_bl, "")
        finish_column(ax_br, "")
        ax_tl.set_title("Bernoulli (noisy)", fontsize=14, fontweight="bold")
        ax_tr.set_title("Noiseless", fontsize=14, fontweight="bold")
        for ax in (ax_tl, ax_tr, ax_bl, ax_br):
            ax.tick_params(axis="both", which="major", labelsize=12)

        ax_tl.spines["right"].set_visible(False)
        ax_bl.spines["right"].set_visible(False)
        ax_tr.spines["left"].set_visible(False)
        ax_br.spines["left"].set_visible(False)
        ax_tr.tick_params(labelleft=False)
        ax_br.tick_params(labelleft=False)
        ax_tl.set_ylabel("Best Config Reward", fontweight="bold", fontsize=14)
        ax_tr.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        out_path = (
            output_dir / f"{benchmark}_noise_comparison_{n_iterations}iters_{exp_type}.pdf"
        )
        plt.savefig(out_path)
        print(f"Saved to {out_path}")

    plt.show()


def plot_cumulative_regret_over_iterations(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Plot cumulative regret over iterations for all algorithms at a specific
    iteration budget (defaults to the largest budget present in the CSV)."""
    df = _load_all(benchmark, exp_type, "iterations")

    if n_iterations is None:
        n_iterations = df["n_iterations"].max()
    df = df[df["n_iterations"] == n_iterations]

    if df.empty:
        print(f"No data found for {n_iterations} iterations")
        return

    algorithms = df["algorithm"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    base_markers = ["o", "^", "s", "D", "v", "p"]

    for i, algo in enumerate(algorithms):
        algo_data = df[df["algorithm"] == algo]

        iterations = algo_data["iteration"].values
        cumulative_mean = np.cumsum(algo_data["regret_mean"].values)

        color = _algo_color(i)
        marker = base_markers[i % len(base_markers)]

        ax.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=algo,
            marker=marker,
            markevery=max(1, len(iterations) // 8),
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Iteration", fontweight="bold", fontsize=14)
    ax.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=14)
    ax.set_title(
        f"Cumulative Regret over {n_iterations} Iterations - {benchmark.upper()}",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=12, frameon=True, fancybox=True, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        out_path = (
            output_dir
            / f"{benchmark}_cumulative_regret_{n_iterations}iters_{exp_type}.pdf"
        )
        plt.savefig(out_path)
        print(f"Saved to {out_path}")

    plt.show()


def _plot_simple_regret_bars(ax, df_summary, algorithms, fontsize=14):
    """Bar chart: one bar per algorithm (a single fixed iteration budget),
    with individual per-run simple_regret values scattered on top."""
    x = np.arange(len(algorithms))
    means, stds = [], []
    for algo in algorithms:
        algo_data = df_summary[df_summary["algorithm"] == algo]
        means.append(algo_data["simple_regret_mean"].iloc[0])
        stds.append(algo_data["simple_regret_std"].iloc[0])

    colors = [_algo_color(i) for i in range(len(algorithms))]
    ax.bar(
        x, means, yerr=stds, color=colors, alpha=0.55, capsize=4,
        edgecolor="black", linewidth=0.8, zorder=2,
    )

    rng = np.random.default_rng(0)
    for i, algo in enumerate(algorithms):
        algo_data = df_summary[df_summary["algorithm"] == algo]
        raw = algo_data["simple_regrets"].iloc[0]
        if pd.isna(raw):
            continue
        runs = np.fromstring(str(raw).strip("[]"), sep=" ")
        if runs.size == 0:
            continue
        jitter = i + rng.uniform(-0.25, 0.25, size=runs.size)
        ax.scatter(
            jitter, runs, color=colors[i], alpha=0.8, edgecolors="black",
            linewidths=0.5, s=40, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20, ha="right")
    ax.set_ylabel("Simple Regret", fontweight="bold", fontsize=fontsize)


def plot_simple_regret_vs_iterations(
    benchmark="rf9952", save_fig=False, exp_type="hpo_finite"
):
    """Bar plot of simple regret, one bar per algorithm."""
    df = _load_all(benchmark, exp_type, "summary")

    algorithms = df["algorithm"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_simple_regret_bars(ax, df, algorithms)

    ax.set_title(
        f"Simple Regret by Algorithm - {benchmark.upper()}",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        out_path = (
            output_dir / f"{benchmark}_simple_regret_vs_iterations_{exp_type}.pdf"
        )
        plt.savefig(out_path)
        print(f"Saved to {out_path}")

    plt.show()


def plot_combined_regrets(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Cumulative regret and simple regret side by side, with a common legend."""
    df_iterations = _load_all(benchmark, exp_type, "iterations")
    df_summary = _load_all(benchmark, exp_type, "summary")

    if n_iterations is None:
        n_iterations = df_iterations["n_iterations"].max()
    df_iterations = df_iterations[df_iterations["n_iterations"] == n_iterations]

    if df_iterations.empty:
        print(f"No data found for {n_iterations} iterations")
        return

    algorithms = df_iterations["algorithm"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    base_markers = ["o", "^", "s", "D", "v", "p"]

    # --- LEFT: cumulative regret over iterations, at the chosen budget ---
    for i, algo in enumerate(algorithms):
        algo_data = df_iterations[df_iterations["algorithm"] == algo]

        iterations = algo_data["iteration"].values
        cumulative_mean = np.cumsum(algo_data["regret_mean"].values)

        color = _algo_color(i)
        marker = base_markers[i % len(base_markers)]

        ax1.plot(
            iterations,
            cumulative_mean,
            color=color,
            label=algo,
            marker=marker,
            markevery=max(1, len(iterations) // 8),
            linewidth=2.0,
            markersize=8,
        )

    ax1.set_xlabel("Iteration", fontweight="bold", fontsize=22)
    ax1.set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)
    ax1.tick_params(axis="both", which="major", labelsize=20)
    ax1.set_axisbelow(True)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- RIGHT: simple regret, one bar per algorithm ---
    _plot_simple_regret_bars(ax2, df_summary, algorithms, fontsize=22)
    ax2.tick_params(axis="both", which="major", labelsize=20)
    ax2.set_axisbelow(True)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(algorithms),
        frameon=False,
        prop={"size": 22, "family": "serif", "weight": "bold"},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        output_dir = RESULTS_DIR / "paper_plots"
        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / f"{benchmark}_combined_regrets_{exp_type}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    print("Generating combined regrets plot for RF9952 (finite HPO experiment)...")
    plot_combined_regrets(benchmark="rf9952", save_fig=True, exp_type="hpo_finite")
    plot_best_reward_per_run(benchmark="rf9952", save_fig=True, exp_type="hpo_finite")
