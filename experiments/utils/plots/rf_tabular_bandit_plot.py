"""
Plots for the finite-armed RF tabular HPO experiment
(experiments/rf_tabular_bandit_experiment.py).
"""

import json
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.utils.plots.plot_configs import (
    create_figure_legend,
    get_algorithm_color,
    save_figure,
    set_research_style,
)

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
    """Concat every per-algorithm CSV (rf_tabular_bandit_experiment.py writes one
    file per algorithm, e.g. rf9952_imoss_tabfm_hpo_finite_summary.csv)."""
    pattern = f"{benchmark}_*_{exp_type}_{suffix}.csv"
    paths = sorted(DATA_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {DATA_DIR} -- run "
            f"experiments/rf_tabular_bandit_experiment.py for at least one algorithm first."
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# rf_tabular_bandit_experiment.py's algo_slug() is lossy (both " " and "-" -> "_"),
# so slugs are mapped back to display labels explicitly rather than guessed.
_PRETTY_LABELS = {
    "imoss_tpe": "IMOSS-TPE",
    "imoss": "IMOSS",
    "random_search": "Random Search",
    "imoss_tabfm": "IMOSS-TabFM",
    "ucb_air": "UCB-AIR",
}

# The IMOSS framework variants (same framework, swappable proposal module) --
# as opposed to the external AIR baselines. Handy for the "pluggable proposal"
# figure that shows just the family.
_IMOSS_FAMILY = ["IMOSS", "IMOSS-TPE", "IMOSS-TabFM"]

# Canonical algorithm order -- fixes both the legend order AND the color/marker
# assigned to each algorithm, so a given algorithm looks identical across every
# panel of the multi-benchmark grids below (color from enumerate order would
# otherwise drift if an algorithm is missing from one benchmark's CSV).
_CANONICAL_ORDER = [
    "IMOSS",
    "IMOSS-TPE",
    "IMOSS-TabFM",
    "UCB-AIR",
]
_BASE_MARKERS = ["o", "^", "s", "D", "p"]

# Human-readable OpenML dataset name per benchmark tag, used for subplot titles
# in the multi-benchmark grids (falls back to the upper-cased tag if unknown).
_BENCH_NAMES = {
    "rf146822": "segment",
    "rf31": "credit-g",
    "rf167120": "numerai28.6",
    "rf9952": "phoneme",
    "rf3": "kr-vs-kp",
}


def _style_for(algo: str) -> tuple[str, str]:
    """(color, marker) fixed by the algorithm's canonical position, so the same
    algorithm keeps one look across every panel. Unknown labels fall back to a
    hash-free append at the end of the order."""
    idx = (
        _CANONICAL_ORDER.index(algo)
        if algo in _CANONICAL_ORDER
        else len(_CANONICAL_ORDER)
    )
    return _algo_color(idx), _BASE_MARKERS[idx % len(_BASE_MARKERS)]


def _ordered(present) -> list[str]:
    """Algorithms present AND in _CANONICAL_ORDER, in that order.

    _CANONICAL_ORDER is the source of truth for which algorithms are in the
    comparison: anything not listed there is dropped, so the multi-benchmark
    grids never pick up an un-styled series.
    """
    present = set(present)
    return [a for a in _CANONICAL_ORDER if a in present]


def _bench_title(benchmark: str) -> str:
    return _BENCH_NAMES.get(benchmark, benchmark.upper())


def _load_best_rewards(benchmark: str, n_iterations: int | None) -> tuple[dict, int]:
    """Read best_reward from every per-run JSON checkpoint, grouped by algorithm.

    rf_tabular_bandit_experiment.py logs `best_config`/`best_reward` per run, but only
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
                f"experiments/rf_tabular_bandit_experiment.py for at least one algorithm first."
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
            f"{DATA_DIR} -- re-run experiments/rf_tabular_bandit_experiment.py (best_reward "
            f"logging only applies to runs executed after it was added)."
        )
    return rewards_by_algo, n_iterations


def _load_pull_counts(benchmark: str, n_iterations: int | None) -> tuple[dict, int]:
    """Read the pull-concentration fields from every per-run JSON checkpoint.

    rf_tabular_bandit_experiment.py logs, per run, how heavily an algorithm concentrated
    its budget: `most_suggested_count` (pulls of the arm suggested most often),
    `best_config_suggestions` (pulls of the config it returned as best), and
    `is_best_most_suggested` (whether those are the same arm). Those fields live
    only in the per-run JSON checkpoints, so this reads them directly (mirrors
    _load_best_rewards).

    Returns dict[label] -> {"most_suggested": [...], "best_config": [...],
    "is_same": [...]} plus the resolved n_iterations.
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
                f"experiments/rf_tabular_bandit_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    counts_by_algo = defaultdict(
        lambda: {"most_suggested": [], "best_config": [], "is_same": []}
    )
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        # Older checkpoints predate these fields -- skip them.
        if data.get("most_suggested_count") is None:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        counts_by_algo[label]["most_suggested"].append(data["most_suggested_count"])
        counts_by_algo[label]["best_config"].append(
            data.get("best_config_suggestions", 0)
        )
        counts_by_algo[label]["is_same"].append(
            bool(data.get("is_best_most_suggested"))
        )

    if not counts_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with logged pull counts for T={n_iterations} in "
            f"{DATA_DIR} -- re-run experiments/rf_tabular_bandit_experiment.py (pull-count "
            f"logging only applies to runs executed after it was added)."
        )
    return counts_by_algo, n_iterations


def _load_simple_regret_traces(
    benchmark: str, n_iterations: int | None
) -> tuple[dict, int]:
    """Read the per-iteration anytime simple-regret trace from every per-run JSON.

    rf_tabular_bandit_experiment.py logs `simple_regret_trace` -- at each step, the true
    regret of the config the optimizer would return as best if stopped then (its
    actual selection strategy). Only in the per-run JSONs, so read them directly.

    Returns dict[label] -> list of per-run traces (each a 1-D np.ndarray of length
    n_iterations) plus the resolved n_iterations.
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
                f"experiments/rf_tabular_bandit_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    traces_by_algo = defaultdict(list)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        # Older checkpoints predate this field -- skip them.
        trace = data.get("simple_regret_trace")
        if not trace:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        traces_by_algo[label].append(np.asarray(trace, dtype=float))

    if not traces_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with a simple_regret_trace for T={n_iterations} in "
            f"{DATA_DIR} -- re-run experiments/rf_tabular_bandit_experiment.py (delete the "
            f"old checkpoints first; they are skipped if present and predate this field)."
        )
    return traces_by_algo, n_iterations


def _load_trace_field(
    benchmark: str, n_iterations: int | None, field: str
) -> tuple[dict, int]:
    """Read an arbitrary per-iteration trace `field` from every per-run JSON.

    Generic reader behind the near-optimal-pull-count / far-pull-mean-gap grids
    (field="regrets", the per-pull noiseless regret). Returns dict[label] ->
    list of per-run traces (1-D np.ndarray) plus the resolved n_iterations.
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
                f"experiments/rf_tabular_bandit_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    traces_by_algo = defaultdict(list)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        # Older checkpoints predate this field -- skip them.
        trace = data.get(field)
        if not trace:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        traces_by_algo[label].append(np.asarray(trace, dtype=float))

    if not traces_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with a {field!r} for T={n_iterations} in "
            f"{DATA_DIR} -- re-run experiments/rf_tabular_bandit_experiment.py (delete the "
            f"old checkpoints first; they are skipped if present and predate this field)."
        )
    return traces_by_algo, n_iterations


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
        2,
        1,
        sharex=True,
        figsize=figsize,
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


def _load_true_optimum(benchmark: str):
    """Load the true optimum for whichever RF grid `benchmark` refers to.

    Benchmark tags look like "rf9952" / "rf3", optionally suffixed
    "noiseless" (see rf_tabular_bandit_experiment.py's benchmark_tag) -- the bm_id
    is parsed back out of that tag so the reference line always matches the
    benchmark actually being plotted, rather than always loading bm_id=9952.
    """
    match = re.match(r"rf(\d+)", benchmark)
    if match is None:
        return None
    try:
        from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark

        return RFTabularFiniteBenchmark(bm_id=int(match.group(1))).max_value
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
    max_value = _load_true_optimum(benchmark)

    all_vals = np.concatenate([rewards_by_algo[a] for a in algorithms])
    rng = np.random.default_rng(0)

    def draw(ax):
        for i, algo in enumerate(algorithms):
            rewards = rewards_by_algo[algo]
            color = _algo_color(i)
            x = i + rng.uniform(-0.15, 0.15, size=len(rewards))
            ax.scatter(
                x,
                rewards,
                color=color,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                s=60,
                zorder=3,
            )
            ax.hlines(
                np.mean(rewards), i - 0.25, i + 0.25, color=color, linewidth=3, zorder=4
            )
        if max_value is not None:
            ax.axhline(
                max_value,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"True optimum ({max_value:.4f})",
                zorder=2,
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
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_best_reward_{n_iterations}iters_{exp_type}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_pull_concentration(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Compare the most-suggested arm's pull count vs the returned best config's
    pull count, per algorithm -- i.e. is the arm an algorithm pulls most the same
    one it reports as best?

    Two bars per algorithm (means over runs): the most-suggested arm's pull count
    and the best config's pull count, with per-run points jittered over each bar.
    Each algorithm is annotated with the fraction of runs where the two coincide
    (`is_best_most_suggested`). IMOSS variants pick best via "most_pulled" so the
    two counts always match (100%); argmax-style baselines (e.g. UCB-AIR) can
    diverge.
    """
    from matplotlib.patches import Patch

    counts_by_algo, n_iterations = _load_pull_counts(benchmark, n_iterations)
    # Canonical order + per-algorithm colors consistent with every other figure.
    algorithms = _ordered(counts_by_algo.keys())
    rng = np.random.default_rng(0)

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color encodes the algorithm; the solid-vs-hatched fill encodes the series
    # (most-suggested vs best-config). The legend uses neutral swatches so it
    # only communicates the fill distinction, not an algorithm color.
    for i, algo in enumerate(algorithms):
        data = counts_by_algo[algo]
        most, best = data["most_suggested"], data["best_config"]
        color, _ = _style_for(algo)

        ax.bar(
            i - width / 2,
            np.mean(most),
            width,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        ax.bar(
            i + width / 2,
            np.mean(best),
            width,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
            zorder=2,
        )

        for offset, vals in ((-width / 2, most), (width / 2, best)):
            x = i + offset + rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                x,
                vals,
                color="black",
                alpha=0.55,
                s=18,
                zorder=3,
            )

        frac_same = float(np.mean(data["is_same"])) if data["is_same"] else 0.0
        y_top = max(np.mean(most), np.mean(best))
        ax.text(
            i,
            y_top,
            f"{frac_same:.0%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=20, ha="right")
    ax.set_ylabel("Pull count", fontweight="bold", fontsize=14)
    ax.set_title(
        f"Most-Suggested vs Best-Config Pulls - {_bench_title(benchmark)} (T={n_iterations})",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.12)

    ax.legend(
        [
            Patch(facecolor="0.6", edgecolor="black"),
            Patch(facecolor="0.6", edgecolor="black", hatch="///"),
        ],
        ["Most-suggested arm", "Returned best config"],
        loc="upper right",
        fontsize=11,
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_pull_concentration_{n_iterations}iters_{exp_type}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_pull_concentration_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
):
    """plot_pull_concentration tiled over benchmarks: one subplot each, side by
    side, with a single shared legend for the solid/hatched fill distinction.

    Same content per panel as plot_pull_concentration (most-suggested vs
    returned-best pull counts, per-run points, and the % of runs the two
    coincide), with per-algorithm colors and order consistent across all panels.
    """
    from matplotlib.patches import Patch

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    width = 0.35
    rng = np.random.default_rng(0)
    resolved = n_iterations
    for ax, benchmark in zip(axes, benchmarks):
        counts_by_algo, resolved = _load_pull_counts(benchmark, n_iterations)
        algorithms = _ordered(counts_by_algo.keys())

        for i, algo in enumerate(algorithms):
            data = counts_by_algo[algo]
            most, best = data["most_suggested"], data["best_config"]
            color, _ = _style_for(algo)

            ax.bar(
                i - width / 2,
                np.mean(most),
                width,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
            ax.bar(
                i + width / 2,
                np.mean(best),
                width,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
                hatch="///",
                zorder=2,
            )

            for offset, vals in ((-width / 2, most), (width / 2, best)):
                x = i + offset + rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(x, vals, color="black", alpha=0.55, s=18, zorder=3)

            frac_same = float(np.mean(data["is_same"])) if data["is_same"] else 0.0
            ax.text(
                i,
                max(np.mean(most), np.mean(best)),
                f"{frac_same:.0%}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=20, ha="right")
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.12)

    axes[0].set_ylabel("Pull count", fontweight="bold", fontsize=22)

    create_figure_legend(
        fig,
        [
            Patch(facecolor="0.6", edgecolor="black"),
            Patch(facecolor="0.6", edgecolor="black", hatch="///"),
        ],
        ["Most-suggested arm", "Returned best config"],
        ncol=2,
        fontsize=20,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_pull_concentration_grid_{resolved}iters_{exp_type}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_anytime_simple_regret(
    benchmark="rf9952",
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
    algorithms=None,
    logy=True,
    name_suffix="",
):
    """Anytime simple regret: regret of the returned best config vs iteration.

    One line per algorithm (mean over runs, with a shaded inter-quartile band),
    reading `simple_regret_trace` from the per-run JSONs. This is the standard
    HPO convergence plot -- it shows both that the IMOSS variants converge fast
    *and* how the returned answer's quality differs (e.g. IMOSS-TabFM converging
    lower than IMOSS-TPE). Pass `algorithms` to restrict to a subset of labels
    (e.g. `_IMOSS_FAMILY` for a clean framework-only figure), and `name_suffix`
    to keep that figure's saved file separate from the all-algorithms one.
    """
    traces_by_algo, n_iterations = _load_simple_regret_traces(benchmark, n_iterations)
    labels = algorithms if algorithms is not None else sorted(traces_by_algo.keys())
    labels = [a for a in labels if a in traces_by_algo]

    fig, ax = plt.subplots(figsize=(10, 6))
    base_markers = ["o", "^", "s", "D", "v", "p"]

    for i, algo in enumerate(labels):
        runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
        iters = np.arange(1, runs.shape[1] + 1)
        mean = runs.mean(axis=0)
        q25, q75 = np.percentile(runs, [25, 75], axis=0)

        color = _algo_color(i)
        marker = base_markers[i % len(base_markers)]
        ax.plot(
            iters,
            mean,
            color=color,
            label=algo,
            marker=marker,
            markevery=max(1, len(iters) // 8),
            linewidth=2.5,
            markersize=8,
        )
        ax.fill_between(iters, q25, q75, color=color, alpha=0.15, linewidth=0)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontweight="bold", fontsize=14)
    ax.set_ylabel("Simple Regret (best config so far)", fontweight="bold", fontsize=14)
    ax.set_title(
        f"Anytime Simple Regret - {benchmark.upper()} (T={n_iterations})",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=12, frameon=True, fancybox=True, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_anytime_simple_regret{name_suffix}_{n_iterations}iters_{exp_type}.pdf"
        )
        save_figure(out_path)

    plt.show()


def plot_noise_comparison(
    benchmark="rf9952", n_iterations=None, save_fig=False, exp_type="hpo_finite"
):
    """Compare each algorithm's best-reward spread under Bernoulli (noisy,
    default rf_tabular_bandit_experiment.py behavior) vs noiseless (reward = f(x)
    directly, noise=False) rewards -- checks how much of the run-to-run
    spread/outliers (e.g. IMOSS-TPE's occasional bad runs) comes from reward
    noise itself rather than the search-space/optimizer mechanics.

    Requires experiments/rf_tabular_bandit_experiment.py to have been run for both
    noise=True (default) and noise=False for the algorithms being compared.
    """
    noisy, n_iterations = _load_best_rewards(benchmark, n_iterations)
    noiseless, _ = _load_best_rewards(f"{benchmark}noiseless", n_iterations)

    algorithms = sorted(set(noisy) & set(noiseless))
    if not algorithms:
        raise ValueError(
            "No algorithm has runs under both noise=True and noise=False -- "
            "run experiments/rf_tabular_bandit_experiment.py for both first."
        )

    max_value = _load_true_optimum(benchmark)
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
                x,
                rewards,
                color=color,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.5,
                s=55,
                zorder=3,
            )
            ax.hlines(
                np.mean(rewards), i - 0.25, i + 0.25, color=color, linewidth=3, zorder=4
            )
        if max_value is not None:
            ax.axhline(
                max_value,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
                label=f"True optimum ({max_value:.4f})" if optimum_label else None,
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
            2,
            2,
            figsize=(14, 7),
            sharex="col",
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
            kwargs = dict(
                transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1
            )
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
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_noise_comparison_{n_iterations}iters_{exp_type}.pdf"
        )
        save_figure(out_path)

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
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_cumulative_regret_{n_iterations}iters_{exp_type}.pdf"
        )
        save_figure(out_path)

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
        x,
        means,
        yerr=stds,
        color=colors,
        alpha=0.55,
        capsize=4,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
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
            jitter,
            runs,
            color=colors[i],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            s=40,
            zorder=3,
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
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{benchmark}_simple_regret_vs_iterations_{exp_type}.pdf"
        )
        save_figure(out_path)

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
    create_figure_legend(fig, handles, labels, ncol=len(algorithms))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        out_path = RESULTS_DIR / "paper_plots" / f"{benchmark}_combined_regrets_{exp_type}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_cumulative_regret_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
):
    """Cumulative regret vs iteration, one subplot per benchmark, side by side.

    One line per algorithm (cumsum of mean per-iteration regret), colors/markers
    fixed per algorithm across all panels (see _style_for) and a single shared
    legend on top. y-axes are independent since regret scale differs per
    benchmark. `n_iterations` resolves per benchmark (None -> that benchmark's
    largest budget in the CSV).
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict[str, Any] = {}
    for ax, benchmark in zip(axes, benchmarks):
        df = _load_all(benchmark, exp_type, "iterations")
        budget = df["n_iterations"].max() if n_iterations is None else n_iterations
        df = df[df["n_iterations"] == budget]
        if df.empty:
            ax.set_title(f"{_bench_title(benchmark)}\n(no data)", fontweight="bold")
            continue

        for algo in _ordered(df["algorithm"].unique()):
            algo_data = df[df["algorithm"] == algo]
            iterations = algo_data["iteration"].values
            cumulative_mean = np.cumsum(algo_data["regret_mean"].values)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iterations,
                cumulative_mean,
                color=color,
                label=algo,
                marker=marker,
                markevery=max(1, len(iterations) // 8),
                linewidth=2.0,
                markersize=8,
            )
            seen.setdefault(algo, line)

        ax.set_xlabel("Iteration", fontweight="bold", fontsize=22)
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Cumulative Regret", fontweight="bold", fontsize=22)

    ordered_labels = _ordered(seen.keys())
    create_figure_legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        ncol=len(ordered_labels),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_cumulative_regret_grid_{exp_type}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def _identity_transform(runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, q25, q75) of the raw per-iteration trace, no further transform."""
    mean = runs.mean(axis=0)
    q25, q75 = np.percentile(runs, [25, 75], axis=0)
    return mean, q25, q75


def _near_optimal_count_transform(epsilon: float):
    """(mean, q25, q75) of the cumulative count of pulls with regret <= epsilon."""

    def transform(regret_runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_runs = np.cumsum(regret_runs <= epsilon, axis=1)
        mean = count_runs.mean(axis=0)
        q25, q75 = np.percentile(count_runs, [25, 75], axis=0)
        return mean, q25, q75

    return transform


def _far_pull_mean_gap_transform(epsilon: float):
    """(mean, q25, q75) of the running mean gap among regret > epsilon pulls."""

    def transform(regret_runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        far_mask = regret_runs > epsilon
        far_regret = np.where(far_mask, regret_runs, 0.0)
        far_count = np.cumsum(far_mask, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_gap_runs = np.cumsum(far_regret, axis=1) / far_count
        mean_gap_runs[far_count == 0] = np.nan  # no far pull yet -- undefined
        mean = np.nanmean(mean_gap_runs, axis=0)
        q25 = np.nanpercentile(mean_gap_runs, 25, axis=0)
        q75 = np.nanpercentile(mean_gap_runs, 75, axis=0)
        return mean, q25, q75

    return transform


def _trace_grid(
    benchmarks,
    load_fn,
    transform_fn,
    ylabel,
    out_name,
    *,
    n_iterations,
    save_fig,
    exp_type,
    algorithms,
    logy,
    grid_which,
    use_margins,
):
    """Shared engine behind the anytime trace grids (one subplot per benchmark,
    mean + IQR band per algorithm, fixed canonical colors/markers, one shared
    legend on top). `load_fn(benchmark, n_iterations) -> (traces_by_algo, budget)`
    supplies the raw per-run traces; `transform_fn(runs) -> (mean, q25, q75)`
    turns them into what gets plotted.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict[str, Any] = {}
    for ax, benchmark in zip(axes, benchmarks):
        traces_by_algo, budget = load_fn(benchmark, n_iterations)
        labels = algorithms if algorithms is not None else traces_by_algo.keys()
        labels = _ordered([a for a in labels if a in traces_by_algo])

        for algo in labels:
            runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            mean, q25, q75 = transform_fn(runs)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                label=algo,
                marker=marker,
                markevery=max(1, len(iters) // 8),
                linewidth=2.0,
                markersize=8,
            )
            ax.fill_between(iters, q25, q75, color=color, alpha=0.15, linewidth=0)
            seen.setdefault(algo, line)

        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration", fontweight="bold", fontsize=22)
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, which=grid_which, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if use_margins:
            ax.margins(y=0.08)

    axes[0].set_ylabel(ylabel, fontweight="bold", fontsize=22)

    ordered_labels = _ordered(seen.keys())
    create_figure_legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        ncol=len(ordered_labels),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_{out_name}_{exp_type}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_anytime_simple_regret_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
    algorithms=None,
    logy=True,
):
    """Anytime simple regret vs iteration, one subplot per benchmark, side by side.

    Mirrors plot_anytime_simple_regret (mean over runs + IQR band, reading
    `simple_regret_trace` from the per-run JSONs) but tiles the three benchmarks
    with fixed per-algorithm colors/markers and one shared legend. Pass
    `algorithms` to restrict to a subset (e.g. _IMOSS_FAMILY).
    """
    _trace_grid(
        benchmarks,
        _load_simple_regret_traces,
        _identity_transform,
        "Simple Regret (best config so far)",
        "anytime_simple_regret_grid",
        n_iterations=n_iterations,
        save_fig=save_fig,
        exp_type=exp_type,
        algorithms=algorithms,
        logy=logy,
        grid_which="both",
        use_margins=False,
    )


def plot_near_optimal_pull_count_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
    algorithms=None,
    epsilon=0.01,
):
    """Cumulative count of pulls landing on a near-optimal arm (noiseless regret
    <= epsilon), vs iteration, one subplot per benchmark.

    Built from the per-pull noiseless regret (`regrets`). The optimizer whose
    curve climbs fastest here is the one paying near-zero gaps most often;
    pair with plot_far_pull_mean_gap_grid to see how costly its misses are
    (that band contributes little of the total cumulative regret -- most of
    it comes from the misses' average gap size).

    `epsilon` sets the near-optimal band in accuracy units (default 0.01, i.e.
    within 1 point of the true best -- chosen so all three benchmarks have a
    comparable-sized near-optimal arm set, roughly the top 1-12%).
    """
    load_fn = partial(_load_trace_field, field="regrets")
    _trace_grid(
        benchmarks,
        load_fn,
        _near_optimal_count_transform(epsilon),
        f"Pulls with regret ≤ {epsilon:g}",
        "near_optimal_pull_count_grid",
        n_iterations=n_iterations,
        save_fig=save_fig,
        exp_type=exp_type,
        algorithms=algorithms,
        logy=False,
        grid_which="major",
        use_margins=True,
    )


def plot_far_pull_mean_gap_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
    algorithms=None,
    epsilon=0.01,
    logy=True,
):
    """Anytime mean gap of the NON-near-optimal pulls (noiseless regret > epsilon)
    so far, vs iteration, one subplot per benchmark.

    Companion to plot_near_optimal_pull_count_grid -- that plot counts pulls
    inside the epsilon-band, but on these benchmarks that band contributes only
    ~2-5% of cumulative regret; the rest comes from how bad the "misses" are.
    This plot tracks exactly that: among pulls that missed the near-optimal band,
    what's their average gap so far? A surrogate-guided proposal (e.g. TabFM)
    biases even its misses toward the good region of the space, so its curve
    should sit below a uniform-proposal optimizer's (e.g. vanilla IMOSS) even
    when the latter lands in the band more often -- explaining cases where more
    near-optimal pulls does NOT mean lower cumulative regret.

    Built from the per-pull noiseless regret (`regrets`).
    """
    load_fn = partial(_load_trace_field, field="regrets")
    _trace_grid(
        benchmarks,
        load_fn,
        _far_pull_mean_gap_transform(epsilon),
        f"Mean gap of pulls with regret > {epsilon:g}",
        "far_pull_mean_gap_grid",
        n_iterations=n_iterations,
        save_fig=save_fig,
        exp_type=exp_type,
        algorithms=algorithms,
        logy=logy,
        grid_which="both",
        use_margins=True,
    )


if __name__ == "__main__":
    # The three benchmarks run by experiments/rf_tabular_bandit_experiment.py, spanning
    # reward-noise regimes: segment (clean), credit-g (noisy), numerai28.6 (hard).
    benchmarks = ("rf146822", "rf31", "rf167120")
    n_iterations = 5000

    print("Generating multi-benchmark cumulative-regret grid...")
    plot_cumulative_regret_grid(
        benchmarks=benchmarks, save_fig=True, exp_type="hpo_finite"
    )
    print("Generating multi-benchmark near-optimal-pull-count grid...")
    plot_near_optimal_pull_count_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
        exp_type="hpo_finite",
    )
    print("Generating multi-benchmark far-pull mean-gap grid...")
    plot_far_pull_mean_gap_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
        exp_type="hpo_finite",
    )
