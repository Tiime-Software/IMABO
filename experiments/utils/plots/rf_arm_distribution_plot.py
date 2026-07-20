"""Plots for the per-iteration suggested-arm-distribution experiment
(experiments/rf_arm_distribution_experiment.py).
"""

import json
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    create_figure_legend,
    save_figure,
    set_research_style,
)
from experiments.utils.plots.rf_tabular_bandit_plot import (
    _PRETTY_LABELS,
    _bench_title,
    _ema,
    _ordered,
    _style_for,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "hpo_finite_arm_distribution"

set_research_style()

# The IMOSS proposal-oracle family run by rf_arm_distribution_experiment.py
# (no UCB-AIR here -- see that module's docstring).
# _IMOSS_FAMILY = ["IMOSS", "IMOSS-TPE", "IMOSS-TabFM"]
_IMOSS_FAMILY = ["IMOSS-TabFM"]


def _load_trace_field(
    benchmark: str, n_iterations: int | None, field: str
) -> tuple[dict, int]:
    """Read an arbitrary per-iteration trace `field` from every per-run JSON
    (mirrors rf_tabular_bandit_plot._load_trace_field, pointed at this
    experiment's own DATA_DIR instead)."""
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
                f"experiments/rf_arm_distribution_experiment.py for at least one algorithm first."
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
        trace = data.get(field)
        if not trace:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        traces_by_algo[label].append(np.asarray(trace, dtype=float))

    if not traces_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with a {field!r} for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py first."
        )
    return traces_by_algo, n_iterations


def plot_cumulative_regret_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=None,
):
    """Cumulative regret vs iteration, one subplot per benchmark, side by side.

    Mirrors rf_tabular_bandit_plot.plot_cumulative_regret_grid, but built from
    the per-pull noiseless regret (`regrets`) logged in this experiment's own
    per-run JSONs directly (no aggregated CSV exists here -- see
    rf_arm_distribution_experiment.py) rather than from a CSV. No uncertainty
    band, for the same reason as the other module: the cumulative mean grows
    ~O(t) while the cumulative std of a sum grows only ~O(sqrt(t)), so any
    band shrinks to a couple percent of the total well before t=5000.
    """
    algorithms = algorithms if algorithms is not None else _IMOSS_FAMILY

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        labels = _ordered([a for a in algorithms if a in traces_by_algo])

        for algo in labels:
            runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            cumulative_mean = np.cumsum(runs.mean(axis=0))
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iters,
                cumulative_mean,
                color=color,
                label=algo,
                marker=marker,
                markevery=max(1, len(iters) // 8),
                linewidth=2.0,
                markersize=8,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Cumulative Regret",
        fontweight="bold",
        fontsize=adaptive_label_fontsize(axes[0]),
    )

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
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_cumulative_regret_grid_arm_distribution.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def _load_shadow_traces(
    benchmark: str, n_iterations: int | None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """Read `shadow_reward_mean` / `shadow_reward_std` from every per-run JSON.

    Returns dict[label] -> (mean_runs, std_runs), each an (n_runs,
    n_iterations) array, plus the resolved n_iterations. mean_runs[k] is the
    k-th seed's per-iteration mean true reward of the N_SHADOW suggestions
    drawn from that seed's live optimizer state (see
    rf_arm_distribution_experiment._shadow_copy); std_runs[k] is that same
    seed's per-iteration std of those N_SHADOW draws.
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
                f"experiments/rf_arm_distribution_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    means_by_algo = defaultdict(list)
    stds_by_algo = defaultdict(list)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        if data.get("shadow_reward_mean") is None:
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        means_by_algo[label].append(np.asarray(data["shadow_reward_mean"], dtype=float))
        stds_by_algo[label].append(np.asarray(data["shadow_reward_std"], dtype=float))

    if not means_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with shadow_reward_mean/std for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py first."
        )
    traces = {
        label: (np.vstack(means_by_algo[label]), np.vstack(stds_by_algo[label]))
        for label in means_by_algo
    }
    return traces, n_iterations


def plot_suggested_arm_distribution_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=None,
    smoothing_span=1,
):
    """Per-iteration reward of the arm each oracle would currently suggest,
    mean +/- std of N_SHADOW independent live draws (see
    rf_arm_distribution_experiment.py), one subplot per benchmark.

    The mean line and the std band are each averaged across the n_runs seeds
    (then EMA-smoothed, span in iterations) -- this is the "live suggestion
    quality" signal: Random's should stay roughly flat with a wide,
    non-shrinking band (always samples uniformly, ignoring history);
    TPE/TabFM should trend toward the true optimum with the band narrowing as
    they concentrate on promising regions.
    """
    algorithms = algorithms if algorithms is not None else _IMOSS_FAMILY

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    resolved = n_iterations
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces, resolved = _load_shadow_traces(benchmark, n_iterations)
        labels = _ordered([a for a in algorithms if a in traces])

        for algo in labels:
            mean_runs, std_runs = traces[algo]
            iters = np.arange(1, mean_runs.shape[1] + 1)
            mean = _ema(mean_runs.mean(axis=0), smoothing_span)
            std = _ema(std_runs.mean(axis=0), smoothing_span)
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
            ax.fill_between(
                iters, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Suggested-Arm Reward",
        fontweight="bold",
        fontsize=adaptive_label_fontsize(axes[0]),
    )

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
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_suggested_arm_distribution_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    benchmarks = ("rf146822", "rf31", "rf167120")
    n_iterations = 5000

    print("Generating multi-benchmark suggested-arm distribution grid...")
    plot_suggested_arm_distribution_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )
    print("Generating multi-benchmark cumulative-regret grid...")
    plot_cumulative_regret_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )
