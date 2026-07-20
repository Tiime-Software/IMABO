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

from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    algorithm_style,
    create_figure_legend,
    get_algorithm_color,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "hpo_finite"

set_research_style()


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


_PRETTY_LABELS = {
    "imoss_tpe": "IMOSS-TPE",
    "imoss": "IMOSS",
    "random_search": "Random Search",
    "imoss_tabfm": "IMOSS-TabFM",
    "ucb_air": "UCB-AIR",
}

_CANONICAL_ORDER = [
    "IMOSS",
    "IMOSS-TPE",
    "IMOSS-TabFM",
    "UCB-AIR",
]
_BASE_MARKERS = ["o", "^", "s", "D", "p"]

_BENCH_NAMES = {
    "rf146822": "segment",
    "rf31": "credit-g",
    "rf167120": "numerai28.6",
    "rf9952": "phoneme",
    "rf3": "kr-vs-kp",
}


def _style_for(algo: str) -> tuple[str, str]:
    """(color, marker) fixed per algorithm, so the same method keeps one look
    across every panel and every figure. Delegates to the shared canonical
    registry (plot_configs.algorithm_style) -- the single source of truth used
    by every plot script; the local _CANONICAL_ORDER/_algo_color/_BASE_MARKERS
    below are kept only for _ordered() and any legacy callers."""
    return algorithm_style(algo)


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


def _bench_max_value(benchmark: str) -> float:
    """The known optimum for a benchmark tag (e.g. "rf146822" -> bm_id=146822),
    used to convert a regret trace back into reward. RFTabularFiniteBenchmark
    is a pure local CSV lookup (see that module), so this needs no re-running
    of the experiment -- and with default n_values it reconstructs the exact
    grid rf_tabular_bandit_experiment.py used, so max_value matches the
    regrets already logged.
    """
    tag = benchmark[2:] if benchmark.startswith("rf") else benchmark
    tag = tag.removesuffix("noiseless")
    return RFTabularFiniteBenchmark(bm_id=int(tag)).max_value


def _ema(x: np.ndarray, span: float) -> np.ndarray:
    """Causal exponential moving average along the last axis (TensorBoard-style
    smoothing), same length as input. span=1 is a no-op; larger spans smooth
    more heavily at the cost of a slight lag."""
    if span <= 1:
        return x
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


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


def plot_cumulative_regret_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    exp_type="hpo_finite",
):
    """Cumulative regret vs iteration, one subplot per benchmark, side by side."""
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict[str, Any] = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        df = _load_all(benchmark, exp_type, "iterations")
        budget = df["n_iterations"].max() if n_iterations is None else n_iterations
        df = df[df["n_iterations"] == budget]
        if df.empty:
            ax.set_title(f"{_bench_title(benchmark)}\n(no data)", fontweight="bold")
            continue

        for algo in _ordered(df["algorithm"].unique()):
            algo_data = df[df["algorithm"] == algo]
            iterations = algo_data["iteration"].values
            cumulative_mean = np.cumsum(algo_data["regret_mean"].values) / np.arange(
                1, len(iterations) + 1
            )
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
            RESULTS_DIR / "paper_plots" / f"{tag}_cumulative_regret_grid_{exp_type}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_realized_arm_reward_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=None,
    smoothing_span=100,
):
    """Per-iteration reward of the arm actually pulled, mean +/- std across the
    n_runs seeds (EMA-smoothed), one subplot per benchmark.

    Unlike rf_arm_distribution_plot.plot_suggested_arm_distribution_grid --
    which probes a *shadow* copy of the optimizer to see what it would
    currently suggest -- this uses the real trajectory: at each iteration,
    every seed's regret of the arm it actually pulled is converted back to
    reward (reward = max_value - regret, see _bench_max_value) and the
    mean/std is taken across seeds at that iteration.
    """
    algorithms = algorithms if algorithms is not None else []

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict[str, Any] = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        max_value = _bench_max_value(benchmark)
        labels = algorithms if algorithms else traces_by_algo.keys()
        labels = _ordered([a for a in labels if a in traces_by_algo])

        for algo in labels:
            regret_runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            reward_runs = max_value - regret_runs
            iters = np.arange(1, reward_runs.shape[1] + 1)
            mean = _ema(reward_runs.mean(axis=0), smoothing_span)
            std = _ema(reward_runs.std(axis=0), smoothing_span)
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
        "Realized-Arm Reward",
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
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_realized_arm_reward_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


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
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
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
        if idx == n // 2:  # Middle subplot
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, which=grid_which, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if use_margins:
            ax.margins(y=0.08)

    axes[0].set_ylabel(
        ylabel, fontweight="bold", fontsize=adaptive_label_fontsize(axes[0])
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
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_{out_name}_{exp_type}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # The three benchmarks run by experiments/rf_tabular_bandit_experiment.py, spanning
    # reward-noise regimes: segment (clean), credit-g (noisy), numerai28.6 (hard).
    benchmarks = ("rf146822", "rf31", "rf167120")
    n_iterations = 5000

    print("Generating multi-benchmark cumulative-regret grid...")
    plot_cumulative_regret_grid(
        benchmarks=benchmarks, save_fig=True, exp_type="hpo_finite"
    )
    print("Generating multi-benchmark realized-arm-reward grid...")
    plot_realized_arm_reward_grid(benchmarks=benchmarks, save_fig=True)
