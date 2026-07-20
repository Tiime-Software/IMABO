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
    paper_style,
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
_IMOSS_FAMILY = ["IMOSS", "IMOSS-TPE", "IMOSS-TabFM"]
ALL = _IMOSS_FAMILY + ["UCB-AIR"]


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
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Read `shadow_probe_iterations` / `shadow_reward_mean` / `shadow_reward_std`
    from every per-run JSON.

    Returns dict[label] -> (probe_iterations, mean_runs, std_runs). Since this
    experiment probes every ORACLE_PROBE_EVERY iterations rather than every
    iteration (see rf_arm_distribution_experiment.py -- probing every
    iteration is prohibitively expensive for IMOSS-TabFM), `probe_iterations`
    is the shared x-axis (1-D, same schedule for every seed of a given
    n_iterations budget); mean_runs/std_runs are each (n_runs,
    n_probes). mean_runs[k] is the k-th seed's mean true reward of the
    N_SHADOW oracle draws at that probe (see
    rf_arm_distribution_experiment._oracle_propose); std_runs[k] is that same
    seed's std of those N_SHADOW draws.
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

    iterations_by_algo = defaultdict(list)
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
        iterations_by_algo[label].append(
            np.asarray(data["shadow_probe_iterations"], dtype=float)
        )
        means_by_algo[label].append(np.asarray(data["shadow_reward_mean"], dtype=float))
        stds_by_algo[label].append(np.asarray(data["shadow_reward_std"], dtype=float))

    if not means_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with shadow_reward_mean/std for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py first."
        )
    traces = {
        label: (
            iterations_by_algo[label][0],  # same probe schedule for every seed
            np.vstack(means_by_algo[label]),
            np.vstack(stds_by_algo[label]),
        )
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
    """Reward of the arm each oracle would propose, probed every
    ORACLE_PROBE_EVERY iterations (see rf_arm_distribution_experiment.py),
    mean +/- std of N_SHADOW independent draws at each probe, one subplot per
    benchmark.

    The mean line and the std band are each averaged across the n_runs seeds
    (then EMA-smoothed across probes, span in probes not iterations, since the
    x-axis is now sparse) -- this is the "oracle proposal quality" signal:
    Random's should stay roughly flat with a wide, non-shrinking band (always
    samples uniformly, ignoring history); TPE/TabFM should trend toward the
    true optimum with the band narrowing as they concentrate on promising
    regions.
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
            iters, mean_runs, std_runs = traces[algo]
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
        "Oracle Proposal Quality",
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


def _load_candidate_mse_traces(
    benchmark: str, n_iterations: int | None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Read `tabfm_candidate_mse_iterations`/`tabfm_candidate_mse` from every
    per-run JSON (only IMOSS-TabFM logs this field -- see
    rf_arm_distribution_experiment.run_single_experiment).

    Real TabFM fit+predict calls happen on a per-seed clock (the first call
    once enough arms are rewarded, then every `refit_every` calls -- see
    IMABOTabFM.suggest_new), so the logged iteration of the k-th call can
    differ slightly seed to seed, unlike the shadow probe's fixed
    `oracle_probe_every` clock. Seeds are aligned by call index and
    truncated to the shortest run's call count; the shared x-axis is the
    mean logged iteration at each (aligned) call index across seeds.

    Returns dict[label] -> (iterations, mse_mean, mse_std), each 1-D, one
    entry per aligned call index.
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

    iterations_by_algo = defaultdict(list)
    mse_by_algo = defaultdict(list)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        if not data.get("tabfm_candidate_mse"):
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        iterations_by_algo[label].append(
            np.asarray(data["tabfm_candidate_mse_iterations"], dtype=float)
        )
        mse_by_algo[label].append(np.asarray(data["tabfm_candidate_mse"], dtype=float))

    if not mse_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with tabfm_candidate_mse for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py for "
            f"IMOSS-TabFM first."
        )

    traces = {}
    for label, mse_runs_list in mse_by_algo.items():
        min_calls = min(len(m) for m in mse_runs_list)
        iters = np.vstack([it[:min_calls] for it in iterations_by_algo[label]]).mean(
            axis=0
        )
        mse_runs = np.vstack([m[:min_calls] for m in mse_runs_list])
        traces[label] = (iters, mse_runs.mean(axis=0), mse_runs.std(axis=0))
    return traces, n_iterations


def plot_tabfm_candidate_mse_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
    log_scale=True,
):
    """TabFM's candidate-pool MSE vs iteration, one subplot per benchmark.

    MSE is the predicted mean reward (from the surrogate's ensemble) against
    the benchmark's true mean reward, averaged over the n_candidates pool of
    each real TabFM fit+predict call (see
    rf_arm_distribution_experiment._log_candidate_mse and
    IMABOTabFM.on_candidates_scored). Only IMOSS-TabFM logs this field.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces, _ = _load_candidate_mse_traces(benchmark, n_iterations)

        for algo, (iters, mean, std) in traces.items():
            mean = _ema(mean, smoothing_span)
            std = _ema(std, smoothing_span)
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

        if log_scale:
            ax.set_yscale("log")
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
        "TabFM Candidate-Pool MSE",
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
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_tabfm_candidate_mse_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


# Generated directly at its final print width (see plot_configs.
# paper_figure_width_in), unlike the other plots in this file which are
# generated oversized and shrunk to fit \linewidth -- with up to 6 panels,
# there isn't room for that convention here regardless of column count.
def plot_regret_and_oracle_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    regret_algorithms=None,
    oracle_algorithms=None,
    smoothing_span=1,
    columns=2,
    conference="aaai",
):
    """Paper figure: cumulative regret (top row) and oracle proposal quality
    (bottom row), one column per benchmark, sharing a single top legend and
    per-column x-axis -- combines plot_cumulative_regret_grid and
    plot_suggested_arm_distribution_grid into one figure-sized panel instead
    of two, so both are read together and only the bottom row needs an
    "Iteration" label.

    Sized via paper_figure_width_in(columns, conference) for a `columns`-wide
    (1 or 2) placement at (close to) 100% scale -- font sizes are plain pt
    values tuned for print, not the generate-big-then-shrink convention the
    other (single-row) plots in this file use. With `benchmarks` at its
    default length of 3, `columns=1` packs 6 panels into one column's width
    and will be cramped -- prefer `columns=2` (a LaTeX `figure*`) unless
    you've also cut down the number of benchmarks shown.

    regret_algorithms defaults to ALL (includes UCB-AIR, which only has a
    regret trace -- see rf_arm_distribution_experiment.has_oracle);
    oracle_algorithms defaults to _IMOSS_FAMILY (the only ones with a
    shadow-probed oracle trace).
    """
    regret_algorithms = regret_algorithms if regret_algorithms is not None else ALL
    oracle_algorithms = (
        oracle_algorithms if oracle_algorithms is not None else _IMOSS_FAMILY
    )

    style = paper_style(conference=conference, columns=columns)

    n = len(benchmarks)
    # Upper bound on the shared legend's size (the actual `seen` set built
    # below is a subset of this union) -- computed up front, before loading
    # any data, so the figure can be sized to fit it from the start instead
    # of overlapping once the legend is added.
    max_labels = _ordered(set(regret_algorithms) | set(oracle_algorithms))
    n_legend_rows = style.n_legend_rows(len(max_labels))
    # Row height is fixed in inches, not proportional to width: it's driven
    # by how much text (2-line rotated row labels) needs to fit vertically,
    # not by the figure's column width -- at columns=1 the panels are
    # narrower but each row still needs the same vertical room.
    height_in = 2 * 1.4 + 0.22 * (n_legend_rows - 1)
    fig, axes = plt.subplots(2, n, figsize=(style.width_in, height_in), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    seen: dict = {}

    for ax, benchmark in zip(axes[0], benchmarks):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        labels = _ordered([a for a in regret_algorithms if a in traces_by_algo])

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
                markevery=style.markevery(len(iters)),
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
            seen.setdefault(algo, line)

        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0, 0].set_ylabel(
        "Cumulative\nRegret", fontweight="bold", fontsize=style.label_fontsize
    )

    for idx, (ax, benchmark) in enumerate(zip(axes[1], benchmarks)):
        traces, _ = _load_shadow_traces(benchmark, n_iterations)
        labels = _ordered([a for a in oracle_algorithms if a in traces])

        for algo in labels:
            iters, mean_runs, std_runs = traces[algo]
            mean = _ema(mean_runs.mean(axis=0), smoothing_span)
            std = _ema(std_runs.mean(axis=0), smoothing_span)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                label=algo,
                marker=marker,
                markevery=style.markevery(len(iters)),
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
            ax.fill_between(
                iters,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        style.style_axis(ax)

    axes[1, 0].set_ylabel(
        "Oracle Proposal\nQuality", fontweight="bold", fontsize=style.label_fontsize
    )

    for ax in axes.flat:
        ax.label_outer()

    ordered_labels = _ordered(seen.keys())
    n_legend_rows = style.legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        n_labels=len(max_labels),
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97 if n_legend_rows == 1 else 0.95], h_pad=0.6, w_pad=0.6
    )

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_regret_and_oracle_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    benchmarks = ("rf146822", "rf31", "rf167120")
    n_iterations = 5000

    print("Generating combined regret + oracle-proposal-quality grid...")
    plot_regret_and_oracle_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
    )

    # print("Generating TabFM candidate-pool MSE grid...")
    # plot_tabfm_candidate_mse_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )
