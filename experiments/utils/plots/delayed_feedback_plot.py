"""Plots for the delayed/censored-reward experiment
(experiments/delayed_feedback_experiment.py).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.benchmarks.delayed.delay_model import DelayModel
from experiments.utils.plots.plot_configs import (
    RESEARCH_COLORS,
    _bench_title,
    adaptive_label_fontsize,
    create_figure_legend,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "delayed_feedback"
SEVERITY_DATA_DIR = RESULTS_DIR / "delayed_feedback_severity"

set_research_style()

# This experiment's own algorithm identities (distinct from the shared
# ALGORITHM_STYLES in plot_configs.py, which name *oracle* families -- these
# name the same IMABO instance run under different delay/environment
# conditions, see delayed_feedback_experiment.Algorithm) -- kept local so
# this module can't accidentally perturb any other figure's styling.
_PRETTY_LABELS = {
    "imabo_delayed": "IMABO-Delayed",
    "imabo_naive": "IMABO-Naive",
    "imabo_nodelay": "IMABO-NoDelay",
    "random_search": "Random Search",
}

_CANONICAL_ORDER = ["IMABO-NoDelay", "IMABO-Delayed", "IMABO-Naive", "Random Search"]

_STYLE = {
    "IMABO-NoDelay": (RESEARCH_COLORS["success"], "s"),
    "IMABO-Delayed": (RESEARCH_COLORS["primary"], "^"),
    "IMABO-Naive": (RESEARCH_COLORS["danger"], "D"),
    "Random Search": (RESEARCH_COLORS["neutral"], "p"),
}

# Only these two ever go through the delayed heap (see run_delayed vs
# run_baseline in experiments/benchmarks/delayed/simulator.py) -- the only
# ones with a non-trivial pending/arrival/censoring trace.
_DELAYED_ALGORITHMS = ["IMABO-Delayed", "IMABO-Naive"]


def _ordered(present) -> list[str]:
    present = set(present)
    return [a for a in _CANONICAL_ORDER if a in present]


def _style_for(algo: str) -> tuple[str, str]:
    return _STYLE.get(algo, ("#000000", "o"))


def _run_paths(benchmark: str, n_iterations: int | None):
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
                f"experiments/delayed_feedback_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    paths = sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json"))
    return paths, run_pattern, n_iterations


def _load_trace_field(
    benchmark: str, n_iterations: int | None, field: str
) -> tuple[dict, int]:
    """Read an arbitrary per-iteration trace `field` from every per-run JSON
    in this experiment's own DATA_DIR."""
    paths, run_pattern, n_iterations = _run_paths(benchmark, n_iterations)

    traces_by_algo = defaultdict(list)
    for path in paths:
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
            f"{DATA_DIR} -- run experiments/delayed_feedback_experiment.py first."
        )
    return traces_by_algo, n_iterations


def plot_cumulative_regret_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=None,
):
    """Cumulative regret vs iteration, one subplot per benchmark, one line
    per algorithm -- the headline comparison: does the delay-aware switching
    rule (IMABO-Delayed) recover most of the no-delay skyline's performance,
    while the delay-oblivious IMABO-Naive (same environment, ignores
    pending/censored pulls) lags behind it?
    """
    algorithms = algorithms if algorithms is not None else _CANONICAL_ORDER

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
            RESULTS_DIR / "paper_plots" / f"{tag}_delayed_feedback_regret_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_pending_queue_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
):
    """Pending-feedback queue size over time, one subplot per benchmark, one
    line per delayed algorithm (IMABO-NoDelay/Random have no queue -- see
    run_baseline in experiments/benchmarks/delayed/simulator.py)."""
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "num_pending")
        labels = [a for a in _DELAYED_ALGORITHMS if a in traces_by_algo]

        for algo in labels:
            runs = np.vstack(traces_by_algo[algo])
            iters = np.arange(1, runs.shape[1] + 1)
            mean = runs.mean(axis=0)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                label=algo,
                linewidth=2.0,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=20, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Pending Feedback Queue Size",
        fontweight="bold",
        fontsize=adaptive_label_fontsize(axes[0]),
    )

    ordered_labels = [a for a in _DELAYED_ALGORITHMS if a in seen]
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
            RESULTS_DIR / "paper_plots" / f"{tag}_delayed_feedback_pending_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_delay_distribution(
    n_samples: int = 20000,
    seed: int = 0,
    save_fig: bool = False,
):
    """Histogram of the fitted Gleipnir delay distribution, sampled directly
    from GleipnirDelayModel (no run checkpoints needed) -- annotates the
    censored fraction. Purely a calibration/context figure, not derived from
    any experiment result."""
    rng = np.random.default_rng(seed)
    model = DelayModel()

    delays = []
    n_censored = 0
    for _ in range(n_samples):
        d = model.sample_delay_steps(rng)
        if d is None:
            n_censored += 1
        else:
            delays.append(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        delays, bins=50, color=RESEARCH_COLORS["primary"], alpha=0.75, edgecolor="black"
    )
    ax.set_xlabel("Delay (steps ~ hours)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(
        "Gleipnir-Calibrated Delay Distribution\n"
        f"Censored: {n_censored}/{n_samples} ({100 * n_censored / n_samples:.1f}%)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR / "paper_plots" / "delayed_feedback_delay_distribution.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_regret_vs_arrivals(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
):
    """Instantaneous regret vs. how much feedback arrived that step, one
    subplot per benchmark, one line per delayed algorithm -- does more
    concurrent feedback correlate with better subsequent picks? Regret is
    grouped by `num_arrived_this_step` and averaged (mean +/- std across
    groups, pooling all seeds), then plotted against the arrival count.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        regret_traces, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        arrival_traces, _ = _load_trace_field(
            benchmark, n_iterations, "num_arrived_this_step"
        )
        labels = [a for a in _DELAYED_ALGORITHMS if a in regret_traces]

        for algo in labels:
            regrets = np.concatenate(regret_traces[algo])
            arrivals = np.concatenate(arrival_traces[algo]).astype(int)

            unique_counts = sorted(set(arrivals.tolist()))
            means = [regrets[arrivals == c].mean() for c in unique_counts]
            stds = [regrets[arrivals == c].std() for c in unique_counts]

            color, marker = _style_for(algo)
            (line,) = ax.plot(
                unique_counts,
                means,
                color=color,
                marker=marker,
                label=algo,
                linewidth=2.0,
                markersize=8,
            )
            ax.fill_between(
                unique_counts,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:
            ax.set_xlabel(
                "Feedback Arrived This Step",
                fontweight="bold",
                fontsize=adaptive_label_fontsize(ax),
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=20, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Instantaneous Regret",
        fontweight="bold",
        fontsize=adaptive_label_fontsize(axes[0]),
    )

    ordered_labels = [a for a in _DELAYED_ALGORITHMS if a in seen]
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
            / f"{tag}_delayed_feedback_regret_vs_arrivals.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def _load_severity_regret(
    benchmark: str, sweep_name: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Final cumulative regret (sum of the `regrets` trace) vs. severity, one
    entry per swept algorithm -- see
    experiments/delayed_feedback_severity_experiment.py.

    Returns dict[label] -> (severities, mean, std), severities sorted
    ascending, mean/std taken across seeds at each severity.
    """
    pattern = re.compile(
        rf"{re.escape(benchmark)}_(.+)_{re.escape(sweep_name)}_"
        rf"([0-9.]+)_(\d+)iters_run(\d+)\.json$"
    )
    by_algo: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(
        SEVERITY_DATA_DIR.glob(f"{benchmark}_*_{sweep_name}_*_run*.json")
    ):
        match = pattern.match(path.name)
        if not match:
            continue
        slug, severity_str, _n_iterations, _run = match.groups()
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        with open(path) as f:
            data = json.load(f)
        by_algo[label][float(severity_str)].append(float(np.sum(data["regrets"])))

    if not by_algo:
        raise FileNotFoundError(
            f"No severity-sweep checkpoints for sweep={sweep_name!r} in "
            f"{SEVERITY_DATA_DIR} -- run "
            f"experiments/delayed_feedback_severity_experiment.py first."
        )

    traces = {}
    for label, per_severity in by_algo.items():
        severities = np.array(sorted(per_severity))
        means = np.array([np.mean(per_severity[s]) for s in severities])
        stds = np.array([np.std(per_severity[s]) for s in severities])
        traces[label] = (severities, means, stds)
    return traces


def _load_reference_regret(benchmark: str) -> dict[str, tuple[float, float]]:
    """Mean/std of final cumulative regret for the severity-invariant
    reference algorithms (IMABO-NoDelay, Random Search) -- see
    experiments/delayed_feedback_severity_experiment.run_reference_algorithms.
    """
    pattern = re.compile(
        rf"{re.escape(benchmark)}_(.+)_reference_(\d+)iters_run(\d+)\.json$"
    )
    by_algo: dict[str, list[float]] = defaultdict(list)
    for path in sorted(SEVERITY_DATA_DIR.glob(f"{benchmark}_*_reference_*_run*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        slug, _n_iterations, _run = match.groups()
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        with open(path) as f:
            data = json.load(f)
        by_algo[label].append(float(np.sum(data["regrets"])))
    return {
        label: (float(np.mean(v)), float(np.std(v))) for label, v in by_algo.items()
    }


def plot_regret_vs_severity(
    sweep_name: str,
    xlabel: str,
    benchmarks=("rf146822", "rf31", "rf167120"),
    save_fig=False,
    log_x=False,
):
    """Final cumulative regret vs. severity of the swept nuisance parameter
    (delay or censoring), one subplot per benchmark, one line per swept
    algorithm plus flat reference lines for the severity-invariant skyline
    and Random Search.

    This is the standard plot the delayed-bandit literature uses to show
    *how much* a delay-aware correction matters -- final-round regret as a
    function of the delay severity (e.g. Howson et al., 2023, "Delayed
    Feedback in Generalised Linear Bandits Revisited", Figs 3-4) -- rather
    than a single fixed-severity regret-vs-time curve (see
    plot_cumulative_regret_grid): it directly shows whether the gap between
    IMABO-Delayed and IMABO-Naive widens as delay/censoring worsens.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces = _load_severity_regret(benchmark, sweep_name)
        references = _load_reference_regret(benchmark)

        for algo in _DELAYED_ALGORITHMS:
            if algo not in traces:
                continue
            severities, mean, std = traces[algo]
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                severities,
                mean,
                color=color,
                marker=marker,
                label=algo,
                linewidth=2.0,
                markersize=8,
            )
            ax.fill_between(
                severities, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            seen.setdefault(algo, line)

        all_severities = np.concatenate([traces[a][0] for a in traces])
        xmin, xmax = all_severities.min(), all_severities.max()
        for algo in ("IMABO-NoDelay", "Random Search"):
            if algo not in references:
                continue
            mean, std = references[algo]
            color, _ = _style_for(algo)
            (line,) = ax.plot(
                [xmin, xmax],
                [mean, mean],
                color=color,
                linestyle="--",
                linewidth=2.0,
                label=algo,
            )
            ax.fill_between(
                [xmin, xmax],
                [mean - std] * 2,
                [mean + std] * 2,
                color=color,
                alpha=0.1,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        if log_x:
            ax.set_xscale("log")
        if idx == n // 2:
            ax.set_xlabel(
                xlabel, fontweight="bold", fontsize=adaptive_label_fontsize(ax)
            )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=20, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Final Cumulative Regret",
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
            / f"{tag}_delayed_feedback_regret_vs_{sweep_name}_severity.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_regret_vs_delay_severity(
    benchmarks=("rf146822", "rf31", "rf167120"), save_fig=False
):
    plot_regret_vs_severity(
        "delay",
        "Delay Severity (x expected delay)",
        benchmarks=benchmarks,
        save_fig=save_fig,
        log_x=True,
    )


def plot_regret_vs_censoring_severity(
    benchmarks=("rf146822", "rf31", "rf167120"), save_fig=False
):
    plot_regret_vs_severity(
        "censor",
        "Feedback Frequency (1 - censoring rate)",
        benchmarks=benchmarks,
        save_fig=save_fig,
        log_x=False,
    )


if __name__ == "__main__":
    benchmarks = ("rf146822", "rf31", "rf167120")
    # Auto-detect from whatever checkpoints exist rather than hardcoding a
    # budget here -- avoids drifting out of sync with n_iter in
    # experiments/delayed_feedback_experiment.py's own __main__ block.
    n_iterations = None

    print("Generating delay distribution calibration figure...")
    plot_delay_distribution(save_fig=True)

    print("Generating cumulative regret grid...")
    plot_cumulative_regret_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating pending-queue grid...")
    plot_pending_queue_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating regret-vs-arrivals grid...")
    plot_regret_vs_arrivals(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating regret-vs-delay-severity grid...")
    plot_regret_vs_delay_severity(benchmarks=benchmarks, save_fig=True)

    print("Generating regret-vs-censoring-severity grid...")
    plot_regret_vs_censoring_severity(benchmarks=benchmarks, save_fig=True)
