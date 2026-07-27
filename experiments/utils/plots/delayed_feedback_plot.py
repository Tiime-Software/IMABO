"""Plots for the delayed/censored-reward experiment
(experiments/delayed_feedback_experiment.py).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.delayed_feedback_experiment import (
    BENCHMARK,
    LCBENCH_INSTANCES,
    NASBENCH201_INSTANCES,
    PATIENCE_QUANTILE,
)
from experiments.utils.plots.plot_configs import (
    RESEARCH_COLORS,
    _bench_title,
    adaptive_label_fontsize,
    create_figure_legend,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"

# These figures live in the APPENDIX, where we don't need exact main-paper
# print size -- so they follow the paper_style CONVENTION (Wong colors via
# _STYLE, shared top legend via create_figure_legend, style_axis look, markevery
# geometry) but override paper_style's tiny print fonts/lines with larger,
# readable appendix values. `columns` is still honored: it drives how the shared
# legend wraps (via style.legend_ncol) and scales the figure width.
_APPENDIX_PANEL_W_IN = 4.2  # per-benchmark panel width
_APPENDIX_PANEL_H_IN = 3.4  # per-panel height (before legend headroom)


def _appendix_style(columns: int = 2, max_legend_single_row: int = 3):
    """A paper_style bundle with the shared conventions kept but the font/line
    sizes bumped to appendix-readable (the main-paper defaults are ~6.5-9pt,
    sized for a printed column; these figures are viewed larger)."""
    return paper_style(
        "aaai",
        columns=columns,
        markevery_divisor=10,
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        legend_fontsize=16,
        linewidth=2.2,
        markersize=8,
        max_legend_single_row=max_legend_single_row,
    )


def _appendix_figsize(n_panels: int, n_legend_rows: int = 1):
    """Readable appendix figure size: fixed per-panel box, plus a top band for
    the shared legend (~0.55in per legend row) so it never crowds the panel
    titles."""
    return (
        _APPENDIX_PANEL_W_IN * n_panels,
        _APPENDIX_PANEL_H_IN + 0.55 * n_legend_rows,
    )


def _appendix_legend_rect_top(n_legend_rows: int) -> float:
    """Fraction of the figure height left for the panels (the rest is the
    legend band reserved by `_appendix_figsize`), for tight_layout's `rect`."""
    fig_h = _APPENDIX_PANEL_H_IN + 0.55 * n_legend_rows
    return 1.0 - (0.55 * n_legend_rows) / fig_h


# Data dirs and default benchmark tags follow the active benchmark family in
# delayed_feedback_experiment.py, so the same plot module serves the LCBench
# (mixed-space, runtime delay), RF (finite, injected delay), and Criteo (large
# finite, real ad-conversion delay) runs without editing paths here.
if BENCHMARK == "lcbench":
    DATA_DIR = RESULTS_DIR / "delayed_feedback_lcbench"
    SEVERITY_DATA_DIR = RESULTS_DIR / "delayed_feedback_lcbench_severity"
    DEFAULT_BENCHMARKS = tuple(f"lc{i}" for i in LCBENCH_INSTANCES)
    _FIG_PREFIX = "lcbench"
elif BENCHMARK == "nasbench201":
    DATA_DIR = RESULTS_DIR / "delayed_feedback_nasbench201"
    SEVERITY_DATA_DIR = RESULTS_DIR / "delayed_feedback_nasbench201_severity"
    DEFAULT_BENCHMARKS = tuple(f"nb201{i}" for i in NASBENCH201_INSTANCES)
    _FIG_PREFIX = "nasbench201"
else:
    raise ValueError(
        f"Unknown BENCHMARK {BENCHMARK!r}; delayed plots support "
        f"'lcbench' and 'nasbench201'."
    )

set_research_style()

# This experiment's own algorithm identities (distinct from the shared
# ALGORITHM_STYLES in plot_configs.py, which name *oracle* families -- these
# name the same IMABO instance run under different delay/environment
# conditions, see delayed_feedback_experiment.Algorithm) -- kept local so
# this module can't accidentally perturb any other figure's styling.
_PRETTY_LABELS = {
    "imabo_delayed": "IMOSS-TPE Delayed",
    "imabo_naive": "IMOSS-TPE Naive",
    "imabo_nodelay": "IMOSS-TPE",
    "ucb_air": "UCB-AIR",
}

_CANONICAL_ORDER = [
    "IMOSS-TPE",
    "IMOSS-TPE Delayed",
    "IMOSS-TPE Naive",
    "UCB-AIR",
]

_STYLE = {
    "IMOSS-TPE": (RESEARCH_COLORS["success"], "s"),
    "IMOSS-TPE Delayed": (RESEARCH_COLORS["primary"], "^"),
    "IMOSS-TPE Naive": (RESEARCH_COLORS["danger"], "D"),
    "UCB-AIR": (RESEARCH_COLORS["neutral"], "p"),
}

# Only these two ever go through the delayed heap (see run_delayed vs
# run_baseline in experiments/benchmarks/delayed/simulator.py) -- the only
# ones with a non-trivial pending/arrival/censoring trace.
_DELAYED_ALGORITHMS = ["IMOSS-TPE Delayed", "IMOSS-TPE Naive"]


def _ordered(present) -> list[str]:
    present = set(present)
    return [a for a in _CANONICAL_ORDER if a in present]


def _style_for(algo: str) -> tuple[str, str]:
    return _STYLE.get(algo, ("#000000", "o"))


def _run_paths(benchmark: str, n_iterations: int | None, stem_suffix: str = ""):
    """`stem_suffix` matches the same-named kwarg in
    `delayed_feedback_experiment.run_multiple_experiments`: "" for the default
    condition, "_ff100" for the no-censoring (feedback_freq=1.0) condition --
    the two live under distinct filenames so neither glob picks up the other."""
    suffix_re = re.escape(stem_suffix)
    run_pattern = re.compile(
        rf"{re.escape(benchmark)}_(.+)_(\d+)iters{suffix_re}_run(\d+)\.json$"
    )

    if n_iterations is None:
        available = {
            int(m.group(2))
            for p in DATA_DIR.glob(f"{benchmark}_*_*iters{stem_suffix}_run*.json")
            if (m := run_pattern.match(p.name))
        }
        if not available:
            raise FileNotFoundError(
                f"No run checkpoints found in {DATA_DIR} -- run "
                f"experiments/delayed_feedback_experiment.py for at least one algorithm first."
            )
        n_iterations = max(available)

    paths = sorted(
        DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters{stem_suffix}_run*.json")
    )
    return paths, run_pattern, n_iterations


def _load_trace_field(
    benchmark: str, n_iterations: int | None, field: str, stem_suffix: str = ""
) -> tuple[dict, int]:
    """Read an arbitrary per-iteration trace `field` from every per-run JSON
    in this experiment's own DATA_DIR."""
    paths, run_pattern, n_iterations = _run_paths(benchmark, n_iterations, stem_suffix)

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
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    save_fig=False,
    algorithms=None,
    columns: int = 1,
):
    """Cumulative regret vs iteration, one subplot per benchmark, one line
    per algorithm -- the headline comparison: does the delay-aware switching
    rule (IMOSS-TPE Delayed) recover most of the no-delay skyline's performance,
    while the delay-oblivious IMOSS-TPE Naive (same environment, ignores
    pending/censored pulls) lags behind it?

    `columns` (1 = single text column, 2 = full text width / AAAI `figure*`)
    sets the physical figure width via paper_style, so the same function
    renders at whichever placement the figure is embedded at.
    """
    algorithms = algorithms if algorithms is not None else _CANONICAL_ORDER
    # Force the legend onto one row regardless of how many algorithms are
    # plotted -- doesn't affect width, fonts, or any other figure's legend.
    style = _appendix_style(columns, max_legend_single_row=len(algorithms))

    n = len(benchmarks)
    n_rows = style.n_legend_rows(len(algorithms))
    fig, axes = plt.subplots(
        1,
        n,
        figsize=_appendix_figsize(n, n_rows),
    )
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        labels = _ordered([a for a in algorithms if a in traces_by_algo])

        for algo in labels:
            runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            # Per-seed running-average regret first, THEN mean/std across seeds
            # -- cumsum-then-average-across-seeds only gives the right mean
            # (cumsum is linear), but the band needs each seed's own trajectory.
            cum_per_seed = np.cumsum(runs, axis=1) / iters[None, :]
            cumulative_mean = cum_per_seed.mean(axis=0)
            cumulative_std = cum_per_seed.std(axis=0)
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
            ax.fill_between(
                iters,
                cumulative_mean - cumulative_std,
                cumulative_mean + cumulative_std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Online Avg. Regret", fontweight="bold", fontsize=style.label_fontsize
    )

    ordered_labels = _ordered(seen.keys())
    style.legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        n_labels=len(algorithms),
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_delayed_feedback_regret_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_censoring_comparison_grid(
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    save_fig=False,
    columns: int = 1,
):
    """Average regret vs iteration, comparing the default calibrated
    censoring condition (feedback_freq=0.2) against the no-censoring
    condition (feedback_freq=1.0, checkpointed under `stem_suffix="_ff100"`
    in delayed_feedback_experiment.py) for the two delay-exposed algorithms,
    plus the no-delay skyline as a common reference. Isolates the two costs
    stacked on top of each other in the default run:
        skyline vs. (algo, ff=1.0)        -> pure delay cost (no censoring)
        (algo, ff=1.0) vs. (algo, ff=0.2) -> additional censoring cost

    Color = algorithm identity (same _STYLE as every other figure in this
    file); linestyle = censoring condition (solid = default ff=0.2, dashed =
    no-censoring ff=1.0). The skyline never consults a delay model, so it has
    only one condition and is always solid.
    """
    style = _appendix_style(columns)

    swept_algos = ("IMOSS-TPE Delayed", "IMOSS-TPE Naive")
    # Legend order: skyline, then each swept algo's two conditions together.
    legend_order = ["IMOSS-TPE"] + [
        f"{a} ({cond})" for a in swept_algos for cond in ("ff=0.2", "ff=1.0")
    ]

    n = len(benchmarks)
    n_rows = style.n_legend_rows(len(legend_order))
    fig, axes = plt.subplots(1, n, figsize=_appendix_figsize(n, n_rows))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        default_traces, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        no_censor_traces, _ = _load_trace_field(
            benchmark, n_iterations, "regrets", stem_suffix="_ff100"
        )

        def _plot(algo, traces, linestyle, legend_label):
            if algo not in traces:
                return
            runs = np.vstack(traces[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            # Per-seed running average FIRST, then mean/std across seeds --
            # same discipline as plot_cumulative_regret_grid.
            cum_per_seed = np.cumsum(runs, axis=1) / iters[None, :]
            mean = cum_per_seed.mean(axis=0)
            std = cum_per_seed.std(axis=0)
            color, _ = _style_for(algo)
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=style.linewidth,
                label=legend_label,
            )
            ax.fill_between(
                iters,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(legend_label, line)

        _plot("IMOSS-TPE", default_traces, "-", "IMOSS-TPE")
        for algo in swept_algos:
            _plot(algo, default_traces, "-", f"{algo} (ff=0.2)")
            _plot(algo, no_censor_traces, "--", f"{algo} (ff=1.0)")

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Average Regret", fontweight="bold", fontsize=style.label_fontsize
    )

    ordered_labels = [label for label in legend_order if label in seen]
    style.legend(
        fig,
        [seen[label] for label in ordered_labels],
        ordered_labels,
        n_labels=len(legend_order),
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_delayed_feedback_censoring_comparison_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_regret_vs_observations(
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    algorithms=None,
    save_fig=False,
    columns: int = 1,
):
    """Simple regret of the best config found so far (lower is better) vs the
    CUMULATIVE number of rewards actually received -- one subplot per
    benchmark.

    This re-plots the x-axis in units of *observations collected* rather than
    *iterations elapsed*. IMOSS-TPE (no delay) collects one reward per step, so
    it reaches N observations in N steps; IMOSS-TPE Delayed collects only ~20%
    of its pulls (80% censored), so it needs ~5x more steps to reach the same
    N. Plotting all against observations-received normalizes away that
    sample-count gap and asks the sharper question: *given the same amount of
    real feedback, does the delayed/censored learner climb toward the optimum
    as fast?*

    When the curves track each other, the answer is yes -- censoring's only
    cost is fewer observations per wall-clock step ("a few steps late"), not
    worse learning per observation. Curves are truncated to the largest
    observation count reachable by every seed of every plotted algorithm (the
    censored learners' observation budget), so nothing is extrapolated past
    what was actually collected.
    """
    algorithms = (
        algorithms if algorithms is not None else ["IMOSS-TPE", "IMOSS-TPE Delayed"]
    )
    style = _appendix_style(columns)

    n = len(benchmarks)
    n_rows = style.n_legend_rows(len(algorithms))
    fig, axes = plt.subplots(
        1,
        n,
        figsize=_appendix_figsize(n, n_rows),
    )
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        regret_traces, _ = _load_trace_field(
            benchmark, n_iterations, "simple_regret_trace"
        )
        arrived_traces, _ = _load_trace_field(
            benchmark, n_iterations, "num_arrived_this_step"
        )
        # Only algorithms that have BOTH a regret trace and an arrivals trace.
        # UCB-AIR has no per-step arrivals in this pipeline; skip it here.
        labels = _ordered(
            [a for a in algorithms if a in regret_traces and a in arrived_traces]
        )

        # Common x-range = the smallest per-seed total observation count across
        # every plotted algorithm, so no curve is interpolated past its data.
        max_common_obs = min(
            int(np.vstack(arrived_traces[a]).sum(axis=1).min()) for a in labels
        )
        grid = np.arange(1, max_common_obs + 1)

        for algo in labels:
            arr_runs = np.vstack(arrived_traces[algo])  # n_runs x n_iterations
            sr_runs = np.vstack(regret_traces[algo])
            # For each seed: metric as a function of cumulative observations,
            # resampled onto the shared grid. cumsum(arrivals) is non-decreasing,
            # so np.interp maps "N observations collected" -> "incumbent value at
            # the step that count was reached".
            curves = np.vstack(
                [np.interp(grid, np.cumsum(a), s) for a, s in zip(arr_runs, sr_runs)]
            )
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                grid,
                mean,
                color=color,
                label=algo,
                marker=marker,
                markevery=style.markevery(len(grid)),
                # linewidth=style.linewidth,
                linewidth=1.2,
                markersize=style.markersize,
            )
            ax.fill_between(
                grid,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel(
                "Cumulative Observations Received",
                fontweight="bold",
                fontsize=style.label_fontsize,
            )
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Simple Regret",
        fontweight="bold",
        fontsize=style.label_fontsize,
    )

    ordered_labels = _ordered(seen.keys())
    style.legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        n_labels=len(algorithms),
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_delayed_feedback_regret_vs_observations.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_pending_queue_grid(
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    save_fig=False,
):
    """Pending-feedback queue size over time, one subplot per benchmark, one
    line per delayed algorithm (IMOSS-TPE/Random have no queue -- see
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


def plot_active_set_grid(
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    algorithms=None,
    save_fig=False,
    columns: int = 1,
):
    """Rewards observed per active arm (cumsum(num_arrived_this_step) /
    num_active) vs iteration, one subplot per benchmark -- how much
    confirmed feedback backs up each arm the optimizer is currently
    tracking, as a single combined statistic instead of two separately-scaled
    curves (active-arm count vs. cumulative rewards).

    A low ratio means arms are added faster than they're confirmed (spread
    thin, under-observed); a high/rising ratio means each active arm
    accumulates real feedback before new ones are admitted. Naive's admission
    ceiling (t**beta) ignores how much feedback has actually arrived, so under
    delay/censoring it tends to run thin; Delayed's ceiling uses an *effective*
    t that discounts pending (unobserved) pulls by the empirical reward
    frequency, which is exactly the mechanism meant to keep this ratio higher.

    Defaults to the two delay-exposed algorithms (IMOSS-TPE Naive vs Delayed,
    matching plot_pending_queue_grid) since those are the pair the delay-aware
    switching rule is meant to separate; pass `algorithms=_CANONICAL_ORDER` to
    add the no-delay skyline and UCB-AIR's own AIR active set for reference.
    """
    algorithms = (
        algorithms if algorithms is not None else _DELAYED_ALGORITHMS + ["IMOSS-TPE"]
    )
    style = _appendix_style(columns)

    n = len(benchmarks)
    n_rows = style.n_legend_rows(len(algorithms))
    fig, axes = plt.subplots(
        1,
        n,
        figsize=_appendix_figsize(n, n_rows),
    )
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        active_traces, _ = _load_trace_field(benchmark, n_iterations, "num_active")
        arrived_traces, _ = _load_trace_field(
            benchmark, n_iterations, "num_arrived_this_step"
        )
        labels = _ordered(
            [a for a in algorithms if a in active_traces and a in arrived_traces]
        )

        for algo in labels:
            active_runs = np.vstack(active_traces[algo])  # n_runs x n_iterations
            arr_runs = np.vstack(arrived_traces[algo])
            iters = np.arange(1, active_runs.shape[1] + 1)
            # Per-seed ratio FIRST (cumulative rewards / active arms at each
            # t), THEN mean/std across seeds -- ratio-of-means != mean-of-ratios,
            # same discipline used elsewhere in this file.
            cum_rewards = np.cumsum(arr_runs, axis=1)
            ratio_runs = cum_rewards / active_runs
            mean = ratio_runs.mean(axis=0)
            std = ratio_runs.std(axis=0)
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                label=algo,
                linewidth=style.linewidth,
            )
            ax.fill_between(
                iters,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            ax.text(
                iters[-1],
                mean[-1],
                f" {mean[-1]:.1f}",
                color=color,
                fontweight="bold",
                va="center",
                ha="left",
                fontsize=style.tick_fontsize,
            )
            seen.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)
        # Headroom on the right for the end-of-curve value annotations.
        ax.set_xlim(iters[0], iters[-1] * 1.12)

    axes[0].set_ylabel(
        "Rewards per Active Arm",
        fontweight="bold",
        fontsize=style.label_fontsize,
    )

    ordered_labels = _ordered(seen.keys())
    style.legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        n_labels=len(algorithms),
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_delayed_feedback_active_set_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_reward_arrivals_histogram(
    benchmarks=DEFAULT_BENCHMARKS,
    n_iterations=None,
    save_fig=False,
):
    """Distribution of how many rewards arrive per step, IMOSS-TPE (no delay)
    vs. IMOSS-TPE Delayed -- one subplot per benchmark.

    IMOSS-TPE always delivers exactly one reward every step (a single spike at
    x=1, by construction of run_baseline). IMOSS-TPE Delayed's arrivals are
    sparse and bursty: mostly 0 (nothing arrives that step, especially with
    Bernoulli censoring on top of patience), with an occasional higher count
    when a backlog clears all at once. Built from `num_arrived_this_step`,
    pooled across every seed and step, normalized to a probability mass so the
    two algorithms are comparable regardless of how many (seed, step) pairs
    each contributes. Log-y since the x=1 (NoDelay) / x=0 (Delayed) spikes
    dominate everything else.
    """
    algos = ["IMOSS-TPE", "IMOSS-TPE Delayed"]
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for ax, benchmark in zip(axes, benchmarks):
        traces_by_algo, _ = _load_trace_field(
            benchmark, n_iterations, "num_arrived_this_step"
        )
        present = [a for a in algos if a in traces_by_algo]
        if not present:
            continue
        max_count = max(int(np.max(np.concatenate(traces_by_algo[a]))) for a in present)
        bins = np.arange(0, max_count + 2) - 0.5  # integer-centered bins

        for algo in present:
            pooled = np.concatenate(traces_by_algo[algo]).astype(int)
            counts, _ = np.histogram(pooled, bins=bins)
            probs = counts / counts.sum()
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                np.arange(0, max_count + 1),
                probs,
                color=color,
                marker=marker,
                label=algo,
                linewidth=2.0,
                markersize=7,
            )
            seen.setdefault(algo, line)

        ax.set_yscale("log")
        ax.set_xlabel(
            "Rewards Arrived That Step",
            fontweight="bold",
            fontsize=adaptive_label_fontsize(ax),
        )
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=20, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Fraction of Steps (log scale)",
        fontweight="bold",
        fontsize=adaptive_label_fontsize(axes[0]),
    )

    ordered_labels = [a for a in algos if a in seen]
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
            / f"{tag}_delayed_feedback_arrivals_histogram.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def _plot_delay_distribution_lcbench_grid(
    n_samples: int,
    seed: int,
    save_fig: bool,
    patience_steps: int | None,
    colors: tuple,
) -> None:
    """1x3 calibration grid, one panel per LCBench instance -- each instance
    has its own runtime-driven delay model (its own `_time_unit` calibration),
    so unlike RF/NAS-Bench-201 (one shared delay model) the distribution
    genuinely differs across instances and a single representative panel would
    hide that."""
    from experiments.benchmarks.delayed.delay_model import RuntimeDelayModel
    from experiments.benchmarks.delayed.lcbench_bandit import LCBenchMixedBenchmark
    from experiments.benchmarks.delayed.simulator import patience_for_quantile

    BLUE, VERM, GREY = colors
    instances = LCBENCH_INSTANCES
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        rng = np.random.default_rng(seed)
        bench = LCBenchMixedBenchmark(instance=instance)
        model = RuntimeDelayModel()
        point_patience = (
            patience_for_quantile(bench, model, q=PATIENCE_QUANTILE, seed=seed)
            if patience_steps is None
            else patience_steps
        )
        opt_space = bench._bench.get_opt_space()
        try:
            opt_space.seed(seed)
        except Exception:
            pass
        cfgs = bench._sample_opt_configs(n_samples)
        delays, n_censored = [], 0
        for c in cfgs:
            steps = bench.expected_runtime_steps(c)
            d = model.sample_delay_steps(rng, expected_steps=steps)
            if d is None or d > point_patience:
                n_censored += 1  # never observed within the patience window
            else:
                delays.append(d)

        delays = np.asarray(delays, dtype=float)
        median_d = float(np.median(delays)) if len(delays) else 0.0

        d1 = delays + 1.0  # shift so delay=0 is representable on a log axis
        bins = np.logspace(0, np.log10(d1.max() + 1), 45) if len(d1) else 10
        ax.hist(d1, bins=bins, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
        ax.axvline(median_d + 1, color=VERM, lw=2)
        ax.axvline(point_patience + 1, color="black", lw=1.6, ls="--")
        ymax = ax.get_ylim()[1]
        ax.text(
            median_d + 1,
            ymax * 0.98,
            f" median = {median_d:.0f}",
            color=VERM,
            va="top",
            ha="left",
            fontweight="bold",
        )
        ax.text(
            point_patience + 1,
            ymax * 0.72,
            f" patience = {point_patience}",
            color="black",
            va="top",
            ha="left",
        )
        ax.set_xlabel(
            "Delay = predicted training time (steps, log scale)", fontweight="bold"
        )
        ax.set_title(
            f"{_bench_title(f'lc{instance}')}\n"
            f"Censored by patience window: {n_censored}/{n_samples} "
            f"({100 * n_censored / n_samples:.1f}%)",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Count of observed pulls", fontweight="bold")
    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{_FIG_PREFIX}_delayed_feedback_delay_distribution.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_delay_distribution(
    n_samples: int = 20000,
    seed: int = 0,
    save_fig: bool = False,
    patience_steps: int | None = None,
):
    """Calibration/context figure: the distribution of feedback delay the
    experiment actually faces, on a log-x axis (the delay is heavy-tailed, so a
    linear axis crushes all mass into the first bin). Annotates the median, the
    patience window, and the censored fraction. Not derived from any run
    checkpoint.

    `patience_steps=None` (default) uses the SAME per-benchmark quantile the
    experiments use (PATIENCE_QUANTILE of this benchmark's own delay
    distribution), so the "censored by window" annotation matches the runs.

    LCBench: delay is *endogenous* -- each configuration's own
    surrogate-predicted training time (RuntimeDelayModel over a random sample of
    configs). Censoring here is the fraction of configs whose runtime exceeds
    the patience window (crashed/preempted-job analogue). This is the honest
    calibration picture for the runtime-driven setting.

    RF: the fitted log-normal DelayModel with Bernoulli censoring (the original
    injected-delay setting).
    """
    rng = np.random.default_rng(seed)
    BLUE, VERM, GREY = (
        RESEARCH_COLORS["primary"],
        RESEARCH_COLORS["danger"],
        RESEARCH_COLORS["neutral"],
    )

    if BENCHMARK == "nasbench201":
        from experiments.benchmarks.delayed.delay_model import RuntimeDelayModel
        from experiments.benchmarks.delayed.nasbench201_bandit import (
            NASBench201Benchmark,
            NB201_OPS,
        )
        from experiments.benchmarks.delayed.simulator import patience_for_quantile

        bench = NASBench201Benchmark(instance=NASBENCH201_INSTANCES[0])
        model = RuntimeDelayModel()
        if patience_steps is None:
            patience_steps = patience_for_quantile(
                bench, model, q=PATIENCE_QUANTILE, seed=seed
            )
        # sample random architectures (6 edges x 5 ops) and their runtime delays
        delays, n_censored = [], 0
        for _ in range(n_samples):
            cfg = {
                f"edge_{i}": NB201_OPS[rng.integers(len(NB201_OPS))] for i in range(6)
            }
            steps = bench.expected_runtime_steps(cfg)
            d = model.sample_delay_steps(rng, expected_steps=steps)
            if d is None or d > patience_steps:
                n_censored += 1
            else:
                delays.append(d)
        title = (
            f"NAS-Bench-201 runtime-driven delay ({_bench_title(DEFAULT_BENCHMARKS[0])})\n"
            f"Censored by patience window: {n_censored}/{n_samples} "
            f"({100 * n_censored / n_samples:.1f}%)"
        )
        xlabel = "Delay = architecture training time (steps, log scale)"
    elif BENCHMARK == "lcbench":
        # One panel per LCBench instance (its own runtime-driven delay model),
        # rather than a single instance standing in for all three -- each
        # instance has its own `_time_unit` calibration (see
        # LCBenchMixedBenchmark._estimate_reference_optimum), so the delay
        # distribution genuinely differs instance to instance.
        _plot_delay_distribution_lcbench_grid(
            n_samples=n_samples,
            seed=seed,
            save_fig=save_fig,
            patience_steps=patience_steps,
            colors=(BLUE, VERM, GREY),
        )
        return
    else:
        raise ValueError(
            f"Unknown BENCHMARK {BENCHMARK!r}; delayed calibration figure "
            f"supports 'lcbench' and 'nasbench201'."
        )

    delays = np.asarray(delays, dtype=float)
    median_d = float(np.median(delays)) if len(delays) else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    d1 = delays + 1.0  # shift so delay=0 is representable on a log axis
    bins = np.logspace(0, np.log10(d1.max() + 1), 45) if len(d1) else 10
    ax.hist(d1, bins=bins, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(median_d + 1, color=VERM, lw=2)
    ax.axvline(patience_steps + 1, color="black", lw=1.6, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(
        median_d + 1,
        ymax * 0.98,
        f" median = {median_d:.0f}",
        color=VERM,
        va="top",
        ha="left",
        fontweight="bold",
    )
    ax.text(
        patience_steps + 1,
        ymax * 0.72,
        f" patience = {patience_steps}",
        color="black",
        va="top",
        ha="left",
    )
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Count of observed pulls", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{_FIG_PREFIX}_delayed_feedback_delay_distribution.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_regret_vs_arrivals(
    benchmarks=DEFAULT_BENCHMARKS,
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
    """Final average regret (mean of the `regrets` trace, i.e. cumulative sum /
    n_iterations) vs. severity, one entry per swept algorithm -- see
    experiments/delayed_feedback_experiment.py (severity sweep section).

    Averaged rather than summed so the y-axis scale doesn't depend on
    N_ITERATIONS: a raw sum grows ~linearly with the horizon, so a
    statistically-insignificant gap between algorithms looks visually bigger
    just from lengthening the run (both the real signal and the noise scale
    up together) -- the average is horizon-invariant, so the same underlying
    noise looks the same regardless of how long the sweep is run for.

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
        by_algo[label][float(severity_str)].append(float(np.mean(data["regrets"])))

    if not by_algo:
        raise FileNotFoundError(
            f"No severity-sweep checkpoints for sweep={sweep_name!r} in "
            f"{SEVERITY_DATA_DIR} -- run "
            f"experiments/delayed_feedback_experiment.py (severity sweep section) first."
        )

    traces = {}
    for label, per_severity in by_algo.items():
        severities = np.array(sorted(per_severity))
        means = np.array([np.mean(per_severity[s]) for s in severities])
        stds = np.array([np.std(per_severity[s]) for s in severities])
        traces[label] = (severities, means, stds)
    return traces


def _load_reference_regret(benchmark: str) -> dict[str, tuple[float, float]]:
    """Mean/std of final average regret (see `_load_severity_regret`) for the
    severity-invariant reference algorithms (IMOSS-TPE, UCB-AIR) -- see
    experiments.delayed_feedback_experiment.run_reference_algorithms.
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
        by_algo[label].append(float(np.mean(data["regrets"])))
    return {
        label: (float(np.mean(v)), float(np.std(v))) for label, v in by_algo.items()
    }


def _load_severity_trajectories(
    benchmark: str, sweep_name: str, algo_slug: str
) -> dict[float, np.ndarray]:
    """Full per-step regret trajectories for ONE algorithm across every value
    of a severity sweep, from SEVERITY_DATA_DIR. Returns dict[severity] ->
    (n_runs x n_iterations) array of the raw `regrets` traces, so the caller
    can form cumulative regret and average across seeds.
    """
    pattern = re.compile(
        rf"{re.escape(benchmark)}_{re.escape(algo_slug)}_{re.escape(sweep_name)}_"
        rf"([0-9.]+)_(\d+)iters_run(\d+)\.json$"
    )
    by_sev: dict[float, list[np.ndarray]] = defaultdict(list)
    for path in sorted(
        SEVERITY_DATA_DIR.glob(f"{benchmark}_{algo_slug}_{sweep_name}_*_run*.json")
    ):
        match = pattern.match(path.name)
        if not match:
            continue
        severity_str, _n_iterations, _run = match.groups()
        with open(path) as f:
            data = json.load(f)
        by_sev[float(severity_str)].append(np.asarray(data["regrets"], dtype=float))
    if not by_sev:
        raise FileNotFoundError(
            f"No {sweep_name!r}-sweep checkpoints for {benchmark}/{algo_slug} in "
            f"{SEVERITY_DATA_DIR} -- run the severity sweep first."
        )
    return {s: np.vstack(runs) for s, runs in by_sev.items()}


def _load_reference_trajectory(benchmark: str, algo_slug: str) -> np.ndarray | None:
    """Per-step `regrets` traces (n_runs x n_iterations) of a severity-invariant
    reference algorithm (e.g. imabo_nodelay, ucb_air) from the severity dir, or
    None if not present."""
    runs = []
    for path in sorted(
        SEVERITY_DATA_DIR.glob(f"{benchmark}_{algo_slug}_reference_*_run*.json")
    ):
        with open(path) as f:
            runs.append(np.asarray(json.load(f)["regrets"], dtype=float))
    return np.vstack(runs) if runs else None


def plot_cumulative_regret_by_severity(
    sweep_name: str = "delay",
    benchmarks=DEFAULT_BENCHMARKS,
    save_fig=False,
    columns: int = 1,
):
    """Cumulative regret vs iteration across a delay- (or censoring-) severity
    sweep -- one subplot per benchmark.

    Three things per subplot:
      * IMOSS-TPE Delayed at each severity -- SOLID, colored on a gradient
        (light = mild, dark = severe).
      * IMOSS-TPE Naive at each severity -- DASHED, same severity colors.
      * IMOSS-TPE (no delay) -- a single black reference curve (the skyline;
        severity-invariant, so one line).

    Color encodes severity, linestyle encodes algorithm. Unlike
    plot_regret_vs_{delay,censoring}_severity (which collapse each sweep point
    to a single endpoint value), this shows the whole regret trajectory, so you
    can watch the delayed/naive curves fan out and steepen away from the
    skyline as delay worsens -- and see that at every severity the delay-aware
    (solid) curve sits below the naive (dashed) one of the same color.
    Reads per-step `regrets` traces from the severity sweep in SEVERITY_DATA_DIR.
    """
    from matplotlib.lines import Line2D

    style = _appendix_style(columns)
    severities_present = None
    # Only these delay-severity multiples, as requested (drop 0.5x).
    keep_severities = {0.25, 1.0, 2.0, 4.0, 8.0}

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=_appendix_figsize(n, 1))
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")

    def _color(severities, sev):
        frac = severities.index(sev) / max(1, len(severities) - 1)
        return cmap(0.12 + 0.76 * frac)

    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        # Only the delay-aware learner's severity gradient (Naive dropped); the
        # black skyline is the no-delay reference. No solid/dashed algorithm
        # distinction any more, so curves are labeled purely by severity.
        delayed = _load_severity_trajectories(benchmark, sweep_name, "imabo_delayed")
        severities = [s for s in sorted(delayed) if s in keep_severities]
        severities_present = severities

        for sev in severities:
            runs = delayed[sev]
            iters = np.arange(1, runs.shape[1] + 1)
            mean = np.cumsum(runs, axis=1).mean(axis=0) / iters
            ax.plot(
                iters,
                mean,
                color=_color(severities, sev),
                linestyle="--",
                linewidth=style.linewidth,
            )

        # No-delay skyline: single black reference curve (severity-invariant).
        ref = _load_reference_trajectory(benchmark, "imabo_nodelay")
        if ref is not None:
            iters = np.arange(1, ref.shape[1] + 1)
            ax.plot(
                iters,
                np.cumsum(ref, axis=1).mean(axis=0) / iters,
                color="black",
                linewidth=style.linewidth + 0.4,
                zorder=5,
            )

        if idx == n // 2:
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Average Regret", fontweight="bold", fontsize=style.label_fontsize
    )

    order = severities_present or []
    # Legend: one entry per severity color, plus the no-delay skyline.
    handles = [
        Line2D(
            [0],
            [0],
            color=_color(order, s),
            linestyle="--",
            linewidth=style.linewidth + 0.6,
            label=f"{s}x",
        )
        for s in order
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=style.linewidth + 0.6,
            label="No Delay",
        )
    ]
    create_figure_legend(
        fig,
        handles,
        [h.get_label() for h in handles],
        ncol=len(handles),
        bbox_y=1.0,
        fontsize=style.legend_fontsize,
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(1)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_delayed_feedback_cumregret_by_{sweep_name}_severity.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_regret_vs_severity(
    sweep_name: str,
    xlabel: str,
    benchmarks=DEFAULT_BENCHMARKS,
    save_fig=False,
    log_x=False,
    columns: int = 1,
):
    """Final average regret (per-step, horizon-invariant -- see
    `_load_severity_regret`) vs. severity of the swept nuisance parameter
    (delay or censoring), one subplot per benchmark, one line per swept
    algorithm plus flat reference lines for the severity-invariant skyline
    and UCB-AIR.

    This is the standard plot the delayed-bandit literature uses to show
    *how much* a delay-aware correction matters -- final-round regret as a
    function of the delay severity (e.g. Howson et al., 2023, "Delayed
    Feedback in Generalised Linear Bandits Revisited", Figs 3-4) -- rather
    than a single fixed-severity regret-vs-time curve (see
    plot_cumulative_regret_grid): it directly shows whether the gap between
    IMOSS-TPE Delayed and IMOSS-TPE Naive widens as delay/censoring worsens.
    """
    style = _appendix_style(columns)
    n = len(benchmarks)
    n_labels = len(_DELAYED_ALGORITHMS) + 2  # + skyline + UCB-AIR references
    n_rows = style.n_legend_rows(n_labels)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=_appendix_figsize(n, n_rows),
    )
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
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
            ax.fill_between(
                severities,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(algo, line)

        all_severities = np.concatenate([traces[a][0] for a in traces])
        xmin, xmax = all_severities.min(), all_severities.max()
        for algo in ("IMOSS-TPE", "UCB-AIR"):
            if algo not in references:
                continue
            mean, std = references[algo]
            color, _ = _style_for(algo)
            (line,) = ax.plot(
                [xmin, xmax],
                [mean, mean],
                color=color,
                linestyle="--",
                linewidth=style.linewidth,
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
            ax.set_xlabel(xlabel, fontweight="bold", fontsize=style.label_fontsize)
        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Final Average Regret", fontweight="bold", fontsize=style.label_fontsize
    )

    ordered_labels = _ordered(seen.keys())
    style.legend(
        fig,
        [seen[a] for a in ordered_labels],
        ordered_labels,
        n_labels=n_labels,
    )

    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

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
    benchmarks=DEFAULT_BENCHMARKS, save_fig=False, columns: int = 1
):
    plot_regret_vs_severity(
        "delay",
        "Delay Severity (x expected delay)",
        benchmarks=benchmarks,
        save_fig=save_fig,
        log_x=True,
        columns=columns,
    )


def plot_regret_vs_censoring_severity(
    benchmarks=DEFAULT_BENCHMARKS, save_fig=False, columns: int = 1
):
    plot_regret_vs_severity(
        "censor",
        "Feedback Frequency (1 - censoring rate)",
        benchmarks=benchmarks,
        save_fig=save_fig,
        log_x=False,
        columns=columns,
    )


def _skyline_normalized(sweep_name, benchmarks):
    """For each swept algorithm, final average regret divided by the
    per-benchmark no-delay skyline (IMOSS-TPE), averaged over benchmarks. The
    ratio is unaffected by using an average instead of a raw sum (both
    numerator and denominator are computed over the same N_ITERATIONS, so the
    horizon cancels out) -- only the un-normalized `plot_regret_vs_severity`
    y-axis actually needed the average to become horizon-invariant.

    Normalizing each benchmark by its own skyline puts the tasks on one
    comparable [1.0 = skyline] scale so a single averaged curve is meaningful.
    Returns dict[algo] -> (severities, mean_ratio, std_ratio_over_benchmarks).
    """
    per_algo_curves: dict[str, list[np.ndarray]] = defaultdict(list)
    severities_ref = None
    for benchmark in benchmarks:
        traces = _load_severity_regret(benchmark, sweep_name)
        refs = _load_reference_regret(benchmark)
        if "IMOSS-TPE" not in refs:
            raise FileNotFoundError(
                f"No IMOSS-TPE reference for {benchmark} in {SEVERITY_DATA_DIR} "
                f"-- run the severity experiment's reference algorithms first."
            )
        skyline = refs["IMOSS-TPE"][0]
        if skyline <= 0:
            continue
        for algo in _DELAYED_ALGORITHMS:
            if algo not in traces:
                continue
            severities, means, _ = traces[algo]
            severities_ref = severities if severities_ref is None else severities_ref
            per_algo_curves[algo].append(means / skyline)
    out = {}
    for algo, curves in per_algo_curves.items():
        stacked = np.vstack(curves)  # n_benchmarks x n_severities
        out[algo] = (severities_ref, stacked.mean(axis=0), stacked.std(axis=0))
    return out, severities_ref


def plot_delay_effectiveness(
    benchmarks=DEFAULT_BENCHMARKS, save_fig=False, columns: int = 2
):
    """Headline effectiveness figure: final cumulative regret normalized by the
    no-delay skyline (=1.0), averaged over benchmarks, vs. delay severity (left)
    and censoring severity (right). One statement: the delay-aware rule
    (IMOSS-TPE Delayed) tracks the no-delay skyline as the delay regime worsens,
    while the delay-oblivious IMOSS-TPE Naive degrades away from it.

    This is the effectiveness-vs-skyline view; the per-benchmark, un-normalized
    backing is plot_regret_vs_{delay,censoring}_severity. `columns` (1 | 2) sets
    the paper_style figure width; defaults to 2 (full width) since this is a
    two-panel figure.
    """
    delay_curves, delay_sev = _skyline_normalized("delay", benchmarks)
    censor_curves, censor_sev = _skyline_normalized("censor", benchmarks)

    style = _appendix_style(columns)
    n_labels = len(_DELAYED_ALGORITHMS) + 1  # + skyline
    n_rows = style.n_legend_rows(n_labels)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=_appendix_figsize(2, n_rows),
    )
    panels = [
        (
            axes[0],
            delay_curves,
            delay_sev,
            "Delay Severity (x expected delay)",
            True,
        ),
        (
            axes[1],
            censor_curves,
            censor_sev,
            "Feedback Frequency (1 - censoring rate)",
            False,
        ),
    ]
    seen: dict = {}
    for ax, curves, sev, xlabel, log_x in panels:
        for algo in _DELAYED_ALGORITHMS:
            if algo not in curves:
                continue
            severities, mean, std = curves[algo]
            color, marker = _style_for(algo)
            (line,) = ax.plot(
                severities,
                mean,
                color=color,
                marker=marker,
                label=algo,
                linewidth=style.linewidth,
                markersize=style.markersize,
            )
            ax.fill_between(
                severities,
                mean - std,
                mean + std,
                color=color,
                alpha=style.band_alpha,
                linewidth=0,
            )
            seen.setdefault(algo, line)
        # skyline reference at 1.0
        sky_color, _ = _style_for("IMOSS-TPE")
        (sky,) = ax.plot(
            [severities.min(), severities.max()],
            [1.0, 1.0],
            color=sky_color,
            linestyle="--",
            linewidth=style.linewidth,
            label="IMOSS-TPE (skyline)",
        )
        seen.setdefault("IMOSS-TPE", sky)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=style.label_fontsize)
        style.style_axis(ax)

    axes[0].set_ylabel(
        "Final Regret / No-Delay Skyline",
        fontweight="bold",
        fontsize=style.label_fontsize,
    )
    order = [a for a in ("IMOSS-TPE",) + tuple(_DELAYED_ALGORITHMS) if a in seen]
    style.legend(fig, [seen[a] for a in order], order, n_labels=n_labels)
    plt.tight_layout(rect=[0, 0, 1, _appendix_legend_rect_top(n_rows)])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_delayed_feedback_effectiveness.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    benchmarks = DEFAULT_BENCHMARKS
    # Auto-detect from whatever checkpoints exist rather than hardcoding a
    # budget here -- avoids drifting out of sync with n_iter in
    # experiments/delayed_feedback_experiment.py's own __main__ block.
    n_iterations = None
    save_fig = True

    print("Generating cumulative regret grid...")
    plot_cumulative_regret_grid(benchmarks=benchmarks, save_fig=save_fig)

    print("Generating simple-regret-vs-observations figure...")
    plot_regret_vs_observations(benchmarks=benchmarks, save_fig=save_fig)

    print("Generating active-set grid...")
    plot_active_set_grid(benchmarks=benchmarks, save_fig=save_fig)

    plot_regret_vs_censoring_severity(benchmarks=benchmarks, save_fig=save_fig)
