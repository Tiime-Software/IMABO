"""Plots for the per-iteration suggested-arm-distribution experiment
(experiments/rf_arm_distribution_experiment.py).
"""

import json
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from experiments.utils.plots.plot_configs import (
    _PRETTY_LABELS,
    _bench_title,
    _ema,
    _ordered,
    _style_for,
    adaptive_label_fontsize,
    create_figure_legend,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "hpo_finite_arm_distribution"

set_research_style()

_IMOSS_FAMILY = ["IMOSS", "IMOSS-TPE", "IMOSS-TabFM"]
ALL = _IMOSS_FAMILY + ["UCB-AIR"]

_ORACLE_LABELS = {"IMOSS": "Random", "IMOSS-TPE": "TPE", "IMOSS-TabFM": "TabFM"}


def _load_trace_field(
    benchmark: str, n_iterations: int | None, field: str
) -> tuple[dict, int]:
    """Read an arbitrary per-iteration trace `field` from every per-run JSON
    in this experiment's own DATA_DIR."""
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

    Built from the per-pull noiseless regret (`regrets`) logged in this
    experiment's own per-run JSONs directly (no aggregated CSV exists for
    this experiment -- see rf_arm_distribution_experiment.py). No uncertainty
    band: the cumulative mean grows ~O(t) while the cumulative std of a sum
    grows only ~O(sqrt(t)), so any band shrinks to a couple percent of the
    total well before t=5000.
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


# (iter_field, predicted_field, true_field) for the two predicted-vs-true
# sources logged by rf_arm_distribution_experiment.run_single_experiment:
# the chosen configs across the N_SHADOW draws, and the full scored candidate
# pool (all n_candidates) from the one real TabFM fit per probe.
_SUGGESTION_FIELDS = (
    "tabfm_suggestion_probe_iterations",
    "tabfm_suggestion_predicted_rewards",
    "tabfm_suggestion_true_rewards",
)
_CANDIDATE_FIELDS = (
    "tabfm_candidate_probe_iterations",
    "tabfm_candidate_predicted_rewards",
    "tabfm_candidate_true_rewards",
)


def _load_predicted_true_raw(
    benchmark: str,
    n_iterations: int | None,
    fields: tuple[str, str, str] = _SUGGESTION_FIELDS,
) -> tuple[dict[str, list[dict[int, tuple[list[float], list[float]]]]], int]:
    """Read raw per-probe (predicted, true) reward pairs from every per-run
    JSON (only IMOSS-TabFM logs these fields -- see
    rf_arm_distribution_experiment.run_single_experiment).

    `fields` picks the source: _SUGGESTION_FIELDS (the chosen configs, one
    value per shadow-probe draw) or _CANDIDATE_FIELDS (the whole scored
    candidate pool, ~n_candidates values per probe). Kept raw and
    unaggregated so any per-value metric (squared error, signed bias, ...)
    can be derived downstream via _metric_traces without rerunning.

    Returns dict[label] -> list (one per seed) of dict[iteration] ->
    (predicted_rewards, true_rewards).
    """
    iter_field, pred_field, true_field = fields
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

    raw_by_algo: dict[str, list[dict[int, tuple[list[float], list[float]]]]] = (
        defaultdict(list)
    )
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        if not data.get(pred_field):
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        per_probe = {
            it: (preds, trues)
            for it, preds, trues in zip(
                data[iter_field], data[pred_field], data[true_field]
            )
        }
        raw_by_algo[label].append(per_probe)

    if not raw_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with {pred_field!r} for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py for "
            f"IMOSS-TabFM first."
        )
    return raw_by_algo, n_iterations


def _metric_traces(
    benchmark: str,
    n_iterations: int | None,
    metric: Callable[[float, float], float],
    fields: tuple[str, str, str] = _SUGGESTION_FIELDS,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Turn _load_predicted_true_raw's (predicted, true) pairs into
    dict[label] -> (iterations, mean, std) for an arbitrary per-value
    `metric(predicted, true)` -- e.g. squared error for MSE, signed error
    (predicted - true) for bias. Each probe's per-seed value is the mean of
    `metric` over that probe's values (draws for the suggestion source, the
    whole candidate pool for the candidate source); seeds are then aligned by
    intersecting the set of logged iterations across seeds (probes get
    skipped seed to seed before enough rewarded arms exist for TabFM to fit
    at all) before averaging across seeds.

    Returns dict[label] -> (iterations, mean, std), each 1-D, one entry per
    iteration common to every seed.
    """
    raw_by_algo, n_iterations = _load_predicted_true_raw(benchmark, n_iterations, fields)

    traces = {}
    for label, per_seed_maps in raw_by_algo.items():
        common_iters = sorted(set.intersection(*(set(m) for m in per_seed_maps)))
        per_seed_values = np.array(
            [
                [
                    float(np.mean([metric(p, t) for p, t in zip(*m[it])]))
                    for it in common_iters
                ]
                for m in per_seed_maps
            ]
        )
        traces[label] = (
            np.asarray(common_iters, dtype=float),
            per_seed_values.mean(axis=0),
            per_seed_values.std(axis=0),
        )
    return traces, n_iterations


def _load_suggestion_mse_traces(benchmark: str, n_iterations: int | None):
    return _metric_traces(
        benchmark, n_iterations, lambda p, t: (p - t) ** 2, _SUGGESTION_FIELDS
    )


def _load_candidate_mse_traces(benchmark: str, n_iterations: int | None):
    return _metric_traces(
        benchmark, n_iterations, lambda p, t: (p - t) ** 2, _CANDIDATE_FIELDS
    )


def _load_suggestion_bias_traces(benchmark: str, n_iterations: int | None):
    return _metric_traces(benchmark, n_iterations, lambda p, t: p - t)


def _plot_suggestion_metric_grid(
    load_fn: Callable,
    ylabel: str,
    filename_suffix: str,
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
    log_scale=False,
    zero_line=False,
):
    """Shared plotting body for the suggestion-metric grids (MSE, bias, or
    any future per-draw metric) -- one subplot per benchmark, mean line +
    std band across seeds, using `load_fn(benchmark, n_iterations)` to get
    each benchmark's (iterations, mean, std) traces.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces, _ = load_fn(benchmark, n_iterations)

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

        if zero_line:
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
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
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_{filename_suffix}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_tabfm_suggestion_error_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
    log_scale=True,
):
    """TabFM's suggested-config MSE vs iteration, one subplot per benchmark.

    Each point is the MSE between TabFM's predicted reward (in reward units)
    and the true reward for the configs it actually suggests (the N_SHADOW=10
    shadow-probe picks at that checkpoint -- see
    rf_arm_distribution_experiment.run_single_experiment and
    IMABOTabFM.on_suggestion), logged on the fixed oracle_probe_every
    schedule. This is the picks-only counterpart to
    plot_tabfm_candidate_mse_grid (which averages over the whole pool). Only
    IMOSS-TabFM logs this field.
    """
    _plot_suggestion_metric_grid(
        _load_suggestion_mse_traces,
        "TabFM Suggested-Config MSE",
        "tabfm_suggestion_mse_grid",
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        smoothing_span=smoothing_span,
        log_scale=log_scale,
    )


def plot_tabfm_candidate_mse_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
    log_scale=True,
):
    """TabFM's candidate-pool MSE vs iteration, one subplot per benchmark.

    Each point is the MSE between TabFM's predicted reward (in reward units)
    and the true reward over the *whole* scored candidate pool (all
    n_candidates from the real TabFM fit at that probe -- see
    rf_arm_distribution_experiment.run_single_experiment and
    IMABOTabFM.on_candidates_scored), not just the chosen config. It measures
    TabFM's accuracy across the candidate space, logged on the fixed
    oracle_probe_every schedule. Only IMOSS-TabFM logs this field.
    """
    _plot_suggestion_metric_grid(
        _load_candidate_mse_traces,
        "TabFM Candidate-Pool MSE",
        "tabfm_candidate_mse_grid",
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        smoothing_span=smoothing_span,
        log_scale=log_scale,
    )


def plot_tabfm_suggestion_bias_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
):
    """TabFM's suggested-config signed bias (predicted - true) vs iteration,
    one subplot per benchmark.

    Unlike MSE, this keeps the sign, so a systematic over- vs under-estimate
    is visible rather than folded into a magnitude. In practice it sits near
    zero (a small negative offset -- mild shrinkage toward the training mean,
    since picks are top-of-distribution). A dashed zero line marks perfect
    calibration. Only IMOSS-TabFM logs this field; no log scale since bias
    can be negative.
    """
    _plot_suggestion_metric_grid(
        _load_suggestion_bias_traces,
        "TabFM Suggested-Config Bias (Predicted - True)",
        "tabfm_suggestion_bias_grid",
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        smoothing_span=smoothing_span,
        log_scale=False,
        zero_line=True,
    )


def _load_calibration_traces(
    benchmark: str, n_iterations: int | None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Per-probe cross-seed mean/std of four IMOSS-TabFM quantities, aligned
    on the shared probe schedule: the ensemble-mean predicted reward of the
    picks, the ensemble-max predicted reward (the value suggest_method="max"
    actually ranks by), their true reward, and the mean reward label TabFM
    actually fit on that probe (see
    rf_arm_distribution_experiment.run_single_experiment).

    Returns dict keyed by "Predicted (mean)"/"Predicted (max)"/"True"/
    "Train-label mean" -> (iterations, mean, std) -- the direct test of
    whether it's only the ensemble mean that collapses (a max-vs-mean
    selection artifact) while the max the acquisition acts on tracks true.
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

    # per seed: dict[iteration] -> (pred_mean, pred_max, true_mean, train_mean)
    per_seed_maps: list[dict[int, tuple[float, float, float, float]]] = []
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("tabfm_suggestion_predicted_max_rewards"):
            continue
        per_seed_maps.append(
            {
                it: (
                    float(np.mean(preds)),
                    float(np.mean(pmax)),
                    float(np.mean(trues)),
                    float(np.mean(train)),
                )
                for it, preds, pmax, trues, train in zip(
                    data["tabfm_suggestion_probe_iterations"],
                    data["tabfm_suggestion_predicted_rewards"],
                    data["tabfm_suggestion_predicted_max_rewards"],
                    data["tabfm_suggestion_true_rewards"],
                    data["tabfm_train_rewards"],
                )
            }
        )

    if not per_seed_maps:
        raise FileNotFoundError(
            f"No run checkpoints with tabfm_suggestion_predicted_max_rewards for "
            f"T={n_iterations} in {DATA_DIR} -- run "
            f"experiments/rf_arm_distribution_experiment.py for IMOSS-TabFM first."
        )

    common_iters = sorted(set.intersection(*(set(m) for m in per_seed_maps)))
    iters = np.asarray(common_iters, dtype=float)
    traces = {}
    keys = ("Predicted (mean)", "Predicted (max)", "True", "Train-label mean")
    for j, key in enumerate(keys):
        vals = np.array([[m[it][j] for it in common_iters] for m in per_seed_maps])
        traces[key] = (iters, vals.mean(axis=0), vals.std(axis=0))
    return traces, n_iterations


# Fixed colors for the calibration lines (not algorithm identities, so not
# routed through _style_for): the two prediction readouts vs true vs labels.
_CALIBRATION_STYLE = {
    "Predicted (mean)": ("#009E73", "s"),  # green (ensemble mean)
    "Predicted (max)": ("#CC79A7", "D"),  # pink (ensemble max = the acq value)
    "True": ("#0072B2", "o"),  # blue (ground truth)
    "Train-label mean": ("#D55E00", "^"),  # vermillion (what it fit on)
}


def plot_tabfm_calibration_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
):
    """Predicted-mean, predicted-max, true, and training-label mean reward
    for IMOSS-TabFM's picks, one subplot per benchmark -- the diagnostic for
    why the suggested-config bias grows negative.

    Given the training-label mean does NOT fall (ruling out the
    exploration-dilutes-the-context hypothesis), the remaining question is
    whether it's specifically the ensemble *mean* that collapses -- a
    max-vs-mean selection artifact, since suggest_method="max" ranks by the
    ensemble max and preferentially selects high-disagreement candidates
    whose mean sits far below their max. If "Predicted (max)" (the value the
    acquisition actually acts on) tracks True while "Predicted (mean)"
    collapses, that's the artifact; if the max collapses too, it's genuine
    miscalibration. Only IMOSS-TabFM logs these fields.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces, _ = _load_calibration_traces(benchmark, n_iterations)

        for key, (iters, mean, std) in traces.items():
            mean = _ema(mean, smoothing_span)
            std = _ema(std, smoothing_span)
            color, marker = _CALIBRATION_STYLE[key]
            (line,) = ax.plot(
                iters,
                mean,
                color=color,
                label=key,
                marker=marker,
                markevery=max(1, len(iters) // 8),
                linewidth=2.0,
                markersize=8,
            )
            ax.fill_between(
                iters, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            seen.setdefault(key, line)

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
        "Reward", fontweight="bold", fontsize=adaptive_label_fontsize(axes[0])
    )

    # Preserve the Predicted/True/Train-label order, not _ordered's algorithm
    # ordering (these aren't algorithm identities).
    ordered_keys = [k for k in _CALIBRATION_STYLE if k in seen]
    create_figure_legend(
        fig,
        [seen[k] for k in ordered_keys],
        ordered_keys,
        ncol=len(ordered_keys),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_tabfm_calibration_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


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
    (bottom row), one column per benchmark, per-column x-axis -- combines
    plot_cumulative_regret_grid and plot_suggested_arm_distribution_grid into
    one figure-sized panel instead of two, so both are read together and only
    the bottom row needs an "Iteration" label.

    Two independent legends rather than one shared union: a top legend (full
    algorithm names) for the regret row, and a second legend row -- using
    short oracle names (TPE, TabFM, Random; see _ORACLE_LABELS) -- sitting in
    a real gap between the two subplot rows, since the bottom row is about
    which oracle proposes the next arm, not the full algorithm identity.

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

    style = paper_style(conference=conference, columns=columns, markevery_divisor=10)

    n = len(benchmarks)

    top_labels = _ordered(set(regret_algorithms))
    bottom_labels = _ordered(set(oracle_algorithms))
    top_ncol = style.legend_ncol(len(top_labels))
    bottom_ncol = style.legend_ncol(len(bottom_labels))
    top_rows = style.n_legend_rows(len(top_labels))
    bottom_rows = style.n_legend_rows(len(bottom_labels))

    subplot_row_in = 1.4
    top_band_in = 0.16 * top_rows + 0.12
    mid_gap_in = 0.16 * bottom_rows + 0.16
    height_in = top_band_in + 2 * subplot_row_in + mid_gap_in

    fig, axes_all = plt.subplots(
        3,
        n,
        figsize=(style.width_in, height_in),
        gridspec_kw={"height_ratios": [subplot_row_in, mid_gap_in, subplot_row_in]},
    )
    if n == 1:
        axes_all = axes_all.reshape(3, 1)
    axes_top, axes_mid, axes_bottom = axes_all[0], axes_all[1], axes_all[2]
    for ax in axes_mid:
        ax.axis("off")
    for j in range(n):
        axes_bottom[j].sharex(axes_top[j])

    seen_top: dict = {}

    for ax, benchmark in zip(axes_top, benchmarks):
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
            seen_top.setdefault(algo, line)

        ax.set_title(
            _bench_title(benchmark),
            fontweight="bold",
            fontsize=style.title_fontsize,
            pad=4,
        )
        style.style_axis(ax)

    axes_top[0].set_ylabel(
        "Cumulative\nRegret", fontweight="bold", fontsize=style.label_fontsize
    )

    seen_bottom: dict = {}

    for idx, (ax, benchmark) in enumerate(zip(axes_bottom, benchmarks)):
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
            seen_bottom.setdefault(algo, line)

        if idx == n // 2:  # Middle subplot
            ax.set_xlabel("Iteration", fontweight="bold", fontsize=style.label_fontsize)
        style.style_axis(ax)

    axes_bottom[0].set_ylabel(
        "Oracle Proposal\nQuality", fontweight="bold", fontsize=style.label_fontsize
    )

    for ax in list(axes_top) + list(axes_bottom):
        ax.label_outer()

    # Top legend: full algorithm names, for the cumulative-regret row.
    ordered_top = _ordered(seen_top.keys())
    create_figure_legend(
        fig,
        [seen_top[a] for a in ordered_top],
        ordered_top,
        ncol=top_ncol,
        bbox_y=1.0 if top_rows == 1 else 1.0 + 0.03 * (top_rows - 1),
        fontsize=style.legend_fontsize,
    )

    plt.tight_layout(
        rect=[0, 0, 1, (2 * subplot_row_in + mid_gap_in) / height_in],
        h_pad=0.6,
        w_pad=0.6,
    )

    # Second legend: short oracle names (TPE, TabFM, Random), centered in the
    # gap between the two subplot rows using that row's actual post-layout
    # position -- so it can't overlap either panel regardless of how
    # tight_layout redistributes space to fit tick/axis labels.
    mid_pos = axes_mid[0].get_position()
    ordered_bottom = _ordered(seen_bottom.keys())
    create_figure_legend(
        fig,
        [seen_bottom[a] for a in ordered_bottom],
        [_ORACLE_LABELS.get(a, a) for a in ordered_bottom],
        ncol=bottom_ncol,
        bbox_y=mid_pos.y0 + mid_pos.height / 2,
        loc="center",
        fontsize=style.legend_fontsize,
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

    print("Generating TabFM suggested-config MSE grid...")
    plot_tabfm_suggestion_error_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating TabFM candidate-pool MSE grid...")
    plot_tabfm_candidate_mse_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating TabFM suggested-config bias grid...")
    plot_tabfm_suggestion_bias_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )

    print("Generating TabFM calibration grid (predicted vs true vs train-label)...")
    plot_tabfm_calibration_grid(
        benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    )
