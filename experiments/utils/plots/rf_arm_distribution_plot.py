"""Plots for the per-iteration suggested-arm-distribution experiment
(experiments/rf_arm_distribution_experiment.py).
"""

import json
import re
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils.plots.plot_configs import (
    _PRETTY_LABELS,
    _bench_title,
    _ema,
    _ordered,
    _style_for,
    adaptive_label_fontsize,
    create_figure_legend,
    display_name,
    get_algorithm_color,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"
DATA_DIR = RESULTS_DIR / "hpo_finite_arm_distribution"

set_research_style()

_IMOSS_FAMILY = ["IMOSS-Random", "IMOSS-TPE", "IMOSS-mutate-KLxTPE", "IMOSS-TabFM"]
ALL = _IMOSS_FAMILY + ["UCB-AIR"]

_ORACLE_LABELS = {
    "IMOSS-Random": "Random",
    "IMOSS-TPE": "TPE",
    "IMOSS-TPE-univ": "TPE-univ",
    "IMOSS-TabFM": "TabFM",
    "IMOSS-TabPFN": "TabPFN",
    "IMOSS-mutate-KLxTPE": "mutate-KLxPE",
}

# Per-series linestyle overrides (color+marker come from algorithm_style):
# the univariate-TPE variant shares IMOSS-TPE's orange, so it is dashed.
_SERIES_LINESTYLE = {"IMOSS-TPE-univ": "--"}


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
    band="ci95",
):
    """Cumulative regret vs iteration, one subplot per benchmark, side by side.

    Built from the per-pull noiseless regret (`regrets`) logged in this
    experiment's own per-run JSONs directly (no aggregated CSV exists for
    this experiment -- see rf_arm_distribution_experiment.py).

    Uncertainty band (``band``): ``"ci95"`` (default) draws a 95% CI of the
    mean across seeds, ``"sd"`` draws +/-1 across-seed standard deviation, and
    ``None`` reproduces the old band-less figure.

    The band is computed ACROSS SEEDS on each seed's own cumulative curve --
    ``cumsum`` first, then take the spread over runs -- never from the
    within-run dispersion of the per-round regret. That distinction matters:
    an earlier version of this function omitted the band on the grounds that
    "the cumulative mean grows ~O(t) while the cumulative std of a sum grows
    only ~O(sqrt(t))", which is the behaviour of a sum of *independent*
    per-round terms within one run. It is not what a band over seeds shows.
    Seeds differ in which arms enter the active set, so their cumulative
    curves diverge roughly linearly and the across-seed spread stays a roughly
    constant *fraction* of the total. Measured on the stored runs at t=5000 it
    is 9-21% of the mean depending on task and method (UCB-AIR on `segment`:
    412.6 +- 86.7, i.e. 21%), not "a couple percent" -- large enough that
    several method gaps in this figure are within it, so the band has to be
    shown for the comparison to be read honestly.

    Seed counts may differ per method (e.g. the surrogate oracles are
    expensive to rerun), so the resolved ``n`` is appended to each legend
    entry rather than assumed uniform.
    """
    algorithms = algorithms if algorithms is not None else _IMOSS_FAMILY

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    seed_counts: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces_by_algo, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        labels = _ordered([a for a in algorithms if a in traces_by_algo])

        for algo in labels:
            runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            # Per-seed cumulative curves, then aggregate across seeds.
            cumulative_runs = np.cumsum(runs, axis=1)
            cumulative_mean = cumulative_runs.mean(axis=0)
            n_runs = cumulative_runs.shape[0]
            seed_counts[algo] = min(seed_counts.get(algo, n_runs), n_runs)

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
            if band is not None and n_runs > 1:
                sd = cumulative_runs.std(axis=0, ddof=1)
                half = sd if band == "sd" else 1.96 * sd / np.sqrt(n_runs)
                ax.fill_between(
                    iters,
                    cumulative_mean - half,
                    cumulative_mean + half,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
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
    # Seed counts can differ per method, so state each one rather than putting
    # a single "n=..." in the caption and implying they match.
    if band is not None and len(set(seed_counts.values())) > 1:
        legend_labels = [f"{a} (n={seed_counts[a]})" for a in ordered_labels]
    else:
        legend_labels = list(ordered_labels)
    create_figure_legend(
        fig,
        [seen[a] for a in ordered_labels],
        legend_labels,
        ncol=len(ordered_labels),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        tag = "_".join(benchmarks)
        suffix = "" if band is None else f"_{band}"
        out_path = (
            RESULTS_DIR
            / "paper_plots"
            / f"{tag}_cumulative_regret_grid_arm_distribution{suffix}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.close()


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

    plt.close()


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
    raw_by_algo, n_iterations = _load_predicted_true_raw(
        benchmark, n_iterations, fields
    )

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


def _candidate_topk_mse_traces(
    benchmark: str,
    n_iterations: int | None,
    frac: float = 0.1,
    by: str = "true",
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Candidate-pool MSE restricted to the top `frac` of each probe's pool,
    ranked by `by` ("true" reward or "predicted" reward).

    For SMBO the surrogate only needs accuracy where it exploits, not on the
    (mostly bad) bulk of a uniform-random pool; this keeps only the top
    fraction. `by="true"` -> accuracy on the genuinely good arms; `by="predicted"`
    -> accuracy on what TabFM *would* exploit (the convergence-relevant read).
    Per probe: sort the pool by `by`, keep the top ceil(frac * pool) points,
    MSE over that subset; seeds aligned by iteration-intersection then averaged.
    """
    raw_by_algo, n_iterations = _load_predicted_true_raw(
        benchmark, n_iterations, _CANDIDATE_FIELDS
    )

    traces = {}
    for label, per_seed_maps in raw_by_algo.items():
        common_iters = sorted(set.intersection(*(set(m) for m in per_seed_maps)))
        per_seed_values = []
        for m in per_seed_maps:
            row = []
            for it in common_iters:
                preds, trues = np.asarray(m[it][0]), np.asarray(m[it][1])
                k = max(1, int(round(len(preds) * frac)))
                rank_by = trues if by == "true" else preds
                top = np.argsort(rank_by)[::-1][:k]
                row.append(float(np.mean((preds[top] - trues[top]) ** 2)))
            per_seed_values.append(row)
        arr = np.array(per_seed_values)
        traces[label] = (
            np.asarray(common_iters, dtype=float),
            arr.mean(axis=0),
            arr.std(axis=0),
        )
    return traces, n_iterations


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
    algorithms=None,
):
    """Shared plotting body for the suggestion-metric grids (MSE, bias, or
    any future per-draw metric) -- one subplot per benchmark, mean line +
    std band across seeds, using `load_fn(benchmark, n_iterations)` to get
    each benchmark's (iterations, mean, std) traces.

    ``algorithms`` restricts which loaded series are drawn (default: the
    canonical algorithms, via _ordered). The loaders glob every run file
    whose JSON has the metric fields, which includes the TabPFN
    acquisition-sweep variants (imoss_tabpfn_q0.841, ..._ucb_kappa1.96, ...);
    without the filter those variants render as unlabeled default-styled
    curves on top of the main series (the legend already filtered them, so
    they appeared as anonymous black lines).
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        traces, _ = load_fn(benchmark, n_iterations)
        labels = (
            [a for a in algorithms if a in traces]
            if algorithms is not None
            else _ordered(traces.keys())
        )

        for algo in labels:
            iters, mean, std = traces[algo]
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

    plt.close()


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
    TabFMOracle.on_suggestion), logged on the fixed oracle_probe_every
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
    TabFMOracle.on_candidates_scored), not just the chosen config. It measures
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


def plot_tabfm_candidate_topk_mse_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    smoothing_span=1,
    log_scale=True,
    frac=0.1,
    by="true",
):
    """TabFM's candidate MSE restricted to the top-`frac` of each probe's pool
    (ranked by `by`), one subplot per benchmark.

    The whole-pool MSE (plot_tabfm_candidate_mse_grid) is dominated by the
    bulk of mediocre uniform-random candidates, which an SMBO surrogate never
    needs to value precisely. This keeps only the top fraction -- `by="true"`
    for accuracy on the genuinely good arms, `by="predicted"` for accuracy on
    what TabFM would actually exploit. See _candidate_topk_mse_traces. Only
    IMOSS-TabFM logs this field.
    """
    pct = int(round(frac * 100))
    _plot_suggestion_metric_grid(
        partial(_candidate_topk_mse_traces, frac=frac, by=by),
        f"TabFM Top-{pct}% Candidate MSE (by {by})",
        f"tabfm_candidate_top{pct}pct_{by}_mse_grid",
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

    plt.close()


def _load_pull_counts(
    benchmark: str, n_iterations: int | None, run: int | str | None = None
) -> tuple[dict[str, tuple[list[str], dict[tuple, float]]], int]:
    """Read the real-trajectory `suggestion_counts` from the per-run JSONs.

    `run`:
      - int k      -> use only seed k (peaked per-seed footprint).
      - "all"      -> average pull counts per config over all seeds.
      - None       -> sum pull counts per config over all seeds.
    Pooling seeds ("all"/None) smears out the per-seed concentration (each
    seed's Bernoulli noise converges it to a *different* arm), so a single
    seed shows the true peaked footprint. "all" and None differ only by the
    /n_seeds factor -- identical once marker area is normalized per panel.

    Returns dict[label] -> (param_names, {config_key_tuple: pulls}). Counts
    are over every opt.suggest() (explore + exploit), i.e. the actual
    footprint of the algorithm on the search space -- see
    rf_arm_distribution_experiment.run_single_experiment.
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

    counts_by_algo: dict[str, Counter] = defaultdict(Counter)
    pnames_by_algo: dict[str, list[str]] = {}
    seeds_by_algo: dict[str, set] = defaultdict(set)
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        if not match:
            continue
        if isinstance(run, int) and int(match.group(3)) != run:
            continue
        slug = match.group(1)
        with open(path) as f:
            data = json.load(f)
        if not data.get("suggestion_counts"):
            continue
        label = _PRETTY_LABELS.get(slug, slug.replace("_", " ").title())
        for key_list, count in data["suggestion_counts"]:
            counts_by_algo[label][tuple(key_list)] += count
        pnames_by_algo[label] = data["param_names"]
        seeds_by_algo[label].add(int(match.group(3)))

    if not counts_by_algo:
        raise FileNotFoundError(
            f"No run checkpoints with suggestion_counts for T={n_iterations} in "
            f"{DATA_DIR} -- run experiments/rf_arm_distribution_experiment.py first."
        )
    if run == "all":  # average per config over the seeds that contributed
        for label, counts in counts_by_algo.items():
            n_seeds = max(1, len(seeds_by_algo[label]))
            for key in counts:
                counts[key] /= n_seeds
    return (
        {
            label: (pnames_by_algo[label], dict(counts_by_algo[label]))
            for label in counts_by_algo
        },
        n_iterations,
    )


def plot_arm_pulls_landscape_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=("IMOSS-TPE", "IMOSS-TabFM"),
    max_marker=900,
    min_marker=0,
    agg="mean",
    run=0,
):
    """Where each oracle spends its pulls, overlaid on the reward landscape.

    Rows = oracle, columns = benchmark. Background: val_acc over
    (max_depth, max_features), marginalizing the other two hyperparameters by
    `agg`:
      - "mean" (default): typical quality at that cell -- reveals the true
        landscape structure (e.g. credit-g's isolated high-depth+high-features
        corner). A bright cell doesn't guarantee the specific pulled arm is
        good, since it averages over the hidden dims.
      - "max": best arm achievable at that cell -- a bright cell means a good
        arm exists there (unambiguous for the pull-overlay reading), but it
        flattens structure (credit-g's corner looks like a plain depth ridge).
    Overlay: each visited config as a filled crimson circle at its
    (max_depth, max_features) cell, marker area strictly proportional to
    total pulls landing in that cell (real-trajectory suggestion_counts,
    summed over seeds; explore + exploit), scaled to `max_marker`. With the
    default `min_marker=0` the area is a true proportion, so lightly-pulled
    cells (opened once during exploration, then abandoned) shrink to nearly
    nothing and only genuine concentration shows -- this avoids faking broad
    coverage, since the explore phase is forced to open ~sqrt(T) distinct
    arms regardless of the oracle. Raise `min_marker` only if you want a
    visibility floor. Shows the footprint difference -- does TabFM
    concentrate on the bright region more tightly than TPE.
    """
    import pandas as pd

    from experiments.benchmarks.rf_tabular_bandit import (
        PARAM_NAMES,
        RFTabularFiniteBenchmark,
    )

    def _reward_grid(bm_id: int) -> pd.DataFrame:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        df = pd.DataFrame(
            [
                {**dict(zip(PARAM_NAMES, key)), "val_acc": value}
                for key, value in bench.lookup.items()
            ]
        )
        return (
            df.groupby(["max_depth", "max_features"])
            .val_acc.agg(agg)
            .unstack("max_features")
        )

    n_row, n_col = len(algorithms), len(benchmarks)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(6 * n_col, 5 * n_row), squeeze=False
    )

    for c, benchmark in enumerate(benchmarks):
        bm_id = int(benchmark[2:])
        pivot = _reward_grid(bm_id)  # index=max_depth, cols=max_features (sorted)
        depth_vals = np.array(pivot.index, dtype=float)
        feat_vals = np.array(pivot.columns, dtype=float)
        counts_by_algo, _ = _load_pull_counts(benchmark, n_iterations, run=run)

        for r, algo in enumerate(algorithms):
            ax = axes[r][c]
            im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")

            if algo in counts_by_algo:
                pnames, counts = counts_by_algo[algo]
                di, fi = pnames.index("max_depth"), pnames.index("max_features")
                # Aggregate pulls onto (max_depth, max_features) cells.
                cell: dict[tuple[int, int], float] = defaultdict(float)
                for key, cnt in counts.items():
                    yi = int(np.argmin(np.abs(depth_vals - key[di])))
                    xi = int(np.argmin(np.abs(feat_vals - key[fi])))
                    cell[(yi, xi)] += cnt
                if cell:
                    maxc = max(cell.values())
                    ys = [yi for (yi, _) in cell]
                    xs = [xi for (_, xi) in cell]
                    sizes = [
                        min_marker + (max_marker - min_marker) * cell[(yi, xi)] / maxc
                        for yi, xi in cell
                    ]
                    ax.scatter(
                        xs,
                        ys,
                        s=sizes,
                        c="crimson",
                        alpha=0.6,
                        edgecolors="white",
                        linewidths=1.0,
                    )

            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(
                [f"{v:.2g}" for v in pivot.columns],
                rotation=45,
                ha="right",
                fontsize=14,
            )
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels([f"{int(v)}" for v in pivot.index], fontsize=14)
            if r == 0:
                ax.set_title(
                    _bench_title(benchmark), fontweight="bold", fontsize=20, pad=8
                )
            if r == n_row - 1:
                ax.set_xlabel("max_features", fontweight="bold", fontsize=16)
            if c == 0:
                ax.set_ylabel(f"{algo}\nmax_depth", fontweight="bold", fontsize=16)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("val_acc", fontsize=12)
            cbar.ax.tick_params(labelsize=11)

    seed_note = f"seed {run}" if run is not None else "summed over seeds"
    fig.text(
        0.5,
        0.005,
        f"Background: {agg} val_acc per (max_depth, max_features) cell. "
        f"Marker area strictly proportional to total pulls in that cell "
        f"({seed_note}; explore + exploit)",
        ha="center",
        fontsize=13,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if save_fig:
        tag = "_".join(benchmarks)
        suffix = f"_run{run}" if run is not None else ""
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_arm_pulls_landscape_grid{suffix}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.close()


def _candidate_cell_mse_grid(benchmark, n_iterations, run=None, agg="mean"):
    """Per (max_depth, max_features) cell MSE of TabFM's candidate-pool
    predictions vs true reward, from tabfm_candidate_configs/predicted/true
    (only IMOSS-TabFM logs these). Reindexed to the benchmark's full depth x
    features grid (NaN where no candidate landed). `run=k` (int) uses seed k
    only; `run=None` or "all" pools every seed (smoother estimate). `agg` controls how
    per-cell squared errors are combined -- "mean" (default) for the typical
    error in a cell, "max" to instead surface the single worst prediction
    per cell (any pandas groupby-agg string works, e.g. "median").

    Returns (pivot DataFrame indexed by max_depth, columns max_features), n_iterations.
    """
    import pandas as pd

    from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark

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
                f"experiments/rf_arm_distribution_experiment.py first."
            )
        n_iterations = max(available)

    rows = []
    for path in sorted(DATA_DIR.glob(f"{benchmark}_*_{n_iterations}iters_run*.json")):
        match = run_pattern.match(path.name)
        # int -> that seed only; None/"all" -> pool every seed.
        if not match or (isinstance(run, int) and int(match.group(3)) != run):
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("tabfm_candidate_configs"):
            continue
        pn = data["param_names"]
        di, fi = pn.index("max_depth"), pn.index("max_features")
        for cfgs, preds, trues in zip(
            data["tabfm_candidate_configs"],
            data["tabfm_candidate_predicted_rewards"],
            data["tabfm_candidate_true_rewards"],
        ):
            for cfg, p, t in zip(cfgs, preds, trues):
                rows.append((cfg[di], cfg[fi], (p - t) ** 2))

    if not rows:
        raise FileNotFoundError(
            f"No run checkpoints with tabfm_candidate_configs for T={n_iterations} "
            f"in {DATA_DIR} -- rerun experiments/rf_arm_distribution_experiment.py "
            f"for IMOSS-TabFM (this field was added later)."
        )

    df = pd.DataFrame(rows, columns=["max_depth", "max_features", "se"])
    pivot = (
        df.groupby(["max_depth", "max_features"]).se.agg(agg).unstack("max_features")
    )
    bench = RFTabularFiniteBenchmark(bm_id=int(benchmark[2:]))
    pivot = pivot.reindex(
        index=sorted(bench.axes["max_depth"]),
        columns=sorted(bench.axes["max_features"]),
    )
    return pivot, n_iterations


def plot_tabfm_mse_landscape_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    max_marker=900,
    min_marker=0,
    run="all",
    agg="mean",
):
    """TabFM surrogate error over the (max_depth, max_features) landscape, with
    its pull footprint overlaid -- 'where is TabFM bad, and does it pull there?'

    Background = per-cell MSE of TabFM's candidate predictions vs true reward
    (see _candidate_cell_mse_grid; bright = high error) -- `agg="mean"`
    (default) shows the typical error per cell, `agg="max"` the single worst
    prediction. Overlay = crimson pull markers (area strictly proportional to
    pulls). `run` selects BOTH the MSE background and the pull footprint: an
    int for a single seed (peaked footprint + that seed's error map, noisier
    ~100 candidates/cell), "all" to average pulls / pool the error map over
    seeds (smoother, ~1000 candidates/cell), or None to sum. IMOSS-TabFM only.
    """
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for c, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        pivot, _ = _candidate_cell_mse_grid(benchmark, n_iterations, run=run, agg=agg)
        depth_vals = np.array(pivot.index, dtype=float)
        feat_vals = np.array(pivot.columns, dtype=float)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("lightgray")
        im = ax.imshow(
            np.ma.masked_invalid(pivot.values),
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )

        counts_by_algo, _ = _load_pull_counts(benchmark, n_iterations, run=run)
        if "IMOSS-TabFM" in counts_by_algo:
            pnames, counts = counts_by_algo["IMOSS-TabFM"]
            di, fi = pnames.index("max_depth"), pnames.index("max_features")
            cell: dict[tuple[int, int], float] = defaultdict(float)
            for key, cnt in counts.items():
                yi = int(np.argmin(np.abs(depth_vals - key[di])))
                xi = int(np.argmin(np.abs(feat_vals - key[fi])))
                cell[(yi, xi)] += cnt
            if cell:
                maxc = max(cell.values())
                xs = [xi for (_, xi) in cell]
                ys = [yi for (yi, _) in cell]
                sizes = [
                    min_marker + (max_marker - min_marker) * cell[(yi, xi)] / maxc
                    for yi, xi in cell
                ]
                ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    c="crimson",
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=1.0,
                )

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(
            [f"{v:.2g}" for v in pivot.columns], rotation=45, ha="right", fontsize=14
        )
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([f"{int(v)}" for v in pivot.index], fontsize=14)
        ax.set_title(_bench_title(benchmark), fontweight="bold", fontsize=20, pad=8)
        ax.set_xlabel("max_features", fontweight="bold", fontsize=16)
        if c == 0:
            ax.set_ylabel("IMOSS-TabFM\nmax_depth", fontweight="bold", fontsize=16)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"TabFM {agg} MSE", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

    seed_note = (
        "averaged over seeds"
        if run == "all"
        else ("summed over seeds" if run is None else f"seed {run}")
    )
    fig.text(
        0.5,
        0.005,
        f"Background: TabFM candidate-prediction {agg} MSE per (max_depth, max_features) "
        f"cell (bright = worse). Crimson markers = pulls, area proportional to pulls. "
        f"Both {seed_note}.",
        ha="center",
        fontsize=12,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_fig:
        tag = "_".join(benchmarks)
        suffix = f"_run{run}" if run is not None else ""
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_tabfm_mse_landscape_grid{suffix}.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.close()


def plot_arm_pulls_embedding_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=("IMOSS-TPE", "IMOSS-TabFM"),
    max_marker=800,
    min_marker=30,
    levels=14,
):
    """Where each oracle spends its pulls, over the reward landscape on a 2D
    PCA embedding of the full 4D config space -- no marginalization.

    Unlike plot_arm_pulls_landscape_grid (which projects onto only
    max_depth x max_features and marginalizes the other two hyperparameters),
    this embeds all four standardized hyperparameters to 2D via PCA. In each
    panel the background is the true-reward field over that embedding (filled
    `tricontourf` contours of val_acc, so the high-reward region reads as a
    contiguous zone), and pulled arms are overlaid as crimson markers whose
    area scales with total pulls (real-trajectory suggestion_counts, summed
    over seeds; explore + exploit) between `min_marker` and `max_marker`. A
    well-behaved oracle's big markers sit over the bright region. Rows =
    oracle, columns = benchmark.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark

    n_row, n_col = len(algorithms), len(benchmarks)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(6 * n_col, 5 * n_row), squeeze=False
    )

    for c, benchmark in enumerate(benchmarks):
        bm_id = int(benchmark[2:])
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        arms = list(bench.lookup.keys())  # config tuples, PARAM_NAMES order
        rewards = np.array([bench.lookup[a] for a in arms], dtype=float)
        emb = PCA(n_components=2).fit_transform(
            StandardScaler().fit_transform(np.array(arms, dtype=float))
        )
        # Round keys so JSON-roundtripped count keys match the lookup tuples.
        arm_index = {tuple(round(v, 6) for v in a): i for i, a in enumerate(arms)}
        counts_by_algo, _ = _load_pull_counts(benchmark, n_iterations)

        for r, algo in enumerate(algorithms):
            ax = axes[r][c]
            # Background: true-reward field over the embedding, so the
            # high-reward region shows as a contiguous zone.
            tcf = ax.tricontourf(
                emb[:, 0], emb[:, 1], rewards, levels=levels, cmap="viridis"
            )

            pull = np.zeros(len(arms))
            if algo in counts_by_algo:
                _, counts = counts_by_algo[algo]
                for key, cnt in counts.items():
                    idx = arm_index.get(tuple(round(v, 6) for v in key))
                    if idx is not None:
                        pull[idx] += cnt
            pulled = np.where(pull > 0)[0]
            if len(pulled):
                maxp = pull[pulled].max()
                order = pulled[np.argsort(pull[pulled])]  # big markers on top
                sizes = min_marker + (max_marker - min_marker) * pull[order] / maxp
                ax.scatter(
                    emb[order, 0],
                    emb[order, 1],
                    s=sizes,
                    c="crimson",
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.8,
                )

            if r == 0:
                ax.set_title(
                    _bench_title(benchmark), fontweight="bold", fontsize=20, pad=8
                )
            if r == n_row - 1:
                ax.set_xlabel("PC1", fontweight="bold", fontsize=16)
            if c == 0:
                ax.set_ylabel(f"{algo}\nPC2", fontweight="bold", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("val_acc", fontsize=12)
            cbar.ax.tick_params(labelsize=11)

    fig.text(
        0.5,
        0.005,
        "Background: true-reward field over the 2D PCA embedding of the 4 "
        "hyperparameters; crimson markers = pulled arms, area scales with total "
        "pulls (summed over seeds; explore + exploit)",
        ha="center",
        fontsize=13,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_arm_pulls_embedding_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.close()


def plot_arm_pulls_parallel_coords_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    algorithms=("IMOSS-TPE", "IMOSS-TabFM"),
    lw_max=3.5,
    alpha_min=0.10,
):
    """Where each oracle spends its pulls, in full 4D via parallel coordinates.

    Four vertical axes (one per hyperparameter, in PARAM_NAMES order, each
    normalized to its own [min, max]); each pulled arm is a polyline through
    its four values, colored by true reward (val_acc) and with opacity/width
    scaling with total pulls (real-trajectory suggestion_counts, summed over
    seeds; explore + exploit). Nothing is marginalized -- unlike the 2D
    projection, this shows exactly which 4D combinations the oracle
    concentrates on (bold bright lines = heavily-pulled good arms). Rows =
    oracle, columns = benchmark.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from experiments.benchmarks.rf_tabular_bandit import (
        PARAM_NAMES,
        RFTabularFiniteBenchmark,
    )

    params = list(PARAM_NAMES)
    cmap = plt.get_cmap("viridis")
    short = {
        "max_depth": "max_depth",
        "max_features": "max_feat",
        "min_samples_leaf": "min_leaf",
        "min_samples_split": "min_split",
    }

    n_row, n_col = len(algorithms), len(benchmarks)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(7 * n_col, 5 * n_row), squeeze=False
    )

    for c, benchmark in enumerate(benchmarks):
        bm_id = int(benchmark[2:])
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        pmin = {p: min(bench.axes[p]) for p in params}
        pmax = {p: max(bench.axes[p]) for p in params}
        rewards_all = np.array(list(bench.lookup.values()), dtype=float)
        norm = Normalize(vmin=rewards_all.min(), vmax=rewards_all.max())
        reward_lookup = {
            tuple(round(v, 6) for v in k): val for k, val in bench.lookup.items()
        }
        counts_by_algo, _ = _load_pull_counts(benchmark, n_iterations)

        for r, algo in enumerate(algorithms):
            ax = axes[r][c]
            segs, colors, lws = [], [], []
            if algo in counts_by_algo:
                pnames, counts = counts_by_algo[algo]
                pos = [pnames.index(p) for p in params]  # param -> key position
                maxc = max(counts.values()) if counts else 1
                for key, cnt in sorted(counts.items(), key=lambda kv: kv[1]):
                    reward = reward_lookup.get(tuple(round(v, 6) for v in key))
                    if reward is None:
                        continue
                    ys = [
                        (key[pos[j]] - pmin[p]) / ((pmax[p] - pmin[p]) or 1)
                        for j, p in enumerate(params)
                    ]
                    segs.append(np.column_stack([np.arange(len(params)), ys]))
                    frac = cnt / maxc
                    rgba = list(cmap(norm(reward)))
                    rgba[3] = alpha_min + (1 - alpha_min) * frac
                    colors.append(rgba)
                    lws.append(0.4 + lw_max * frac)
            if segs:
                ax.add_collection(LineCollection(segs, colors=colors, linewidths=lws))

            for xj in range(len(params)):
                ax.axvline(xj, color="0.75", linewidth=0.8, zorder=0)
                ax.text(
                    xj,
                    -0.03,
                    f"{pmin[params[xj]]:.2g}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="0.4",
                )
                ax.text(
                    xj,
                    1.03,
                    f"{pmax[params[xj]]:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.4",
                )
            ax.set_xlim(-0.3, len(params) - 0.7)
            ax.set_ylim(-0.08, 1.08)
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels(
                [short[p] for p in params], rotation=20, ha="right", fontsize=11
            )
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(
                    _bench_title(benchmark), fontweight="bold", fontsize=18, pad=10
                )
            if c == 0:
                ax.set_ylabel(algo, fontweight="bold", fontsize=15)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("val_acc", fontsize=11)
            cbar.ax.tick_params(labelsize=10)

    fig.text(
        0.5,
        0.005,
        "Parallel coordinates over the 4 hyperparameters (each normalized to "
        "its own range); each line = a pulled arm, color = true reward, "
        "opacity/width scale with total pulls (summed over seeds; explore + exploit)",
        ha="center",
        fontsize=12,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = (
            RESULTS_DIR / "paper_plots" / f"{tag}_arm_pulls_parallel_coords_grid.pdf"
        )
        save_figure(out_path, bbox_inches="tight")

    plt.close()


def plot_cumulative_regret_gap(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    reference="IMOSS-TPE",
    compared="IMOSS-TabFM",
):
    """Cumulative-regret gap (`reference` - `compared`) vs iteration, one line
    per benchmark, on a single axes.

    Positive => `compared` has the lower cumulative regret (is winning);
    negative => `reference` wins. Because both share the same MOSS exploit
    machinery, the gap isolates the oracle's arm-opening quality over time --
    where a line pulls away from zero shows *when* each oracle earns its
    advantage (e.g. TabFM's ridge-task gains accrue late in the exploit tail,
    while its credit-g deficit opens early in the explore phase). The area
    between a curve and zero is the accumulated regret difference.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, benchmark in enumerate(benchmarks):
        traces, _ = _load_trace_field(benchmark, n_iterations, "regrets")
        if reference not in traces or compared not in traces:
            continue
        ref = np.cumsum(np.vstack(traces[reference]).mean(axis=0))
        cmp_ = np.cumsum(np.vstack(traces[compared]).mean(axis=0))
        gap = ref - cmp_
        iters = np.arange(1, len(gap) + 1)
        color = get_algorithm_color(i)
        ax.plot(iters, gap, color=color, linewidth=2.5, label=_bench_title(benchmark))
        ax.fill_between(
            iters, 0, gap, where=gap >= 0, color=color, alpha=0.10, linewidth=0
        )
        ax.fill_between(
            iters, 0, gap, where=gap < 0, color=color, alpha=0.10, linewidth=0
        )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Iteration", fontweight="bold", fontsize=16)
    ax.set_ylabel(
        f"Cumulative regret gap\n({reference} − {compared})",
        fontweight="bold",
        fontsize=15,
    )
    # Which side is which, annotated in axes-fraction coords.
    ax.text(
        0.015,
        0.97,
        f"↑ {compared} better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="0.3",
    )
    ax.text(
        0.015,
        0.03,
        f"↓ {reference} better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="0.3",
    )
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="best")

    plt.tight_layout()

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_cumulative_regret_gap.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.close()


def plot_regret_and_oracle_grid(
    benchmarks=("rf146822", "rf31", "rf167120"),
    n_iterations=None,
    save_fig=False,
    regret_algorithms=None,
    oracle_algorithms=None,
    smoothing_span=1,
    columns=2,
    conference="aaai",
    out_name="regret_and_oracle_grid",
    final_errorbar=None,
):
    """Paper figure: cumulative regret (top row) and oracle proposal quality
    (bottom row), one column per benchmark, per-column x-axis -- combines
    plot_cumulative_regret_grid and plot_suggested_arm_distribution_grid into
    one figure-sized panel instead of two, so both are read together and only
    the bottom row needs an "Iteration" label.

    ``final_errorbar`` puts the across-seed uncertainty of the cumulative
    regret on the figure as one interval bar per method at the final round
    only (``"ci95"`` for a 95% CI of the mean across seeds, ``"sd"`` for +/-1
    across-seed standard deviation, ``None`` for none) -- computed from each
    seed's own cumulative curve, like plot_cumulative_regret_grid's band. A
    full-length band would bury the five closely spaced curves, and methods
    can end at nearly identical means (Hier-MAB vs IMOSS-TabPFN on segment),
    so the bars are dodged into a short strip just past t=T, one x-slot per
    method in legend order, each sitting level with its own final mean.

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

        for i_algo, algo in enumerate(labels):
            runs = np.vstack(traces_by_algo[algo])  # n_runs x n_iterations
            iters = np.arange(1, runs.shape[1] + 1)
            cumulative_runs = np.cumsum(runs, axis=1)
            cumulative_mean = cumulative_runs.mean(axis=0)
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
                linestyle=_SERIES_LINESTYLE.get(algo, "-"),
            )
            if final_errorbar is not None and runs.shape[0] > 1:
                sd = float(cumulative_runs[:, -1].std(ddof=1))
                half = (
                    sd if final_errorbar == "sd" else 1.96 * sd / np.sqrt(runs.shape[0])
                )
                x_bar = runs.shape[1] * (1.035 + 0.035 * i_algo)
                ax.errorbar(
                    x_bar,
                    cumulative_mean[-1],
                    yerr=half,
                    color=color,
                    fmt="none",
                    elinewidth=1.0,
                    capsize=1.8,
                    capthick=style.capthick,
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
                linestyle=_SERIES_LINESTYLE.get(algo, "-"),
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

    if final_errorbar is not None:
        # The dodged final-round bars extend the data range past t=T, which
        # would otherwise grow the shared x tick labels beyond the horizon
        # (a "6000" tick for a T=5000 run); pin the ticks to the horizon.
        resolved_T = int(max(line.get_xdata()[-1] for line in seen_top.values()))
        for j in range(n):
            axes_bottom[j].set_xticks(np.arange(0, resolved_T + 1, 1000))

    for ax in list(axes_top) + list(axes_bottom):
        ax.label_outer()

    # Top legend: full algorithm names, for the cumulative-regret row. Rendered
    # via display_name so the printed label comes from the one shared override
    # map (plot_configs.DISPLAY_NAME_OVERRIDES) instead of the raw identity --
    # the identity strings stay untouched here, since _ordered/_style_for/
    # _PRETTY_LABELS/_CANONICAL_ORDER all match on them.
    #
    # handlelength/columnspacing/handletextpad below the matplotlib defaults
    # (2.0/2.0/0.8): with a foundation series overlaid (six entries, see
    # rf_arm_distribution_experiment.make_plots' foundation_series) the legend
    # at default spacing renders 7.34in wide in a 7.0in figure, and
    # bbox_inches="tight" then grows the saved PDF to ~7.58in -- shrinking every
    # font ~8% once embedded at width=\textwidth. Trimming the handles and gaps
    # buys the width back without touching the shared legend_fontsize.
    _LEGEND_SPACING = dict(handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    ordered_top = _ordered(seen_top.keys())
    create_figure_legend(
        fig,
        [seen_top[a] for a in ordered_top],
        [display_name(a) for a in ordered_top],
        ncol=top_ncol,
        bbox_y=1.0 if top_rows == 1 else 1.0 + 0.03 * (top_rows - 1),
        fontsize=style.legend_fontsize,
        **_LEGEND_SPACING,
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
        # Same spacing as the top legend: the two sit in one figure, so their
        # handle samples have to be the same length to read as one system.
        **_LEGEND_SPACING,
    )

    if save_fig:
        tag = "_".join(benchmarks)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_{out_name}.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    benchmarks = ("rf146822", "rf31", "rf167120")
    n_iterations = 5000

    print("Generating combined regret + oracle-proposal-quality grid...")
    plot_regret_and_oracle_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
    )

    # print("Generating TabFM suggested-config MSE grid...")
    # plot_tabfm_suggestion_error_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )

    # print("Generating TabFM candidate-pool MSE grid...")
    # plot_tabfm_candidate_mse_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )

    # print("Generating TabFM top-10% candidate MSE grids (by true, by predicted)...")
    # plot_tabfm_candidate_topk_mse_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True, by="true"
    # )
    # plot_tabfm_candidate_topk_mse_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True, by="predicted"
    # )

    print("Generating arm-pulls landscape grid (TPE vs TabFM)...")
    plot_arm_pulls_landscape_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
        run=0,
        agg="max",
    )

    print("Generating TabFM MSE landscape grid...")
    plot_tabfm_mse_landscape_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=True,
        run=0,
        agg="mean",
    )

    # print("Generating arm-pulls parallel-coordinates grid (TPE vs TabFM)...")
    # plot_arm_pulls_parallel_coords_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )

    # print("Generating cumulative-regret gap (TPE - TabFM) vs iteration...")
    # plot_cumulative_regret_gap(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )

    # print("Generating TabFM suggested-config bias grid...")
    # plot_tabfm_suggestion_bias_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )

    # print("Generating TabFM calibration grid (predicted vs true vs train-label)...")
    # plot_tabfm_calibration_grid(
    #     benchmarks=benchmarks, n_iterations=n_iterations, save_fig=True
    # )
