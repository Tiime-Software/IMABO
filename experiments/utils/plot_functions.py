"""Publication-quality plotting utilities for IMABO experiments."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).parent.parent.parent / "results"
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Style ─────────────────────────────────────────────────────────────────────

def set_research_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.shadow": False,
            "figure.dpi": 300,
            "figure.figsize": (10, 5),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "figure.autolayout": False,
        }
    )


set_research_style()

# Colorblind-friendly Wong palette
RESEARCH_COLORS = {
    "primary": "#0072B2",
    "secondary": "#E69F00",
    "success": "#009E73",
    "danger": "#D55E00",
    "warning": "#F0E442",
    "info": "#CC79A7",
    "dark": "#56B4E9",
    "neutral": "#999999",
}

ALGORITHM_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#999999",
]


def get_algorithm_color(index: int) -> str:
    return ALGORITHM_COLORS[index % len(ALGORITHM_COLORS)]


# ── HPO result helpers ─────────────────────────────────────────────────────────

def calculate_mean_and_std(results: list[dict], metric: str = "best_true_values"):
    all_runs = np.array([r[metric] for r in results])
    return np.mean(all_runs, axis=0), np.std(all_runs, axis=0)


def plot_results(
    results_imabo: list[dict],
    results_optuna_bandit: list[dict],
    **kwargs,
):
    metric = kwargs.get("metric", "best_true_values")
    k = kwargs.get("k", 0)
    n_runs = len(results_imabo)
    n_iterations = min(len(results_imabo[0][metric]), len(results_optuna_bandit[0][metric]))

    mean_imabo, std_imabo = calculate_mean_and_std(results_imabo, metric)
    mean_optuna, std_optuna = calculate_mean_and_std(results_optuna_bandit, metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_imabo, label="IMABO")
    ax.plot(mean_optuna, label="Optuna Bandit")
    ax.fill_between(range(len(mean_imabo)), mean_imabo - std_imabo, mean_imabo + std_imabo, alpha=0.2)
    ax.fill_between(range(len(mean_optuna)), mean_optuna - std_optuna, mean_optuna + std_optuna, alpha=0.2)
    ax.set_title(f"{metric} — {n_runs} runs, {n_iterations} iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.set_axisbelow(True)
    plt.tight_layout()


def plot_pairwise_differences_summary(
    results_imabo: list[dict],
    results_optuna_bandit: list[dict],
    **kwargs,
):
    metric = kwargs.get("metric", "best_true_values")
    imabo_runs = np.array([r[metric] for r in results_imabo])
    optuna_runs = np.array([r[metric] for r in results_optuna_bandit])
    n_iterations = min(imabo_runs.shape[1], optuna_runs.shape[1])

    means, stds, p25, p75, iters = [], [], [], [], []
    for t in range(n_iterations):
        diffs = [
            imabo_runs[i, t] - optuna_runs[j, t]
            for i in range(len(results_imabo))
            for j in range(len(results_optuna_bandit))
        ]
        means.append(np.mean(diffs))
        stds.append(np.std(diffs))
        p25.append(np.percentile(diffs, 25))
        p75.append(np.percentile(diffs, 75))
        iters.append(t)

    means_arr = np.array(means)
    stds_arr = np.array(stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(iters, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.15, color=RESEARCH_COLORS["primary"], label="±1 std")
    ax.fill_between(iters, p25, p75, alpha=0.35, color=RESEARCH_COLORS["primary"],
                    label="25th–75th percentile")
    ax.plot(iters, means, color=RESEARCH_COLORS["primary"], linewidth=2.5, label="Mean difference")
    ax.axhline(y=0, color=RESEARCH_COLORS["danger"], linestyle="--", linewidth=1.5,
               alpha=0.8, label="No difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("IMABO − Optuna (difference)")
    ax.set_title("Pairwise Differences Summary", pad=15)
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


# ── Toy-experiment plots ───────────────────────────────────────────────────────

def plot_trajectories_with_confidence_ellipses(
    results_dict: dict,
    algorithms: list[str],
    n_evals: list[int],
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "toy",
):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = np.corrcoef(x, y)[0, 1]
        ellipse = Ellipse(
            (0, 0),
            width=np.sqrt(1 + pearson) * 2,
            height=np.sqrt(1 - pearson) * 2,
            facecolor=facecolor,
            **kwargs,
        )
        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
            .translate(np.mean(x), np.mean(y))
        )
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    fig, ax = plt.subplots(figsize=(10, 7))
    legend_handles = []

    for i, algorithm in enumerate(algorithms):
        color = get_algorithm_color(i)
        xs, ys, sum_data, simple_data = [], [], [], []

        for n_eval in n_evals:
            for key in keys:
                n_iter = int(key.split("_")[-1])
                if n_iter == n_eval:
                    stats = results_dict[key][algorithm]
                    xs.append(stats["simple_regrets"]["mean"])
                    ys.append(float(np.mean(stats["regrets"]["sum_regrets"])))
                    sum_data.append(stats["regrets"]["sum_regrets"])
                    simple_data.append(stats["simple_regrets"]["simple_regrets"])

        lines = []
        for j in range(len(xs) - 1):
            alpha = 0.4 + 0.2 * (j / max(len(xs) - 1, 1))
            line = ax.plot([xs[j], xs[j + 1]], [ys[j], ys[j + 1]],
                           color=color, linewidth=3, alpha=alpha,
                           label=algorithm if j == 0 else "")
            if j == 0:
                lines.extend(line)
        if lines:
            legend_handles.append(lines[0])

        for j, (sr, sr_simple) in enumerate(zip(sum_data, simple_data)):
            confidence_ellipse(sr_simple, sr, ax, n_std=1.0,
                               alpha=0.2, facecolor=color, edgecolor=color)
            ax.scatter(sr_simple, sr, c=color, alpha=0.4, s=15,
                       edgecolors="white", linewidths=0.5, zorder=4)
            ax.plot(np.mean(sr_simple), np.mean(sr), "o", color=color,
                    markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.annotate(f"{n_evals[j]}", (np.mean(sr_simple), np.mean(sr)),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, alpha=0.7, ha="left", va="bottom", zorder=6)

    ax.set_xlabel("Simple Regret (Best Point Found)")
    ax.set_ylabel("Cumulative Regret (Total Regret)")
    ax.set_title("Algorithm Performance Trajectories\n(Lower-left = better)", pad=25)
    if legend_handles:
        ax.legend(legend_handles, algorithms, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(algorithms)),
                  fontsize=13, frameon=True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / f"trajectories_with_confidence_ellipses_{function_name}_{exp_type}.pdf")
    plt.show()


def plot_trajectories(
    results_dict: dict,
    algorithms: list[str],
    n_evals: list[int],
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "toy",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    base_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    markers = [base_markers[i % len(base_markers)] for i in range(len(algorithms))]

    all_xs, all_ys = [], []
    trajectories: dict[str, tuple] = {}

    for i, algorithm in enumerate(algorithms):
        xs, ys = [], []
        for n_eval in n_evals:
            for key in keys:
                if int(key.split("_")[-1]) == n_eval:
                    stats = results_dict[key][algorithm]
                    xs.append(stats["simple_regrets"]["mean"])
                    ys.append(float(stats["regrets"]["sum_regrets"].mean()))
        xs, ys = np.array(xs), np.array(ys)
        trajectories[algorithm] = (xs, ys)
        all_xs.extend(xs)
        all_ys.extend(ys)

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)

    for i, algorithm in enumerate(algorithms):
        xs, ys = trajectories[algorithm]
        color = get_algorithm_color(i)
        for j in range(len(xs) - 1):
            alpha = 0.4 + 0.2 * (j / max(len(xs) - 1, 1))
            ax.plot([xs[j], xs[j + 1]], [ys[j], ys[j + 1]],
                    color=color, linewidth=3, alpha=alpha)
        sizes = np.linspace(60, 120, len(xs))
        ax.scatter(xs, ys, c=color, s=sizes, marker=markers[i],
                   alpha=0.8, edgecolors="white", linewidths=2, label=algorithm, zorder=5)

    ax.set_xlabel("Simple Regret (Best Point Found)")
    ax.set_ylabel("Cumulative Regret (Total Regret)")
    ax.set_title("Performance Trajectories", pad=25, fontweight="bold", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / f"trajectories_{function_name}_{exp_type}.pdf")
    plt.show()


def plot_cumulative_regrets(
    results_dict: dict,
    algorithms: list[str],
    n_evals: list[int],
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "toy",
):
    fig, axes = plt.subplots(len(keys), 2, figsize=(16, 4 * len(keys)), sharex=False)
    if len(keys) == 1:
        axes = axes.reshape(1, -1)

    simple_regret_data: dict[str, dict] = {}
    n_iter_values = []

    for key in keys:
        n_iter = int(key.split("_")[-1])
        n_iter_values.append(n_iter)
        for algorithm in algorithms:
            if algorithm not in simple_regret_data:
                simple_regret_data[algorithm] = {"means": [], "stds": []}
            stats = results_dict[key][algorithm]
            simple_regret_data[algorithm]["means"].append(stats["simple_regrets"]["mean"])
            simple_regret_data[algorithm]["stds"].append(stats["simple_regrets"]["std"])

    for idx, key in enumerate(keys):
        ax_cum = axes[idx, 0]
        ax_simple = axes[idx, 1]
        n_iter = int(key.split("_")[-1])

        for i, algorithm in enumerate(algorithms):
            mean_regrets = results_dict[key][algorithm]["regrets"]["mean"]
            cum = np.cumsum(mean_regrets) / np.arange(1, len(mean_regrets) + 1)
            ax_cum.plot(range(1, len(cum) + 1), cum,
                        color=get_algorithm_color(i), label=algorithm)

        ax_cum.set_ylabel("Cumulative Regret")
        ax_cum.set_title(f"Cumulative Regrets — {n_iter} iterations")
        ax_cum.legend(loc="upper right")
        ax_cum.set_axisbelow(True)

        for i, algorithm in enumerate(algorithms):
            ax_simple.plot(n_iter_values,
                           simple_regret_data[algorithm]["means"],
                           color=get_algorithm_color(i), marker="o", label=algorithm)
            ax_simple.fill_between(
                n_iter_values,
                np.array(simple_regret_data[algorithm]["means"])
                - np.array(simple_regret_data[algorithm]["stds"]),
                np.array(simple_regret_data[algorithm]["means"])
                + np.array(simple_regret_data[algorithm]["stds"]),
                color=get_algorithm_color(i), alpha=0.2,
            )
        ax_simple.set_ylabel("Simple Regret")
        ax_simple.set_title("Simple Regrets vs Iterations")
        ax_simple.set_xlabel("Number of Iterations")
        ax_simple.legend(loc="upper right")
        ax_simple.set_axisbelow(True)

    axes[-1, 0].set_xlabel("Iteration")
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / f"cumulative_regrets_{function_name}_{exp_type}.pdf")
    plt.show()


def plot_simple_regret_ratio(
    results_dict: dict,
    baseline_algorithm: str,
    comparison_algorithm: str,
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "ablation",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    ratios_mean, ratios_std, n_iter_values = [], [], []

    for key in keys:
        n_iter = int(key.split("_")[-1])
        n_iter_values.append(n_iter)
        baseline = results_dict[key][baseline_algorithm]["simple_regrets"]["simple_regrets"]
        comparison = results_dict[key][comparison_algorithm]["simple_regrets"]["simple_regrets"]
        ratios = comparison / baseline
        ratios_mean.append(float(np.mean(ratios)))
        ratios_std.append(float(np.std(ratios)))

    ax.errorbar(n_iter_values, ratios_mean, yerr=ratios_std,
                marker="o", markersize=8, capsize=5, capthick=2,
                color=RESEARCH_COLORS["primary"],
                label=f"{comparison_algorithm} / {baseline_algorithm}")
    ax.axhline(y=1.0, color=RESEARCH_COLORS["danger"], linestyle="--",
               alpha=0.7, label="Equal performance")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Simple Regret Ratio")
    ax.set_title(f"Simple Regret Ratio: {comparison_algorithm} vs {baseline_algorithm}\n"
                 f"Function: {function_name}", pad=20)
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    if max(ratios_mean) / max(min(ratios_mean), 1e-12) > 5:
        ax.set_yscale("log")
        ax.set_ylabel("Simple Regret Ratio (log scale)")
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            PLOT_DIR / f"simple_regret_ratio_{comparison_algorithm}_vs_{baseline_algorithm}_{function_name}_{exp_type}.pdf"
        )
    plt.show()


def plot_dimension_comparison(
    results_dict: dict,
    algorithms: list[str],
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "dimension_comparison",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    dimensions = sorted({int(k.split("_")[1].replace("D", "")) for k in keys})

    for i, algorithm in enumerate(algorithms):
        means, stds = [], []
        for dim in dimensions:
            matching = next((k for k in keys if f"_{dim}D_" in k), None)
            if matching:
                stats = results_dict[matching][algorithm]
                means.append(stats["simple_regrets"]["mean"])
                stds.append(stats["simple_regrets"]["std"])
        ax.errorbar(dimensions, means, yerr=stds, marker="o", markersize=8,
                    capsize=5, capthick=2, color=get_algorithm_color(i),
                    label=algorithm.replace("_", " ").title())

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Simple Regret")
    ax.set_title("Simple Regret vs Dimension", pad=20)
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_xticks(dimensions)
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / f"dimension_comparison_{function_name}_{exp_type}.pdf")
    plt.show()


def plot_k_experiment(
    results: dict,
    function_name: str,
    k_values: list[int],
    save_fig: bool = False,
):
    """Two-panel figure for the MOSS / k ablation.

    Left: simple regret (bar chart, IMABO + Optuna k=...).
    Right: cumulative regret curves over iterations.
    """
    algorithms = ["IMABO"] + [f"Optuna k={k}" for k in k_values]
    fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: simple regret bar chart ────────────────────────────────────────
    means = [results[a]["simple_regrets"]["mean"] for a in algorithms]
    stds = [results[a]["simple_regrets"]["std"] for a in algorithms]
    x_pos = np.arange(len(algorithms))
    colors = [get_algorithm_color(i) for i in range(len(algorithms))]

    bars = ax_bar.bar(x_pos, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="white", linewidth=0.8, alpha=0.85)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        ["IMABO"] + [f"k={k}" for k in k_values], rotation=30, ha="right"
    )
    ax_bar.set_ylabel("Simple Regret")
    ax_bar.set_title(f"Simple Regret — {function_name} (dim={4})")
    ax_bar.set_yscale("log")
    ax_bar.set_axisbelow(True)

    # ── Right: cumulative regret curves ──────────────────────────────────────
    for i, alg in enumerate(algorithms):
        mean_r = results[alg]["regrets"]["mean"]
        std_r = results[alg]["regrets"]["std"]
        cum = np.cumsum(mean_r) / np.arange(1, len(mean_r) + 1)
        cum_lo = np.cumsum(mean_r - std_r) / np.arange(1, len(mean_r) + 1)
        cum_hi = np.cumsum(mean_r + std_r) / np.arange(1, len(mean_r) + 1)
        iters = np.arange(1, len(cum) + 1)
        color = get_algorithm_color(i)
        label = "IMABO" if alg == "IMABO" else alg.replace("Optuna ", "")
        ax_cum.plot(iters, cum, color=color, label=label, linewidth=2)
        ax_cum.fill_between(iters, cum_lo, cum_hi, color=color, alpha=0.12)

    ax_cum.set_xlabel("Iteration")
    ax_cum.set_ylabel("Avg. Cumulative Regret")
    ax_cum.set_title(f"Cumulative Regret — {function_name} (dim={4})")
    ax_cum.legend(loc="upper right", fontsize=10,
                  ncol=2 if len(algorithms) > 4 else 1)
    ax_cum.set_axisbelow(True)

    fig.suptitle(f"MOSS Oracle / k Ablation — {function_name}", y=1.02)
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / f"k_ablation_{function_name}.pdf")
    plt.show()


def plot_ablation_tpe_impact(
    results_by_function: dict,
    algorithms: list[str],
    dims: list[int],
    save_fig: bool = False,
):
    """Simple regret vs dimension for each function, one column per function.

    Produces the main figure for the TPE oracle ablation (Table 2 companion).
    Each subplot: x=dimension, y=simple regret (log scale), lines per algorithm
    with ±1 std shading.
    """
    functions = list(results_by_function.keys())
    n_funcs = len(functions)
    fig, axes = plt.subplots(1, n_funcs, figsize=(5 * n_funcs, 4), sharey=False)
    if n_funcs == 1:
        axes = [axes]

    for ax, fn in zip(axes, functions):
        results_dict = results_by_function[fn]
        for i, algorithm in enumerate(algorithms):
            color = get_algorithm_color(i)
            means, stds = [], []
            for dim in dims:
                key = next((k for k in results_dict if f"_{dim}D_" in k), None)
                if key is None:
                    continue
                stats = results_dict[key][algorithm]
                means.append(stats["simple_regrets"]["mean"])
                stds.append(stats["simple_regrets"]["std"])

            means_arr = np.array(means)
            stds_arr = np.array(stds)
            label = "Random" if algorithm == "Random" else "IMABO"
            linestyle = "--" if algorithm == "Random" else "-"
            ax.plot(dims, means_arr, color=color, linestyle=linestyle,
                    marker="o", markersize=6, linewidth=2, label=label)
            ax.fill_between(dims,
                            np.maximum(means_arr - stds_arr, 1e-9),
                            means_arr + stds_arr,
                            color=color, alpha=0.15)

        ax.set_yscale("log")
        ax.set_xticks(dims)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Simple Regret")
        ax.set_title(fn.capitalize())
        ax.legend(loc="upper left", fontsize=10)
        ax.set_axisbelow(True)

    fig.suptitle("TPE Oracle Impact: IMABO vs Random (Simple Regret)", y=1.02)
    plt.tight_layout()
    if save_fig:
        plt.savefig(PLOT_DIR / "ablation_tpe_impact.pdf")
    plt.show()


def plot_simple_regret_ratio_dimension_comparison(
    results_dict: dict,
    baseline_algorithm: str,
    comparison_algorithm: str,
    keys: list[str],
    function_name: str,
    save_fig: bool = False,
    exp_type: str = "dimension_comparison",
):
    fig, ax = plt.subplots(figsize=(10, 8))

    dimension_data: dict[int, list] = {}
    for key in keys:
        parts = key.split("_")
        dim = int(parts[1].replace("D", ""))
        n_iter = int(parts[2])
        dimension_data.setdefault(dim, []).append((n_iter, key))

    for i, dim in enumerate(sorted(dimension_data)):
        iter_key_pairs = sorted(dimension_data[dim], key=lambda x: x[0])
        n_iter_values = [p[0] for p in iter_key_pairs]
        ratios_mean, ratios_std = [], []

        for _, key in iter_key_pairs:
            baseline = results_dict[key][baseline_algorithm]["simple_regrets"]["simple_regrets"]
            comparison = results_dict[key][comparison_algorithm]["simple_regrets"]["simple_regrets"]
            ratios = comparison / baseline
            ratios_mean.append(float(np.mean(ratios)))
            ratios_std.append(float(np.std(ratios)))

        ax.errorbar(n_iter_values, ratios_mean, yerr=ratios_std,
                    marker="o", markersize=8, capsize=5, capthick=2,
                    color=get_algorithm_color(i), label=f"Dimension {dim}")

    ax.axhline(y=1.0, color=RESEARCH_COLORS["danger"], linestyle="--",
               alpha=0.7, label="Equal performance")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Simple Regret Ratio")
    ax.set_title(f"Simple Regret Ratio: {comparison_algorithm} vs {baseline_algorithm}\n"
                 f"Function: {function_name}", pad=20)
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            PLOT_DIR / f"simple_regret_ratio_dimension_{function_name}_{exp_type}.pdf"
        )
    plt.show()
