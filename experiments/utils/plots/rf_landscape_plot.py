"""Appendix figure: static reward-landscape structure of the RF tabular
benchmarks (experiments/benchmarks/rf_tabular_bandit.py), independent of any
optimizer run.

Top row: mean validation accuracy over (max_depth, max_features), averaged
over the other two hyperparameters -- shows whether the good region is a
depth-only ridge (segment, numerai28.6) or an isolated corner requiring a
specific combination (credit-g).

Bottom row: empirical CDF of arm reward, reward normalized per task to
[0, 1] (0 = worst arm, 1 = best arm) -- shows how broad or sparse each task's
near-optimal region is, independent of each task's absolute accuracy range. A
CDF that rises early (most mass at low reward) means a sparse optimum; one
that stays flat and only rises near 1 means a broad optimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.benchmarks.rf_tabular_bandit import PARAM_NAMES, RFTabularFiniteBenchmark
from experiments.utils.plots.plot_configs import (
    _bench_title,
    get_algorithm_color,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"

set_research_style()


def _mean_reward_grid(bm_id: int) -> pd.DataFrame:
    """Mean val_acc over (max_depth, max_features), averaged over the other
    two hyperparameters, for the same coarsened grid the bandit experiments
    use (default n_values in RFTabularFiniteBenchmark)."""
    bench = RFTabularFiniteBenchmark(bm_id=bm_id)
    df = pd.DataFrame(
        [
            {**dict(zip(PARAM_NAMES, key)), "val_acc": value}
            for key, value in bench.lookup.items()
        ]
    )
    return df.groupby(["max_depth", "max_features"]).val_acc.mean().unstack(
        "max_features"
    )


def _reward_cdf(bm_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(reward normalized to [worst, best] -> [0, 1], empirical CDF), reward
    sorted ascending -- a curve that rises early (most mass at low reward)
    means a sparse optimum; one that stays flat and only rises near 1 means a
    broad optimum."""
    bench = RFTabularFiniteBenchmark(bm_id=bm_id)
    values = np.sort(np.array(list(bench.lookup.values())))
    worst, best = values.min(), values.max()
    normalized = (values - worst) / (best - worst)
    cdf = np.arange(1, len(values) + 1) / len(values)
    return normalized, cdf


def plot_landscape_structure_grid(
    bm_ids: tuple[int, ...] = (146822, 31, 167120),
    save_fig: bool = False,
    conference: str = "aaai",
    columns: int = 2,
) -> None:
    """Appendix figure combining the (max_depth, max_features) heatmap and
    the sorted-reward curve for each RF tabular task, side by side."""
    style = paper_style(conference=conference, columns=columns)
    n = len(bm_ids)

    fig = plt.figure(figsize=(style.width_in, style.width_in * 0.62))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.85], hspace=0.65, wspace=0.5)

    heat_axes = [fig.add_subplot(gs[0, j]) for j in range(n)]
    for ax, bm_id in zip(heat_axes, bm_ids):
        pivot = _mean_reward_grid(bm_id)
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(
            [f"{v:.2g}" for v in pivot.columns],
            rotation=45,
            ha="right",
            fontsize=style.tick_fontsize,
        )
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(
            [f"{int(v)}" for v in pivot.index], fontsize=style.tick_fontsize
        )
        ax.set_title(
            _bench_title(f"rf{bm_id}"),
            fontsize=style.title_fontsize,
            fontweight="bold",
            pad=4,
        )
        ax.set_xlabel("max_features", fontsize=style.label_fontsize)
        if ax is heat_axes[0]:
            ax.set_ylabel("max_depth", fontsize=style.label_fontsize)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cbar.ax.tick_params(labelsize=style.tick_fontsize)
        cbar.set_label("val_acc", fontsize=style.tick_fontsize)

    curve_ax = fig.add_subplot(gs[1, :])
    for i, bm_id in enumerate(bm_ids):
        normalized, cdf = _reward_cdf(bm_id)
        curve_ax.plot(
            normalized,
            cdf,
            color=get_algorithm_color(i),
            linewidth=style.linewidth * 1.6,
            label=_bench_title(f"rf{bm_id}"),
        )
    curve_ax.set_xlabel("Reward", fontsize=style.label_fontsize)
    curve_ax.set_ylabel("Fraction of arms", fontsize=style.label_fontsize)
    style.style_axis(curve_ax)
    curve_ax.legend(
        loc="upper left",
        fontsize=style.legend_fontsize,
        frameon=False,
    )

    if save_fig:
        tag = "_".join(f"rf{bm_id}" for bm_id in bm_ids)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_landscape_structure_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_landscape_structure_grid(save_fig=True)
