"""Appendix figures: static reward-landscape structure of the RF tabular
benchmarks (experiments/benchmarks/rf_tabular_bandit.py), independent of any
optimizer run.

Two separate figures:

- `plot_landscape_heatmap_grid`: mean validation accuracy over (max_depth,
  max_features), averaged over the other two hyperparameters -- shows
  whether the good region is a depth-only ridge (segment, numerai28.6) or an
  isolated corner requiring a specific combination (credit-g).

- `plot_landscape_reward_cdf`: empirical CDF of arm reward, reward
  normalized per task to [0, 1] (0 = worst arm, 1 = best arm) -- shows how
  broad or sparse each task's near-optimal region is, independent of each
  task's absolute accuracy range. A CDF that rises early (most mass at low
  reward) means a sparse optimum; one that stays flat and only rises near 1
  means a broad optimum.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmarks.rf_tabular_bandit import (
    PARAM_NAMES,
    RFTabularFiniteBenchmark,
)
from experiments.utils.plots.plot_configs import (
    _bench_title,
    get_algorithm_color,
    paper_style,
    save_figure,
    set_research_style,
)

RESULTS_DIR = Path(__file__).parents[3] / "results"

set_research_style()

# Fixed per-panel box (in), matching delayed_feedback_plot / toy_plot's
# appendix convention: at the bumped appendix font sizes below, dividing a
# fixed page width across n panels crowds titles/ticks/colorbars together,
# so each panel instead gets its own budget and the figure grows with n.
_APPENDIX_PANEL_W_IN = 4.2
_APPENDIX_PANEL_H_IN = 3.4


def _appendix_style(columns: int = 2, max_legend_single_row: int = 3):
    """A paper_style bundle with the shared conventions kept but the font/
    line sizes bumped to appendix-readable (the main-paper defaults are
    ~6.5-9pt, sized for a printed column; these figures are viewed larger --
    see delayed_feedback_plot._appendix_style)."""
    return paper_style(
        "aaai",
        columns=columns,
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        legend_fontsize=12,
        linewidth=2.2,
        max_legend_single_row=max_legend_single_row,
    )


def _appendix_figsize(n_panels: int):
    """Per-panel appendix figure size -- see toy_plot._appendix_figsize."""
    return (_APPENDIX_PANEL_W_IN * n_panels, _APPENDIX_PANEL_H_IN)


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
    return (
        df.groupby(["max_depth", "max_features"]).val_acc.mean().unstack("max_features")
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


def plot_landscape_heatmap_grid(
    bm_ids: tuple[int, ...] = (146822, 31, 167120),
    save_fig: bool = False,
    columns: int = 2,
) -> None:
    """Appendix figure: (max_depth, max_features) mean-reward heatmap for
    each RF tabular task, side by side."""
    style = _appendix_style(columns=columns)
    n = len(bm_ids)

    fig, heat_axes = plt.subplots(1, n, figsize=_appendix_figsize(n))
    heat_axes = np.atleast_1d(heat_axes)
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

    fig.tight_layout()

    if save_fig:
        tag = "_".join(f"rf{bm_id}" for bm_id in bm_ids)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_landscape_heatmap_grid.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.show()


def plot_landscape_reward_cdf(
    bm_ids: tuple[int, ...] = (146822, 31, 167120),
    save_fig: bool = False,
    columns: int = 2,
) -> None:
    """Appendix figure: empirical CDF of normalized arm reward for each RF
    tabular task, overlaid on one axis, with a shared top legend (see
    plot_configs.create_figure_legend)."""
    style = _appendix_style(columns=columns, max_legend_single_row=len(bm_ids))

    fig, curve_ax = plt.subplots(figsize=_appendix_figsize(1))
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

    handles, labels = curve_ax.get_legend_handles_labels()
    style.legend(fig, handles, labels, bbox_y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    if save_fig:
        tag = "_".join(f"rf{bm_id}" for bm_id in bm_ids)
        out_path = RESULTS_DIR / "paper_plots" / f"{tag}_landscape_reward_cdf.pdf"
        save_figure(out_path, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    plot_landscape_heatmap_grid(save_fig=True)
    plot_landscape_reward_cdf(save_fig=True)
