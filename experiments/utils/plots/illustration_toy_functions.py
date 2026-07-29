"""Illustration of sin1, garland, and rastrigin as 1D line plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.utils.plots.plot_configs import (
    ALGORITHM_COLORS,
    paper_style,
    save_figure,
    set_research_style,
)

set_research_style()

RESULTS_DIR = Path(__file__).parents[3] / "results" / "paper_plots"
RESOLUTION = 1000

FUNCTIONS = [
    ("sin1", [0.0, 1.0], ALGORITHM_COLORS[0], "Sin1"),
    ("garland", [0.0, 1.0], ALGORITHM_COLORS[1], "Garland"),
    ("rastrigin", [-5.12, 5.12], ALGORITHM_COLORS[4], "Rastrigin"),
]

# This figure lives in the APPENDIX, where we don't need exact main-paper
# print size -- so, like ablation_plot.py / toy_plot.py's _appendix_style /
# _appendix_figsize, it follows the paper_style CONVENTION (style_axis look)
# but overrides paper_style's tiny print fonts/lines (~6.5-9pt, sized for
# shrink-to-column at final print) with larger, readable values. No shared
# legend here (each panel is its own function, identified by its title), so
# no legend headroom to reserve.
_APPENDIX_PANEL_W_IN = 4.2  # per-function panel width
_APPENDIX_PANEL_H_IN = 3.4  # per-panel height


def _appendix_style(columns: int = 1):
    return paper_style(
        "aaai",
        columns=columns,
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        linewidth=2.2,
    )


def plot_toy_functions_1d(save_fig=False, columns=1):
    obj = ObjectiveFunctions(dim=1, noise_std=0.1)
    style = _appendix_style(columns)

    fig, axes = plt.subplots(
        1, len(FUNCTIONS), figsize=(_APPENDIX_PANEL_W_IN * len(FUNCTIONS), _APPENDIX_PANEL_H_IN)
    )

    for idx, (name, domain, color, label) in enumerate(FUNCTIONS):
        ax = axes[idx]
        lo, hi = domain

        x = np.linspace(lo, hi, RESOLUTION)
        fn = obj.get_function_by_name(name, noise=False)
        y = np.array([fn([xi]) for xi in x])

        ax.plot(x, y, color=color, linewidth=style.linewidth)
        ax.fill_between(
            x,
            y - obj.noise_std,
            y + obj.noise_std,
            alpha=0.25,
            color=color,
        )
        ax.set_xlim(lo, hi)

        if idx == 1:  # Middle subplot
            ax.set_xlabel(r"$x$", fontweight="bold", fontsize=style.label_fontsize)
        if idx == 0:  # Left subplot
            ax.set_ylabel("f(x)", fontweight="bold", fontsize=style.label_fontsize)

        ax.set_title(label, fontweight="bold", fontsize=style.title_fontsize, pad=4)
        style.style_axis(ax)

    plt.tight_layout()

    if save_fig:
        out_path = RESULTS_DIR / "illustration_toy_functions.pdf"
        save_figure(out_path, bbox_inches="tight", parents=True)

    plt.close()


if __name__ == "__main__":
    plot_toy_functions_1d(save_fig=True)
