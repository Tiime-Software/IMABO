"""Illustration of sin1, garland, rastrigin, and the coordination-barrier
Gaussian as 2D landscape heatmaps (d = 2 member of each family)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.coordination_barrier_experiment import MODE_HI, MODE_LO, make_mu_family
from experiments.utils.plots.plot_configs import paper_style, save_figure, set_research_style

set_research_style()

RESULTS_DIR = Path(__file__).parents[3] / "results" / "paper_plots"
RESOLUTION = 301

FUNCTIONS = [
    ("sin1", [0.0, 1.0], "Sin1"),
    ("garland", [0.0, 1.0], "Garland"),
    ("rastrigin", [-5.12, 5.12], "Rastrigin"),
    ("gaussian", [0.0, 1.0], "Gaussian"),
]

# This figure lives in the APPENDIX, where we don't need exact main-paper
# print size -- so, like ablation_plot.py / toy_plot.py's _appendix_style /
# _appendix_figsize, it follows the paper_style CONVENTION (style_axis look)
# but overrides paper_style's tiny print fonts/lines (~6.5-9pt, sized for
# shrink-to-column at final print) with larger, readable values. No shared
# legend here (each panel is its own function, identified by its title), so
# no legend headroom to reserve.
_APPENDIX_PANEL_W_IN = 3.4  # per-function panel width (square, like a heatmap)
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


def plot_toy_functions_2d(save_fig=False, columns=1):
    obj = ObjectiveFunctions(dim=2, noise_std=0.1)
    style = _appendix_style(columns)
    gaussian_2d = make_mu_family(2)

    fig, axes = plt.subplots(
        1,
        len(FUNCTIONS),
        figsize=(_APPENDIX_PANEL_W_IN * len(FUNCTIONS), _APPENDIX_PANEL_H_IN),
    )

    for idx, (name, domain, label) in enumerate(FUNCTIONS):
        ax = axes[idx]
        lo, hi = domain
        g = np.linspace(lo, hi, RESOLUTION)

        if name == "gaussian":
            fn = lambda x1, x2: gaussian_2d(np.array([x1, x2]))
        else:
            base_fn = obj.get_function_by_name(name)
            fn = lambda x1, x2, base_fn=base_fn: base_fn([x1, x2])
        z = np.array([[fn(x1, x2) for x1 in g] for x2 in g])

        im = ax.imshow(
            z, origin="lower", extent=(lo, hi, lo, hi), cmap="viridis", aspect="equal"
        )
        if name == "gaussian":
            # Mark the local (circle) and global (star) modes, matching the
            # coordination-barrier heatmap's convention (plot_landscapes).
            ax.scatter([MODE_LO], [MODE_LO], marker="o", s=28, facecolor="white",
                       edgecolor="black", linewidth=0.6, clip_on=False, zorder=5)
            ax.scatter([MODE_HI], [MODE_HI], marker="*", s=70, facecolor="white",
                       edgecolor="black", linewidth=0.6, clip_on=False, zorder=5)

        ax.set_title(label, fontweight="bold", fontsize=style.title_fontsize, pad=4)
        mid = (lo + hi) / 2
        # Drop the y-axis's lo tick: it sits right on top of the x-axis's lo
        # tick at the shared bottom-left corner (same value, so one is enough).
        ax.set_xticks([lo, mid, hi])
        ax.set_yticks([mid, hi])
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.grid(False)  # the shared style's grid just striples the heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        cbar.ax.tick_params(labelsize=style.tick_fontsize)

    fig.tight_layout(w_pad=1.0)

    if save_fig:
        out_path = RESULTS_DIR / "illustration_toy_functions.pdf"
        save_figure(out_path, bbox_inches="tight", parents=True)

    plt.close()


if __name__ == "__main__":
    plot_toy_functions_2d(save_fig=True)
