"""Illustration of sin1, garland, and rastrigin as 1D line plots."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.utils.plot_configs import set_research_style, ALGORITHM_COLORS
from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions

set_research_style()

RESULTS_DIR = Path(__file__).parents[2] / "results" / "paper_plots"
RESOLUTION = 1000

FUNCTIONS = [
    ("sin1", [0.0, 1.0], ALGORITHM_COLORS[0], "Sin1"),
    ("garland", [0.0, 1.0], ALGORITHM_COLORS[1], "Garland"),
    ("rastrigin", [-5.12, 5.12], ALGORITHM_COLORS[4], "Rastrigin"),
]


def plot_toy_functions_1d(save_fig=False):
    obj = ObjectiveFunctions(dim=1, noise_std=0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, domain, color, label) in enumerate(FUNCTIONS):
        ax = axes[idx]
        lo, hi = domain

        x = np.linspace(lo, hi, RESOLUTION)
        fn = obj.get_function_by_name(name, noise=False)
        y = np.array([fn([xi]) for xi in x])

        ax.plot(x, y, color=color, linewidth=2.0)
        ax.fill_between(
            x,
            y - obj.noise_std,
            y + obj.noise_std,
            alpha=0.25,
            color=color,
        )
        ax.set_xlim(lo, hi)

        ax.set_xlabel(r"$x$", fontweight="bold", fontsize=22)
        if idx == 0:
            ax.set_ylabel("f(x)", fontweight="bold", fontsize=22)

        ax.set_title(label, fontweight="bold", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / "illustration_toy_functions.pdf", bbox_inches="tight")
        print(f"Saved to {RESULTS_DIR / 'illustration_toy_functions.pdf'}")

    plt.show()


if __name__ == "__main__":
    plot_toy_functions_1d(save_fig=True)
