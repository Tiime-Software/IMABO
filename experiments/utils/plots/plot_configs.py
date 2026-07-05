"""Style configuration and color constants for IMABO experiment plots."""

import matplotlib.pyplot as plt

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
