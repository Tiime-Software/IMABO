"""Style configuration and color constants for IMABO experiment plots."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

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


DISPLAY_NAME_OVERRIDES = {"IMABO": "I-MOSS-TPE"}


def display_name(name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(name, name)


LEGEND_PT_PER_INCH = 1.1
LEGEND_FONTSIZE_MIN = 12
LEGEND_FONTSIZE_MAX = 32

AXIS_LABEL_PT_PER_INCH = 5.2
AXIS_LABEL_FONTSIZE_MIN = 11
AXIS_LABEL_FONTSIZE_MAX = 32


def adaptive_label_fontsize(
    ax,
    pt_per_inch: float = AXIS_LABEL_PT_PER_INCH,
    fmin: float = AXIS_LABEL_FONTSIZE_MIN,
    fmax: float = AXIS_LABEL_FONTSIZE_MAX,
) -> float:
    """Axis-label fontsize scaled from this Axes' smaller dimension (inches).

    Uses whichever of width/height is smaller: an x-label needs horizontal
    room (bounded by axes width) and a rotated y-label needs vertical room
    (bounded by axes height), so a figure that's short (small height, e.g.
    figsize=(14, 5)) must get a smaller label than a tall one even at the
    same column count -- otherwise a long label ("Normalized Cumulative
    Regret") overflows/clips in the short figure. Using fractional width
    alone (ignoring inches) fixed inconsistent sizing across figures with
    different nominal `figsize` for the same column count, but broke this
    case; the smaller-dimension-in-inches version handles both.
    """
    fig = ax.get_figure()
    fig_w_in, fig_h_in = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_in = bbox.width * fig_w_in
    ax_h_in = bbox.height * fig_h_in
    return float(np.clip(min(ax_w_in, ax_h_in) * pt_per_inch, fmin, fmax))


def create_figure_legend(
    fig,
    handles,
    labels,
    *,
    ncol: int,
    bbox_y: float = 1.08,
    fontsize: float | None = None,
) -> None:
    """Add a shared top legend to `fig`."""
    if fontsize is None:
        fig_width, _ = fig.get_size_inches()
        fontsize = np.clip(
            fig_width * LEGEND_PT_PER_INCH, LEGEND_FONTSIZE_MIN, LEGEND_FONTSIZE_MAX
        )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        frameon=False,
        prop={"size": fontsize, "family": "serif", "weight": "bold"},
    )


def save_figure(
    out_path: Path,
    *,
    mkdir: bool = True,
    parents: bool = False,
    verbose: bool = True,
    **savefig_kwargs,
) -> None:
    """Save the current figure to `out_path`, matching each caller's existing
    savefig call exactly (pass e.g. `bbox_inches="tight"`, `dpi=300` through
    `savefig_kwargs` only if the original call used them) -- this only
    deduplicates the surrounding mkdir/savefig/print boilerplate, not the
    save behavior itself.
    """
    if mkdir:
        out_path.parent.mkdir(parents=parents, exist_ok=True)
    plt.savefig(out_path, **savefig_kwargs)
    if verbose:
        print(f"Saved to {out_path}")


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    """Create a covariance confidence ellipse of x and y.

    Args:
        x, y: Input data arrays
        ax: The axis object to plot the ellipse onto
        n_std: Number of standard deviations (default: 1.0)
        facecolor: Color to fill ellipse
        **kwargs: Additional arguments passed to Ellipse patch
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = np.corrcoef(x, y)[0, 1]
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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
