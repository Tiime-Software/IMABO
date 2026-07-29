"""Style configuration and color constants for IMABO experiment plots."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

# File-only plotting: every figure in this package is written to a PDF via
# save_figure/savefig, never displayed. Forcing the non-interactive Agg
# backend here (the shared import point of every plot module) stops the
# default macOS backend from popping a window per figure when a plot script
# is run from the terminal.
matplotlib.use("Agg")
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


# Keep display labels identical to the canonical identities in
# ALGORITHM_STYLES below, so the same method reads the same in every figure.
DISPLAY_NAME_OVERRIDES = {"IMABO": "IMOSS-TPE"}


def display_name(name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(name, name)


# --- Canonical per-algorithm styling --------------------------------------
# A method must look identical in every figure -- same color, same marker --
# no matter which script drew it or what order it appears in. The scripts feed
# raw series names that vary in spelling/casing/suffix ("IMABO", "imoss_tpe",
# "IMABO-beta0.5", "IMOSS-TABFM", ...); normalize_algorithm_name() folds those
# to one canonical identity, and ALGORITHM_STYLES pins each identity to a fixed
# (color, marker). This replaced the old by-list-position coloring in hpo_plot
# / hotpotqa_plot, which let the same method change color between figures.
#
# Every figure draws from ONE palette -- blue, orange, green, vermillion, plus
# reddish purple for the one five-series figure (the RF regret comparison with
# Hier-MAB) -- all from the Wong (Nature Methods, 2011) set, so each plot has
# the same look as the RF figure (no palette's hard-to-read yellow, no gray).
#
# Colorblind note: orange and vermillion are the one pair in this palette that
# converge (both -> gold) under deuteranopia. Every figure pairs them (IMOSS-TPE
# orange + UCB-AIR/Stroquool vermillion), so they always need a second,
# color-independent channel: the line plots have unique per-method MARKERS, and
# the one marker-less chart (hotpotqa's simple-regret bars) gets HATCH patterns
# (see plot_hotpotqa_results). With that, the palette reads correctly under
# deuteranopia/protanopia.
#
# Methods that appear in more than one figure keep one fixed color (IMOSS-TPE
# orange, IMOSS-TabFM green, UCB-AIR vermillion); the figure-specific ones fill
# the remaining slots. Blue and green/vermillion are reused across methods that
# NEVER co-occur, which is safe -- distinctness only has to hold within a single
# figure. Markers stay unique per method regardless.
_PALETTE_BLUE = "#0072B2"
_PALETTE_ORANGE = "#E69F00"
_PALETTE_GREEN = "#009E73"
_PALETTE_VERMILLION = "#D55E00"
_PALETTE_PURPLE = "#CC79A7"
_PALETTE_SKY = "#56B4E9"
ALGORITHM_STYLES: dict[str, tuple[str, str]] = {
    "IMOSS-Random": (_PALETTE_BLUE, "o"),  # rf (random oracle; a.k.a. IMOSS)
    "IMOSS-TPE": (_PALETTE_ORANGE, "^"),  # rf, hotpotqa, hpo   (a.k.a. IMABO)
    "IMOSS-TabFM": (_PALETTE_GREEN, "s"),  # rf, hotpotqa
    "IMOSS-TabPFN": (_PALETTE_GREEN, "s"),  # rf (TabPFN comparison)
    # Per-pull TabPFN variant (fit on individual pulls, not per-arm means).
    "IMOSS-TabPFN-pull": (_PALETTE_GREEN, "X"),  # rf (TabPFN per-pull comparison)
    "UCB-AIR": (_PALETTE_VERMILLION, "D"),  # rf, hotpotqa
    # Factored baseline (AutoRAG-HP's two-level hierarchical MAB). Appears in
    # the main RF regret figure ALONGSIDE the four core colors, so it cannot
    # share a slot: it gets the palette's fifth Wong color (reddish purple),
    # plus a unique marker.
    "Hier-MAB": (_PALETTE_PURPLE, "*"),  # rf, hotpotqa (regret comparison)
    # Continuous LR/SVM comparison: Hier-MAB run at two geometric-grid
    # resolutions (10 vs 100 values per axis). They co-occur with each other
    # (and with HOO-T's blue) but never with plain "Hier-MAB", so the 10-point
    # variant keeps the family's purple/star; the 100-point one takes the
    # remaining Wong color (sky blue) with its own marker.
    "Hier-MAB-10": (_PALETTE_PURPLE, "*"),  # lr/svm (factored, 10-pt grid)
    "Hier-MAB-100": (_PALETTE_SKY, "X"),  # lr/svm (factored, 100-pt grid)
    # Coordination-barrier counterexample: Hier-MAB at three grid resolutions
    # in one figure. One family color (purple) for all three -- they are the
    # same method at different resolutions -- disambiguated by marker plus a
    # per-level linestyle set locally in coordination_barrier_experiment.plot.
    "Hier-MAB-6": (_PALETTE_PURPLE, "*"),  # toy counterexample (6-pt grid)
    "Hier-MAB-11": (_PALETTE_PURPLE, "*"),  # toy counterexample (family, 11-pt)
    "Hier-MAB-21": (_PALETTE_PURPLE, "X"),  # toy counterexample (21-pt grid)
    "Hier-MAB-101": (_PALETTE_PURPLE, "P"),  # toy counterexample (101-pt grid)
    # Univariate-Parzen TPE (multivariate=False): same family as IMOSS-TPE,
    # so same orange; unique marker (plus a dashed linestyle set locally in
    # coordination_barrier_experiment) keeps the pair apart in one panel.
    "IMOSS-TPE-univ": (_PALETTE_ORANGE, "d"),  # toy counterexample
    "Random": (_PALETTE_BLUE, "p"),  # hotpotqa (no overlap with IMOSS)
    "HOO-T": (_PALETTE_BLUE, "v"),  # hpo
    "StoSOO": (_PALETTE_GREEN, "<"),  # hpo
    "Stroquool": (_PALETTE_VERMILLION, ">"),  # hpo
}

_DEFAULT_ALGORITHM_STYLE = ("#000000", "o")

# Raw-name -> canonical-identity aliases (matched after case/suffix folding).
_ALGORITHM_ALIASES = {
    "imabo": "IMOSS-TPE",
    "imabo-notpe": "IMOSS-Random",
    "imoss": "IMOSS-Random",
    "imoss-random": "IMOSS-Random",
    "imoss-tpe": "IMOSS-TPE",
    "imoss-tabfm": "IMOSS-TabFM",
    "imoss-tabpfn": "IMOSS-TabPFN",
    "imoss-tabpfn-pull": "IMOSS-TabPFN-pull",
    "ucb-air": "UCB-AIR",
    "ucbair": "UCB-AIR",
    "hier-mab": "Hier-MAB",
    "hiermab": "Hier-MAB",
    "hier-ucb": "Hier-MAB",
    "autorag-hp": "Hier-MAB",
    "hier-mab-10": "Hier-MAB-10",
    "hier-mab-100": "Hier-MAB-100",
    "hier-mab-6": "Hier-MAB-6",
    "hier-mab-11": "Hier-MAB-11",
    "hier-mab-21": "Hier-MAB-21",
    "hier-mab-101": "Hier-MAB-101",
    "imoss-tpe-univ": "IMOSS-TPE-univ",
    "imoss-tpe-uni": "IMOSS-TPE-univ",
    "random": "Random",
    "random-search": "Random",
    "hoo-t": "HOO-T",
    "hoo": "HOO-T",
    "stosoo": "StoSOO",
    "stroquool": "Stroquool",
}


def normalize_algorithm_name(name: str) -> str:
    """Fold a raw series name to its canonical identity in ALGORITHM_STYLES.

    Absorbs the spelling/casing/suffix drift across scripts and experiments: a
    trailing run-config tag ("-beta0.5", "_beta_0.8") is dropped, "_" is
    treated as "-", and matching is case-insensitive. Returns the input
    unchanged if nothing matches, so an unknown series still plots (with the
    default style) rather than raising."""
    key = name.strip().lower().replace("_", "-")
    key = re.sub(r"-beta[-\d.]*$", "", key)  # drop trailing run-config tag
    return _ALGORITHM_ALIASES.get(key, name)


def algorithm_style(name: str) -> tuple[str, str]:
    """(color, marker) for a series, consistent across every figure -- see
    normalize_algorithm_name and ALGORITHM_STYLES."""
    return ALGORITHM_STYLES.get(normalize_algorithm_name(name), _DEFAULT_ALGORITHM_STYLE)


def algorithm_color(name: str) -> str:
    return algorithm_style(name)[0]


def algorithm_marker(name: str) -> str:
    return algorithm_style(name)[1]


# --- Print-figure sizing -----------------------------------------------
# Per-conference page geometry. `two_column` says whether the template sets
# body text in two columns (AAAI) or one (NeurIPS, arXiv). Widths in inches:
#   "column" -- one text column (only defined for two-column templates)
#   "text"   -- the full text-block width (\textwidth)
#   AAAI    -- aaai2027.sty: \textwidth 7.0in, \columnsep 0.375in, so one
#              column is (7.0-0.375)/2 = 3.31in. Two columns.
#   NeurIPS -- neurips .sty: single column, \textwidth 5.5in.
#   arXiv   -- article/arxiv.sty on letterpaper, ~1in margins: \textwidth 6.5in.
# `columns` is how many of the template's text columns a figure spans. AAAI
# supports 1 (\columnwidth) or 2 (\textwidth, a `figure*`). NeurIPS and arXiv
# are single-column, so a figure always spans the one text column -- columns=1
# gives the full text width and there is no 2-column span.
CONFERENCE_SPECS = {
    "aaai": {"two_column": True, "column": 3.3, "text": 7.0},
    "neurips": {"two_column": False, "text": 5.5},
    "arxiv": {"two_column": False, "text": 6.5},
}


def paper_figure_width_in(columns: int = 1, conference: str = "aaai") -> float:
    """Physical width (inches) for a figure spanning `columns` text columns of
    `conference`'s template, generated at that size so it embeds at ~100%
    scale without being shrunk afterward.

    Two-column templates (AAAI): columns=1 -> one column, columns=2 -> both
    columns (a LaTeX `figure*`). Single-column templates (NeurIPS, arXiv) have
    only one text column, so columns=1 gives the full text width; there is no
    2-column span, so columns != 1 raises.
    """
    spec = CONFERENCE_SPECS[conference]
    if not spec["two_column"]:
        if columns != 1:
            raise ValueError(
                f"{conference!r} is a single-column format: a figure spans the "
                f"full text width (columns=1); it has no {columns}-column span."
            )
        return spec["text"]
    if columns == 1:
        return spec["column"]
    if columns == 2:
        return spec["text"]
    raise ValueError(f"columns must be 1 or 2 for {conference!r}, got {columns!r}")


# Plain pt sizes for a figure generated via paper_figure_width_in. Same
# regardless of column count -- these are absolute print sizes (~7-9pt, close
# to a paper's 10pt body text), not scaled to the panel's own physical size.
PAPER_TITLE_FONTSIZE = 9
PAPER_TICK_FONTSIZE = 6.5
PAPER_LABEL_FONTSIZE = 8.5
PAPER_LEGEND_FONTSIZE = 8


def legend_ncol_for_columns(
    n_labels: int, columns: int, max_single_row: int = 3
) -> int:
    """Legend column count for a paper_figure_width_in-sized figure: at
    columns=2 (a figure*) there's usually enough width for every entry on one
    row; at columns=1 there usually isn't, so entries beyond max_single_row
    wrap onto a second row instead of squeezing (and shrinking the rest of
    the figure to make room for) one very wide legend row.
    """
    if columns == 2 or n_labels <= max_single_row:
        return n_labels
    return -(-n_labels // 2)  # ceil(n_labels / 2) -> 2 rows

# AAAI-specific aliases, kept as the reference values the above were derived
# from (7.0in \textwidth == paper_figure_width_in(2, "aaai")).
AAAI_TEXTWIDTH_IN = CONFERENCE_SPECS["aaai"]["text"]
AAAI_TITLE_FONTSIZE = PAPER_TITLE_FONTSIZE
AAAI_TICK_FONTSIZE = PAPER_TICK_FONTSIZE
AAAI_LABEL_FONTSIZE = PAPER_LABEL_FONTSIZE
AAAI_LEGEND_FONTSIZE = PAPER_LEGEND_FONTSIZE

LEGEND_PT_PER_INCH = 1.1
LEGEND_FONTSIZE_MIN = 12
LEGEND_FONTSIZE_MAX = 32

AXIS_LABEL_PT_PER_INCH = 5.2
AXIS_LABEL_FONTSIZE_MIN = 11
AXIS_LABEL_FONTSIZE_MAX = 32
LINEWIDTH = 2.5
MARKERSIZE = 8
FONTSIZE_LEGEND = 30


@dataclass(frozen=True)
class PaperStyle:
    """One bundle of everything needed to draw a figure at final print size
    for a given (conference, columns) target: physical width, absolute font
    sizes, and line/marker/grid/legend geometry.

    The point is that retargeting a figure -- single column <-> full width,
    AAAI <-> NeurIPS -- is a one-line change (swap the style), not a hunt for
    every hard-coded `linewidth=`, `fontsize=`, `markersize=` and legend
    `ncol` at the call sites. Get one via `paper_style(conference, columns)`
    and thread it through the plot.

    Sizes are absolute print points/inches (the figure is generated at its
    final embedded size, not oversized-then-shrunk), so they're deliberately
    the SAME across conferences and column counts -- the only things that
    change per target are `width_in` and how the legend wraps. Override any
    field per call via `paper_style(..., linewidth=2.0)` etc.
    """

    conference: str
    columns: int
    width_in: float

    # Absolute print font sizes (see PAPER_* above).
    title_fontsize: float = PAPER_TITLE_FONTSIZE
    label_fontsize: float = PAPER_LABEL_FONTSIZE
    tick_fontsize: float = PAPER_TICK_FONTSIZE
    legend_fontsize: float = PAPER_LEGEND_FONTSIZE

    # Line / marker / grid / errorbar geometry, tuned for the small panels of
    # a print-size figure (the previous per-call hard-coded values).
    linewidth: float = 1.2
    markersize: float = 3.5
    markevery_divisor: int = 8
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.4
    band_alpha: float = 0.15
    capsize: float = 2.5
    capthick: float = 1.0

    max_legend_single_row: int = 3

    def legend_ncol(self, n_labels: int) -> int:
        return legend_ncol_for_columns(
            n_labels, self.columns, self.max_legend_single_row
        )

    def n_legend_rows(self, n_labels: int) -> int:
        return -(-n_labels // self.legend_ncol(n_labels))  # ceil division

    def markevery(self, n_points: int) -> int:
        return max(1, n_points // self.markevery_divisor)

    def style_axis(self, ax, *, grid_axis: str = "both") -> None:
        """Apply the shared paper look to one Axes: tick-label size, a faint
        grid, and no top/right spines."""
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid_axis, alpha=self.grid_alpha, linewidth=self.grid_linewidth)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def legend(
        self, fig, handles, labels, *, n_labels: int | None = None, bbox_y=None
    ) -> int:
        """Add the shared top legend, wrapped for this target's width, and
        return the number of legend rows (so the caller can reserve headroom
        in tight_layout)."""
        n = len(labels) if n_labels is None else n_labels
        rows = self.n_legend_rows(n)
        if bbox_y is None:
            bbox_y = 1.0 if rows == 1 else 1.03
        create_figure_legend(
            fig,
            handles,
            labels,
            ncol=self.legend_ncol(n),
            bbox_y=bbox_y,
            fontsize=self.legend_fontsize,
        )
        return rows


def paper_style(
    conference: str = "aaai", columns: int = 1, **overrides
) -> PaperStyle:
    """Build a PaperStyle for a `columns`-wide (1 = narrow, 2 = full text
    width) placement in `conference`'s template. Extra keyword args override
    individual fields, e.g. `paper_style("neurips", 2, linewidth=2.0)`."""
    return PaperStyle(
        conference=conference,
        columns=columns,
        width_in=paper_figure_width_in(columns=columns, conference=conference),
        **overrides,
    )


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
    loc: str = "upper center",
) -> None:
    """Add a legend to `fig`, anchored at (0.5, bbox_y) via `loc` (default
    "upper center": bbox_y is the legend's top edge, for the usual shared-top-
    legend placement). Pass loc="center" with a bbox_y computed from an actual
    Axes position (see plot_regret_and_oracle_grid) to center a second legend
    inside a reserved gap between subplot rows instead."""
    if fontsize is None:
        fig_width, _ = fig.get_size_inches()
        fontsize = np.clip(
            fig_width * LEGEND_PT_PER_INCH, LEGEND_FONTSIZE_MIN, LEGEND_FONTSIZE_MAX
        )

    fig.legend(
        handles,
        labels,
        loc=loc,
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


# --- Shared helpers for the RF tabular-bandit family of plots -------------
# (rf_arm_distribution_plot.py, rf_landscape_plot.py) -- moved out of the now-
# deleted rf_tabular_bandit_plot.py, which these scripts imported from.
_PRETTY_LABELS = {
    "imoss_tpe": "IMOSS-TPE",
    "imoss_tpe_univ": "IMOSS-TPE-univ",
    "imoss": "IMOSS-Random",
    "imoss_random": "IMOSS-Random",
    "random_search": "Random Search",
    "imoss_tabfm": "IMOSS-TabFM",
    "imoss_tabpfn": "IMOSS-TabPFN",
    "imoss_tabpfn_pull": "IMOSS-TabPFN-pull",
    "ucb_air": "UCB-AIR",
    "hier_mab": "Hier-MAB",
}

_CANONICAL_ORDER = [
    "IMOSS-Random",
    "IMOSS-TPE",
    "IMOSS-TPE-univ",
    "IMOSS-TabFM",
    "IMOSS-TabPFN",
    "IMOSS-TabPFN-pull",
    "UCB-AIR",
    "Hier-MAB",
]

_BENCH_NAMES = {
    "rf146822": "segment",
    "rf31": "credit-g",
    "rf167120": "numerai28.6",
    "rf9952": "phoneme",
    "rf3": "kr-vs-kp",
}


def _bench_title(benchmark: str) -> str:
    return _BENCH_NAMES.get(benchmark, benchmark.upper())


def _ordered(present) -> list[str]:
    """Algorithms present AND in _CANONICAL_ORDER, in that order.

    _CANONICAL_ORDER is the source of truth for which algorithms are in the
    comparison: anything not listed there is dropped, so the multi-benchmark
    grids never pick up an un-styled series.
    """
    present = set(present)
    return [a for a in _CANONICAL_ORDER if a in present]


def _style_for(algo: str) -> tuple[str, str]:
    """(color, marker) fixed per algorithm -- thin alias over algorithm_style,
    kept for the call sites that already spell it this way."""
    return algorithm_style(algo)


def _ema(x: np.ndarray, span: float) -> np.ndarray:
    """Causal exponential moving average along the last axis (TensorBoard-style
    smoothing), same length as input. span=1 is a no-op; larger spans smooth
    more heavily at the cost of a slight lag."""
    if span <= 1:
        return x
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def set_research_style():
    plt.rcParams.update(
        {
            # Default color cycle = the Wong colorblind-safe palette, so even
            # plots that don't go through algorithm_style/get_algorithm_color
            # (i.e. anything relying on matplotlib's automatic C0/C1/... colors)
            # come out colorblind-safe rather than using the default tab10.
            "axes.prop_cycle": plt.cycler(color=ALGORITHM_COLORS),
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
