"""Toy counterexamples: smooth coordination barriers for factored search.

Two d-dimensional families of Bernoulli bandits over [0, 1]^d (B = 1,
matching the observation model of the paper's other experiments), both built
so that reaching the global mode from the local one requires moving EVERY
coordinate together. A method that credits the scalar reward to one
coordinate at a time -- Hier-MAB's structural assumption -- sees each
unilateral step as a regression and stalls on the local mode; the IMOSS
oracles and the tree-based continuous bandits make no such assumption.

Both families place their local mode at MODE_LO * 1 and their global mode at
MODE_HI * 1 with MODE_LO = 1/(2 pi): irrational coordinates, so no uniform
grid contains either mode exactly and Hier-MAB never benefits from a grid
point sitting on a mode.

1. The Gaussian landscape (``family_d2``):

       mu_d(x) = 0.9 * exp(-||x - MODE_HI * 1||^2 / (2 sigma_d^2))
               + 0.6 * exp(-||x - MODE_LO * 1||^2 / (2 sigma_d^2)),

   with sigma_d scaled so the profile of mu_d along the line through both
   modes is identical for every d: growing d changes only how many
   coordinates must move together, not the reward signal along the
   coordinated path.

2. The warped multilinear landscape (``prod_d2``):

       p_d(x) = 1/2 - (c/d) * sum_i t(x_i) + 2c * prod_i t(x_i),

   with 0 < c < 1/4 (c = 0.2) and the raised-cosine warp t (see prod_warp),
   which has its unique zero at MODE_LO and unique one at MODE_HI.
   Multilinear in the warped coordinates, so the global mode 1/2 + c sits at
   MODE_HI * 1, the local mode 1/2 at MODE_LO * 1, and every mixed t-vertex
   strictly below the local mode.

Hier-MAB runs at grid resolutions m in {11, 101} on both families. Regret is
computed on the noiseless mu against its global maximum; both figures show
cumulative regret on a log y-axis.

Usage (from repo root):
    python -m experiments.coordination_barrier_experiment --plot
    python -m experiments.coordination_barrier_experiment --quick  # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.hier_mab import HierMAB
from experiments.baselines.stroquool import TimedOptimizer, hoo_t, stosoo, stroquool
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results" / "coordination_barrier"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BETA = 0.5
T_DEFAULT = 5000
N_SEEDS_DEFAULT = 30
BASE_SEED = 42
N_JOBS = 8

FAMILY_DIMS = (2,)
# Both families place their modes at MODE_LO * 1 and MODE_HI * 1. The values
# are chosen IRRATIONAL -- 1/(2 pi) and 1 - 1/(2 pi) -- so that no uniform
# grid {0, 1/(m-1), ..., 1} contains either mode exactly: Hier-MAB never gets
# the advantage of a grid point sitting on a mode, only ever a nearby one.
MODE_LO = 1.0 / (2.0 * math.pi)  # ~0.15915
MODE_HI = 1.0 - MODE_LO  # ~0.84085
# Base width relative to the mode separation: the separation is
# (HI - LO) * sqrt(d), so scaling the reference width 0.35 by (HI - LO) keeps
# the same landscape shape relative to the separation as a unit-diagonal
# construction with sigma = 0.35 would have.
FAMILY_SIGMA_BASE = 0.35 * (MODE_HI - MODE_LO)

# Coupling strength of the multilinear family (must be in (0, 1/4) for p_d to
# stay a valid probability).
PROD_C = 0.2
# Hier-MAB grid resolutions, shared by both families.
HIER_GRIDS = (11, 101)


def family_sigma(d: int) -> float:
    """Dimension-scaled Gaussian width: sigma_d = 0.21 * sqrt(d / 2).

    With this scaling the profile of mu_d along the diagonal through both
    modes is IDENTICAL for every d, so growing d changes only the geometry --
    how many coordinates must move together -- and not the reward signal
    available along the coordinated path. A fixed sigma would instead make
    mu ~ 0 over almost all of the cube at large d, collapsing the comparison
    into 'who settles on the local mode fastest'.
    """
    return FAMILY_SIGMA_BASE * math.sqrt(d / 2.0)


def make_mu_family(d: int) -> Callable[[np.ndarray], float]:
    s2 = 2 * family_sigma(d) ** 2

    def mu_family(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return float(
            0.9 * math.exp(-float(np.sum((z - MODE_HI) ** 2)) / s2)
            + 0.6 * math.exp(-float(np.sum((z - MODE_LO) ** 2)) / s2)
        )

    return mu_family


def _mu_star_family(d: int) -> float:
    """Global maximum of mu_d. Both Gaussians are radial around two points of
    the main diagonal, so any maximizer lies ON that diagonal: for x off the
    diagonal, projecting onto it decreases both squared distances. A fine 1D
    search along x = t * 1 therefore finds the global maximum."""
    t = np.linspace(0.0, 1.0, 200001)
    s2 = 2 * family_sigma(d) ** 2
    v = 0.9 * np.exp(-d * (t - MODE_HI) ** 2 / s2) + 0.6 * np.exp(
        -d * (t - MODE_LO) ** 2 / s2
    )
    return float(v.max())


def prod_warp(z: np.ndarray) -> np.ndarray:
    """Per-coordinate raised-cosine warp t(x) = (1 - cos(pi (x-LO)/(HI-LO)))/2.

    Smooth on [0, 1], with its unique zero at x = MODE_LO and unique one at
    x = MODE_HI (the argument pi (x-LO)/(HI-LO) stays inside (-pi, 2 pi) for
    x in [0, 1], where cos attains 1 resp. -1 only there), rising again
    toward the domain edges. Composing the multilinear form with t moves its
    two meaningful vertices to the off-grid interior points MODE_LO * 1 and
    MODE_HI * 1 while preserving the structure: t maps [0, 1] onto [0, 1], so
    the composition's range and its maximum 1/2 + c are unchanged.
    """
    return 0.5 * (1.0 - np.cos(np.pi * (z - MODE_LO) / (MODE_HI - MODE_LO)))


def make_mu_prod(d: int) -> Callable[[np.ndarray], float]:
    """Warped multilinear family: with t = prod_warp,

        p_d(x) = 1/2 - (c/d) sum_i t(x_i) + 2c prod_i t(x_i).

    Multilinear in the warped coordinates t_i in [0, 1], so extrema sit at
    t-vertices: t = 1 (i.e. x = MODE_HI * 1) gives the global mode 1/2 + c,
    every mixed t-vertex with k ones gives 1/2 - c k / d < 1/2, and t = 0
    (x = MODE_LO * 1) is the local mode at 1/2. Any single-coordinate move
    away from either mode loses reward immediately, while the 2c bonus
    requires ALL coordinates at once.
    """

    def mu_prod(z: np.ndarray) -> float:
        t = prod_warp(np.asarray(z, dtype=float))
        return float(0.5 - (PROD_C / d) * float(t.sum()) + 2 * PROD_C * float(t.prod()))

    return mu_prod


@dataclass
class Landscape:
    """One benchmark: a noiseless mu over [0, 1]^dim plus its maximum."""

    tag: str
    dim: int
    mu: Callable[[np.ndarray], float]
    mu_star: float
    hier_grids: tuple[int, ...]
    search_space: dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.search_space = {
            f"x{i + 1}": {"lower": 0.0, "upper": 1.0, "log": False}
            for i in range(self.dim)
        }

    @property
    def param_names(self) -> list[str]:
        return [f"x{i + 1}" for i in range(self.dim)]


LANDSCAPES: dict[str, Landscape] = {
    **{
        f"family_d{d}": Landscape(
            f"family_d{d}", d, make_mu_family(d), _mu_star_family(d), HIER_GRIDS
        )
        for d in FAMILY_DIMS
    },
    # The multilinear mu* is exact: the function is multilinear, so its
    # maximum sits at a vertex -- 1/2 + c at the all-ones corner.
    **{
        f"prod_d{d}": Landscape(
            f"prod_d{d}", d, make_mu_prod(d), 0.5 + PROD_C, HIER_GRIDS
        )
        for d in FAMILY_DIMS
    },
}


def active_landscapes() -> list[Landscape]:
    """Landscapes to run/plot (excludes the warped multilinear ``prod_*`` family)."""
    return [land for land in LANDSCAPES.values() if not land.tag.startswith("prod_")]


IMOSS_METHODS = {
    "imoss_random": "IMOSS-Random",
    "imoss_tpe": "IMOSS-TPE",
    "imoss_tabpfn": "IMOSS-TabPFN",
}
# "imoss_tpe_uni" (IMABO with multivariate=False: independent per-coordinate
# Parzen estimators, a factored proposal) stays available in _build for
# offline ablations but is not part of the paper's comparison.
TREE_METHODS = {
    "stosoo": "StoSOO",
    "hoo_t": "HOO-T",
    "stroquool": "Stroquool",
}


def hier_methods(land: Landscape) -> dict[str, str]:
    return {f"hier_mab_{m}": f"Hier-MAB-{m}" for m in land.hier_grids}


def methods_for(land: Landscape, skip_tabpfn: bool = False) -> dict[str, str]:
    out = {**hier_methods(land), **IMOSS_METHODS, **TREE_METHODS}
    if skip_tabpfn:
        out.pop("imoss_tabpfn", None)
    return out


def _build(land: Landscape, slug: str, n_iterations: int, seed: int, tabpfn_model):
    """(optimizer, is_tree) for one method slug on one landscape."""
    if slug.startswith("hier_mab_"):
        m = int(slug.rsplit("_", 1)[1])
        return HierMAB(land.search_space, n_points=m, seed=seed), False
    if slug == "imoss_random":
        return (
            IMABO(
                search_space=land.search_space,
                seed=seed,
                multivariate=True,
                use_tpe=False,
                beta=BETA,
            ),
            False,
        )
    if slug == "imoss_tpe":
        return (
            IMABO(
                search_space=land.search_space, seed=seed, multivariate=True, beta=BETA
            ),
            False,
        )
    if slug == "imoss_tpe_uni":
        return (
            IMABO(
                search_space=land.search_space, seed=seed, multivariate=False, beta=BETA
            ),
            False,
        )
    if slug == "imoss_tabpfn":
        from imabo.tabpfn_optimizer import IMABOTabPFN

        return (
            IMABOTabPFN(
                search_space=land.search_space,
                seed=seed,
                tabpfn_model=tabpfn_model,
                beta=BETA,
                n_estimators=4,
            ),
            False,
        )
    if slug == "stosoo":
        return TimedOptimizer(stosoo, n_iterations, land.dim), True
    if slug == "hoo_t":
        return TimedOptimizer(hoo_t, n_iterations, land.dim, rho=0.4, nu1=10.0), True
    if slug == "stroquool":
        return TimedOptimizer(stroquool, n_iterations, land.dim), True
    raise ValueError(f"unknown method: {slug!r}")


def _ckpt(land: Landscape, slug: str, n_iterations: int, run_idx: int) -> Path:
    return RESULT_DIR / f"{land.tag}_{slug}_{n_iterations}iters_run{run_idx}.json"


def run_one(
    land: Landscape, slug: str, n_iterations: int, run_idx: int, tabpfn_model
) -> dict:
    """One (landscape, method, seed) run, checkpointed. Regret is noiseless;
    the reward fed back is a single Bernoulli draw from a per-seed stream."""
    path = _ckpt(land, slug, n_iterations, run_idx)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    seed = BASE_SEED + run_idx
    rng = np.random.default_rng(seed)
    opt, is_tree = _build(land, slug, n_iterations, seed, tabpfn_model)

    regrets = []
    for _ in range(n_iterations):
        if is_tree and opt.done:
            break
        s = opt.suggest()
        z = (
            np.asarray(s, dtype=float)
            if is_tree
            else np.array([s[p] for p in land.param_names], dtype=float)
        )
        p = land.mu(z)
        reward = float(rng.random() < p)
        if is_tree:
            opt.observe(s, reward)
        else:
            opt.observe(reward)
        regrets.append(land.mu_star - p)

    data = {"regrets": regrets}
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def run_all(
    n_iterations: int, n_seeds: int, n_jobs: int = N_JOBS, skip_tabpfn: bool = False
) -> None:
    # One shared TabPFN handle, loaded only if some TabPFN seed is missing.
    tabpfn_model = None
    if not skip_tabpfn:
        pending_tabpfn = any(
            not _ckpt(land, "imoss_tabpfn", n_iterations, i).exists()
            for land in active_landscapes()
            for i in range(n_seeds)
        )
        if pending_tabpfn:
            from imabo.tabpfn_optimizer import load_tabpfn

            tabpfn_model = load_tabpfn()
            print("TabPFN ready (checkpoint cached; shared across seeds).")

    for land in active_landscapes():
        for slug, name in methods_for(land, skip_tabpfn).items():
            pending = [
                i
                for i in range(n_seeds)
                if not _ckpt(land, slug, n_iterations, i).exists()
            ]
            if not pending:
                continue
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(run_one)(land, slug, n_iterations, i, tabpfn_model)
                for i in tqdm(pending, desc=f"{land.tag}/{name}", leave=False)
            )


def load_runs(
    land: Landscape, slug: str, n_iterations: int, n_seeds: int
) -> list[list[float]]:
    traces = []
    for i in range(n_seeds):
        path = _ckpt(land, slug, n_iterations, i)
        if path.exists():
            with open(path) as f:
                traces.append(json.load(f)["regrets"])
    return traces


def summarize(n_iterations: int, n_seeds: int) -> None:
    for land in active_landscapes():
        print(f"\n{land.tag} (mu* = {land.mu_star:.4f})")
        print(f"{'method':>14}{'cumulative regret':>24}{'avg regret':>12}{'evals':>8}")
        print("-" * 58)
        for slug, name in methods_for(land).items():
            traces = load_runs(land, slug, n_iterations, n_seeds)
            if not traces:
                continue
            tot = np.array([float(np.sum(t)) for t in traces])
            avg = np.array([float(np.mean(t)) for t in traces])
            print(
                f"{name:>14}{tot.mean():13.1f} +- {tot.std():5.1f}"
                f"{avg.mean():12.4f}"
                f"{int(np.mean([len(t) for t in traces])):>8}"
            )


# Linestyles distinguishing the Hier-MAB grid resolutions (they share the
# family's purple in ALGORITHM_STYLES; marker + linestyle tell them apart).
_HIER_LINESTYLE = {
    "hier_mab_6": "-",
    "hier_mab_11": "-",
    "hier_mab_21": "--",
    "hier_mab_101": ":",
    "imoss_tpe_uni": "--",
}


def _draw_panel(ax, land, methods, n_iterations, n_seeds, style, seen):
    from experiments.utils.plots.plot_configs import algorithm_style

    for slug, name in methods.items():
        traces = load_runs(land, slug, n_iterations, n_seeds)
        if not traces:
            continue
        # Tree methods can stop early (StroquOOL); align on the shortest
        # trace within the method, like the LR/SVM figure.
        n_min = min(len(t) for t in traces)
        runs = np.array([t[:n_min] for t in traces])
        steps = np.arange(1, n_min + 1)
        mean = np.cumsum(runs.mean(axis=0))
        color, marker = algorithm_style(name)
        (line,) = ax.plot(
            steps,
            mean,
            color=color,
            label=name,
            marker=marker,
            markevery=style.markevery(n_min),
            markersize=style.markersize,
            linewidth=style.linewidth,
            linestyle=_HIER_LINESTYLE.get(slug, "-"),
        )
        seen.setdefault(name, line)


def plot_landscapes(path: Path) -> None:
    """Heatmaps of the success probability of the d = 2 member of each
    family, local mode marked with a circle and global mode with a star.
    Per-panel color scales: the ranges differ ([~0, 0.9] Gaussian vs
    [0.3, 0.7] multilinear) and the point is each landscape's mode structure,
    not cross-panel value comparison."""
    import matplotlib.pyplot as plt

    from experiments.utils.plots.plot_configs import paper_style

    style = paper_style(conference="aaai", columns=2)
    panels = [
        ("family_d2", "Gaussian", (MODE_LO, MODE_HI)),
        # ("prod_d2", "multilinear", (MODE_LO, MODE_HI)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(0.52 * style.width_in, 1.75))
    g = np.linspace(0.0, 1.0, 301)
    for ax, (tag, title, (lo, hi)) in zip(axes, panels):
        land = LANDSCAPES[tag]
        z = np.array([[land.mu(np.array([x, y])) for x in g] for y in g])
        im = ax.imshow(
            z, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", aspect="equal"
        )
        ax.scatter(
            [lo],
            [lo],
            marker="o",
            s=28,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            clip_on=False,
            zorder=5,
        )
        ax.scatter(
            [hi],
            [hi],
            marker="*",
            s=70,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            clip_on=False,
            zorder=5,
        )
        ax.set_title(title, fontweight="bold", fontsize=style.title_fontsize, pad=4)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.grid(False)  # the shared style's grid just striples the heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        cbar.ax.tick_params(labelsize=style.tick_fontsize)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    print(f"figure saved to {path}")


def plot_results_2x2(n_iterations: int, n_seeds: int, path: Path) -> None:
    """The section's single results figure: rows = landscapes (Gaussian,
    warped multilinear), columns = comparison group (IMOSS family,
    tree-based continuous bandits), both Hier-MAB grid resolutions in every
    panel, cumulative regret on a log y-axis. Uncomment/comment rows in
    ``land_rows`` to switch between the 2x2 (both landscapes) and 1x2 (one
    landscape) layouts -- the grid, figure height, and legend all adapt."""
    import matplotlib.pyplot as plt

    from experiments.utils.plots.plot_configs import create_figure_legend, paper_style

    style = paper_style(conference="aaai", columns=2, markevery_divisor=10)
    land_rows = [
        (LANDSCAPES["family_d2"], "Gaussian"),
    ]
    col_groups = [
        (IMOSS_METHODS, "vs. IMOSS"),
        (TREE_METHODS, "vs. continuous bandits"),
    ]
    n_rows = len(land_rows)

    # A single row would otherwise get the same per-row height as one row of
    # the 2x2 layout, which reads as too flat/squished at the full column
    # width -- give it more height per row instead.
    subplot_row_in = 1.35 if n_rows > 1 else 2.1
    legend_band_in = 0.55
    height_in = n_rows * subplot_row_in + legend_band_in
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(style.width_in, height_in),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    seen: dict = {}
    for r, (land, row_label) in enumerate(land_rows):
        hier = hier_methods(land)
        for c, (group, col_title) in enumerate(col_groups):
            ax = axes[r][c]
            _draw_panel(ax, land, {**group, **hier}, n_iterations, n_seeds, style, seen)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(
                    col_title,
                    fontweight="bold",
                    fontsize=style.title_fontsize,
                    pad=4,
                )
            if r == n_rows - 1:
                ax.set_xlabel(
                    "Iteration", fontweight="bold", fontsize=style.label_fontsize
                )
            style.style_axis(ax)
        axes[r][0].set_ylabel(
            f"{row_label}\nCumulative Regret",
            fontweight="bold",
            fontsize=style.label_fontsize,
        )

    labels = list(seen)
    create_figure_legend(
        fig,
        [seen[l] for l in labels],
        labels,
        ncol=-(-len(labels) // 2),
        bbox_y=1.01,
        fontsize=style.legend_fontsize,
    )
    fig.tight_layout(
        rect=[0, 0, 1, (n_rows * subplot_row_in) / height_in], w_pad=1.0, h_pad=0.8
    )
    fig.savefig(path, bbox_inches="tight")
    print(f"figure saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--n-iterations", type=int, default=T_DEFAULT)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="smoke test: T=200, 2 seeds, no TabPFN"
    )
    args = parser.parse_args()

    n_iterations, n_seeds = args.n_iterations, args.n_seeds
    if args.quick:
        n_iterations, n_seeds = 200, 2

    for land in active_landscapes():
        print(f"{land.tag}: mu* = {land.mu_star:.6f}")
    run_all(n_iterations, n_seeds, n_jobs=args.n_jobs, skip_tabpfn=args.quick)
    summarize(n_iterations, n_seeds)
    if args.plot:
        # plot_landscapes(RESULT_DIR / "coordination_landscapes.pdf")
        plot_results_2x2(
            n_iterations,
            n_seeds,
            RESULT_DIR / f"coordination_2d_{n_iterations}iters.pdf",
        )


if __name__ == "__main__":
    main()
