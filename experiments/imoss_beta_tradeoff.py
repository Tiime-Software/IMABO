"""I-MOSS (IMABO no-oracle) beta sweep as a regret trade-off trajectory.

For each beta (arm-opening exponent), run IMABO(use_tpe=False, beta) and record,
on a time grid, BOTH:
  * cumulative regret up to t   (grows with t)
  * simple regret at t          = fmax_norm - norm(fn0(best_config_at_t))
Averaged over seeds.  Plotting simple-regret (y) against cumulative-regret (x)
parameterised by t gives one trajectory per beta: as t grows a run moves right
(pays more cumulative regret) and down (identifies a better arm).  The endpoint
is the final (cum, simple) pair -- the single point we scattered before.

Usage: python -m experiments.imoss_beta_tradeoff [n_iter] [n_runs] [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.benchmarks.toys.toy_functions import ObjectiveFunctions
from experiments.baselines.ucb_air import UCBAIR, MOSSAIR
from experiments.baselines.qrm2 import QRM2
from imabo import IMABO
from experiments.ucbair_compare import reward_bounds

BETAS = [0.4, 0.5, 0.6, 0.7, 0.8]      # "not too many"
# trajectories to record: the I-MOSS beta family + the two fixed-schedule AIR baselines
ALGOS = [f"IMOSS b={b}" for b in BETAS] + ["UCB-AIR", "MOSS-AIR", "QRM2"]
FUNCS = ["sin1", "garland", "rastrigin"]
N_GRID = 60                             # time points at which to record simple regret


def one_run(function_name, dim, n_iter, seed, bounds, grid):
    obj = ObjectiveFunctions(dim=dim, noise_seed=seed)
    func = obj.get_function_by_name(function_name)
    fn0 = obj.get_function_by_name(function_name, noise=False)
    fmax = obj.get_theoretical_max(function_name)
    ss = obj.get_search_space(function_name)
    lo, hi = bounds
    span = hi - lo
    norm = lambda v: min(1.0, max(0.0, (v - lo) / span))
    fmaxn = norm(fmax * dim)
    grid_set = set(grid)

    def make(name):
        if name == "UCB-AIR":
            return UCBAIR(search_space=ss, beta=1.0, seed=seed)
        if name == "MOSS-AIR":
            return MOSSAIR(search_space=ss, beta=1.0, seed=seed)
        if name == "QRM2":
            return QRM2(search_space=ss, seed=seed)
        # otherwise an I-MOSS beta label like "IMOSS b=0.5"
        b = float(name.split("=")[1])
        return IMABO(search_space=ss, seed=seed, multivariate=True, use_tpe=False, beta=b)

    out = {}
    for name in ALGOS:
        opt = make(name)
        cum = 0.0
        cum_grid = []
        sr_grid = []
        for i in range(1, n_iter + 1):
            x = opt.suggest()
            opt.observe(norm(func(x)))
            cum += fmaxn - norm(fn0(x))
            if i in grid_set:
                bx = opt.best_config
                sr = fmaxn - norm(fn0(bx)) if bx is not None else fmaxn
                cum_grid.append(cum)
                sr_grid.append(sr)
        out[name] = {"cum": cum_grid, "sr": sr_grid}
    return out


def main():
    dim = 4
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    base_seed = 42
    # log-spaced time grid (denser early, where the curves bend most)
    grid = sorted(set(np.unique(np.geomspace(20, n_iter, N_GRID).astype(int)).tolist()))

    results = {}
    for fn in FUNCS:
        bounds = reward_bounds(fn, dim)
        runs = Parallel(n_jobs=8, backend="threading")(
            delayed(one_run)(fn, dim, n_iter, base_seed + r * 1000, bounds, grid)
            for r in range(n_runs)
        )
        agg = {}
        for name in ALGOS:
            cum = np.mean([np.array(run[name]["cum"]) for run in runs], axis=0)
            sr = np.mean([np.array(run[name]["sr"]) for run in runs], axis=0)
            agg[name] = {"cum": cum.tolist(), "sr": sr.tolist()}
        results[fn] = agg
        print(f"[{fn}] done "
              + " ".join(f"{name}:({agg[name]['cum'][-1]:.0f},{agg[name]['sr'][-1]:.3f})"
                         for name in ALGOS))

    meta = {"dim": dim, "n_iter": n_iter, "n_runs": n_runs, "betas": BETAS,
            "algos": ALGOS, "functions": FUNCS, "grid": grid, "results": results}
    outpath = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("imoss_beta_tradeoff.json")
    outpath.write_text(json.dumps(meta))
    print("WROTE", outpath)
    plot(meta, outpath.with_suffix(".png"))
    print("WROTE", outpath.with_suffix(".png"))


def plot(meta, pngpath):
    """Simple regret (y) vs average per-round regret = cum/t (x); one trajectory
    per algorithm, parameterised by t.  As t grows a run moves down-and-left
    (both metrics improve as the early exploration cost is averaged away); the
    marked endpoint is the final (T) state.  The I-MOSS beta family is drawn on
    a viridis ramp; UCB-AIR and MOSS-AIR (fixed ceil(sqrt(t)) schedule) are drawn
    as distinct dashed curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = meta["results"]; betas = meta["betas"]; funcs = meta["functions"]
    grid = np.array(meta["grid"])
    cmap = plt.get_cmap("viridis")
    style = {f"IMOSS b={b}": dict(color=cmap(i / (len(betas) - 1)), ls="-", lw=1.7,
                                  label=f"I-MOSS \u03b2={b}")
             for i, b in enumerate(betas)}
    style["UCB-AIR"] = dict(color="#c0392b", ls=(0, (4, 2)), lw=1.9, label="UCB-AIR")
    style["MOSS-AIR"] = dict(color="#111111", ls=(0, (1, 1)), lw=1.9, label="MOSS-AIR")
    style["QRM2"] = dict(color="#e8992a", ls=(0, (5, 1, 1, 1)), lw=1.9, label="QRM2")
    algos = meta.get("algos", list(style))

    fig, axes = plt.subplots(1, len(funcs), figsize=(4.5 * len(funcs), 4.9))
    for j, fn in enumerate(funcs):
        ax = axes[j]
        for name in algos:
            st = style[name]
            cum = np.array(res[fn][name]["cum"]); sr = np.array(res[fn][name]["sr"])
            avg = cum / grid                       # average per-round regret
            ax.plot(avg, sr, color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.9,
                    zorder=3 if name in ("UCB-AIR", "MOSS-AIR") else 2)
            ax.scatter([avg[-1]], [sr[-1]], color=st["color"], s=40, zorder=5,
                       edgecolor="white", lw=0.8)
        ax.set_title(fn); ax.set_xlabel("average per-round regret  (cum. regret / t)")
        if j == 0:
            ax.set_ylabel("simple regret")
        ax.margins(0.10)
    handles = [plt.Line2D([], [], color=style[n]["color"], ls=style[n]["ls"],
                          lw=style[n]["lw"], label=style[n]["label"]) for n in algos]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.6, loc="upper right")
    fig.suptitle("Regret trade-off: simple regret vs average per-round regret over t "
                 "\u2014 I-MOSS \u03b2 family + UCB-AIR / MOSS-AIR", fontsize=9.6, y=1.02)
    fig.tight_layout()
    fig.savefig(pngpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
