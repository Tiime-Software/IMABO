"""Hier-MAB (AutoRAG-HP) as a factored baseline on the discrete RF grid.

Motivation. Every IMABO baseline so far either enumerates the arm set or draws
from a reservoir. AutoRAG-HP's Hier-MAB does neither: it keeps an incumbent and
perturbs one coordinate at a time, so it covers the K=2250 product grid with
only ``sum_i |A_i| = 29`` low-level arms. It is therefore the one published
online-HPO method that runs on this benchmark with no adaptation at all --
non-contextual, reward-only, no validation set -- which makes "we handle
infinitely many arms" insufficient on its own as a distinction. This script
measures the gap rather than asserting it.

The structural assumption under test is separability: crediting the scalar
reward to the single perturbed axis presumes coordinate-wise ascent reaches a
good configuration. The three tasks were expected to separate on exactly that
axis -- ``segment``'s reward "varies mainly with max_depth" (favourable to a
factored method) while ``credit-g``'s strong configurations are "sparse and
depend more strongly on combinations of hyperparameters" (unfavourable).

Reuses ``RFTabularFiniteBenchmark`` for the data, so the arm set, the
Bernoulli(accuracy) reward model, and the exact-regret computation are the same
ones the paper's own discrete experiment uses; the low-level arm sets are the
benchmark's own axes, so no grid values are invented. Where per-seed result
JSONs from ``rf_arm_distribution_experiment.py`` already exist, they are read
rather than recomputed, so Hier-MAB is compared against the published runs of
IMOSS-TabPFN/TPE/Random and UCB-AIR instead of a reimplementation.

Usage (from repo root):
    python -m experiments.factored_baseline_experiment
    python -m experiments.factored_baseline_experiment --n-seeds 20 --plot
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from experiments.baselines.hier_mab import HierMAB
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark

RESULT_DIR = Path(__file__).parent.parent / "results" / "factored_baseline"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
STORED_DIR = Path(__file__).parent.parent / "results" / "hpo_finite_arm_distribution"

# bm_id -> the task name used in the paper
TASKS = {146822: "segment", 31: "credit-g", 167120: "numerai28.6"}

# Stored-result slugs to compare against, in reporting order.
REFERENCE_SLUGS = {
    "imoss_tabpfn": "IMOSS-TabPFN",
    "imoss_tpe": "IMOSS-TPE",
    "imoss_random": "IMOSS-Random",
    "ucb_air": "UCB-AIR",
}

T_DEFAULT = 5000
# Same base seed as rf_arm_distribution_experiment: run i of every method
# uses seed 42 + i for both the optimizer and the benchmark's Bernoulli
# stream, so Hier-MAB's runs are paired with the stored reference runs.
BASE_SEED = 42


def run_hier_mab(bm_id: int, n_iterations: int, n_seeds: int) -> dict:
    """Run Hier-MAB for `n_seeds` paired runs on one task.

    Each seed is checkpointed to STORED_DIR under the same filename scheme as
    rf_arm_distribution_experiment's per-run JSONs (slug ``hier_mab``), so
    reruns skip finished seeds and the shared plotting loaders
    (rf_arm_distribution_plot._load_trace_field) pick the series up like any
    other method's.
    """
    bench = RFTabularFiniteBenchmark(bm_id=bm_id, seed=0)
    search_space = bench.get_search_space()

    for i in range(n_seeds):
        path = STORED_DIR / f"rf{bm_id}_hier_mab_{n_iterations}iters_run{i}.json"
        if path.exists():
            continue
        seed = BASE_SEED + i
        bench.reset_noise(seed)
        opt = HierMAB(search_space, seed=seed)

        regrets = []
        for _ in range(n_iterations):
            x = opt.suggest()
            opt.observe(bench(x))
            regrets.append(bench.regret(x))

        best = opt.best_config
        with open(path, "w") as fh:
            json.dump(
                {
                    "regrets": regrets,
                    "simple_regrets": float(bench.regret(best)),
                    "best_config": best,
                    "best_reward": float(bench.mean_reward(best)),
                },
                fh,
            )

    return load_stored(bm_id, "hier_mab", n_iterations)


def load_stored(bm_id: int, slug: str, n_iterations: int) -> dict | None:
    """Read the per-seed JSONs written by rf_arm_distribution_experiment."""
    files = sorted(
        glob.glob(str(STORED_DIR / f"rf{bm_id}_{slug}_{n_iterations}iters_run*.json"))
    )
    if not files:
        return None
    cumulative, simple, traces = [], [], []
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        cumulative.append(float(np.sum(data["regrets"])))
        simple.append(float(data["simple_regrets"]))
        traces.append(data["regrets"])
    cumulative_runs = np.cumsum(np.array(traces), axis=1)
    return {
        "cumulative": cumulative,
        "simple": simple,
        "mean_trace": np.mean(np.array(traces), axis=0).tolist(),
        "cumulative_mean": cumulative_runs.mean(axis=0).tolist(),
        "cumulative_sd": cumulative_runs.std(axis=0, ddof=1).tolist(),
        "n_seeds": len(files),
    }


def summarize(results: dict) -> None:
    print(f"\n{'task':13s}{'algorithm':16s}{'cumulative':>20s}{'simple':>18s}")
    print("-" * 67)
    for task, algos in results.items():
        for name, data in algos.items():
            c = np.array(data["cumulative"])
            s = np.array(data["simple"])
            print(
                f"{task:13s}{name:16s}{c.mean():11.1f} +-{c.std():5.1f}"
                f"{s.mean():12.4f} +-{s.std():.4f}"
            )
        print()


def plot(results: dict, path: Path, band: str = "ci95") -> None:
    """Paper figure: cumulative regret with an across-seed uncertainty band,
    one panel per task, generated at final print size (a full-width AAAI
    ``figure*``) with the shared per-algorithm styles -- same conventions as
    plot_regret_and_oracle_grid in rf_arm_distribution_plot.py.

    ``band`` is ``"ci95"`` (95% CI of the mean), ``"sd"`` (+/-1 across-seed
    standard deviation), or ``None``. The band comes from each seed's own
    cumulative curve, aggregated over seeds -- not from within-run dispersion
    of the per-round regret, which understates it badly (measured across-seed
    spread at t=5000 is 9-21% of the mean, and several method gaps here are
    within it).

    Seed counts differ per method: the surrogate oracle IMOSS-TabPFN is read
    from stored runs and needs a gated model to regenerate, so it stays at its
    original count while the cheap methods are rerun with more. Each legend
    entry therefore carries its own n.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.utils.plots.plot_configs import (
        algorithm_color,
        algorithm_marker,
        create_figure_legend,
        paper_style,
    )

    style = paper_style(conference="aaai", columns=2, markevery_divisor=10)

    tasks = list(results)
    n = len(tasks)
    subplot_row_in = 1.55
    legend_band_in = 0.28
    height_in = subplot_row_in + legend_band_in
    fig, axes = plt.subplots(1, n, figsize=(style.width_in, height_in))
    if n == 1:
        axes = [axes]

    seen: dict = {}
    for idx, (ax, task) in enumerate(zip(axes, tasks)):
        for name, data in results[task].items():
            mean = np.asarray(
                data.get("cumulative_mean") or np.cumsum(data["mean_trace"])
            )
            steps = np.arange(1, len(mean) + 1)
            color = algorithm_color(name)
            n_seeds = data.get("n_seeds", len(data["cumulative"]))
            (line,) = ax.plot(
                steps,
                mean,
                label=f"{name} (n={n_seeds})",
                color=color,
                marker=algorithm_marker(name),
                markevery=style.markevery(len(mean)),
                markersize=style.markersize,
                linewidth=style.linewidth,
                linestyle="--" if name == "Hier-MAB" else "-",
            )
            seen.setdefault(f"{name} (n={n_seeds})", line)
            sd = data.get("cumulative_sd")
            if band is not None and sd is not None and n_seeds > 1:
                sd = np.asarray(sd)
                half = sd if band == "sd" else 1.96 * sd / np.sqrt(n_seeds)
                ax.fill_between(
                    steps,
                    mean - half,
                    mean + half,
                    color=color,
                    alpha=style.band_alpha,
                    linewidth=0,
                )
        ax.set_title(
            task, fontweight="bold", fontsize=style.title_fontsize, pad=4
        )
        if idx == n // 2:
            ax.set_xlabel(
                "Iteration", fontweight="bold", fontsize=style.label_fontsize
            )
        style.style_axis(ax)
    axes[0].set_ylabel(
        "Cumulative\nRegret", fontweight="bold", fontsize=style.label_fontsize
    )

    create_figure_legend(
        fig,
        list(seen.values()),
        list(seen.keys()),
        ncol=len(seen),
        bbox_y=1.0,
        fontsize=style.legend_fontsize,
    )
    fig.tight_layout(
        rect=[0, 0, 1, subplot_row_in / height_in], w_pad=0.6
    )
    fig.savefig(path, bbox_inches="tight")
    print(f"figure saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=T_DEFAULT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--band",
        choices=("ci95", "sd", "none"),
        default="ci95",
        help="across-seed uncertainty band drawn on the regret curves",
    )
    args = parser.parse_args()
    band = None if args.band == "none" else args.band

    results: dict = {}
    for bm_id, task in TASKS.items():
        results[task] = {}
        for slug, name in REFERENCE_SLUGS.items():
            stored = load_stored(bm_id, slug, args.n_iterations)
            if stored is not None:
                results[task][name] = stored
        results[task]["Hier-MAB"] = run_hier_mab(
            bm_id, args.n_iterations, args.n_seeds
        )

    out = RESULT_DIR / f"factored_baseline_{args.n_iterations}iters.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"results saved to {out}")

    summarize(results)
    if args.plot:
        plot(
            results,
            RESULT_DIR / f"factored_baseline_{args.n_iterations}iters.pdf",
            band=band,
        )


if __name__ == "__main__":
    main()
