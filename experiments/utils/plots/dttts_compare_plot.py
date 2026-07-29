"""Plot dttts_compare_results.json in the same style as the toy benchmark figure."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from experiments.utils.plots.plot_configs import (
    adaptive_label_fontsize,
    save_figure,
    set_research_style,
)

set_research_style()

RESULTS_FILE = Path(__file__).parents[3] / "results" / "dttts_compare_results.json"

ALGO_STYLE = {
    "IMABO (TPE oracle)": dict(color="#0072B2", ls="-", lw=2.0, label="I-MOSS-TPE"),
    "IMABO (no oracle)": dict(color="#56B4E9", ls="-", lw=2.0, label="IMABO no-oracle"),
    "D-TTTS": dict(color="#D55E00", ls="-", lw=2.0, label="D-TTTS"),
    "Random": dict(color="#333333", ls="--", lw=2.0, label="Random"),
    "TPE": dict(color="#009E73", ls="-", lw=2.0, label="TPE"),
}


def _strip(ax, algo_data: dict, algos: list[str]):
    """Jittered strip plot with a thick mean tick, matching the reference figure."""
    rng = np.random.default_rng(0)
    for x_pos, algo in enumerate(algos):
        vals = np.array(algo_data[algo]["simple_regret_all"])
        style = ALGO_STYLE[algo]
        color = style["color"]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter,
            vals,
            color=color,
            alpha=0.55,
            s=22,
            zorder=3,
            linewidths=0,
        )
        mean = np.mean(vals)
        ax.hlines(
            mean, x_pos - 0.18, x_pos + 0.18, colors=color, linewidth=3.5, zorder=4
        )

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(
        [ALGO_STYLE[a]["label"] for a in algos],
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_dttts_compare(save_fig: bool = False):
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    functions = data["functions"]
    algos = data["algos"]
    results = data["results"]
    n_iter = data["n_iter"]
    n_runs = data["n_runs"]
    sigma = data["sigma"]
    dim = data["dim"]

    sigma_str = f"σ={sigma}" if sigma is not None else "no noise"
    suptitle = (
        f"IMABO vs D-TTTS vs Random on toy functions  "
        f"(dim={dim}, T={n_iter}, {n_runs} seeds, {sigma_str})"
    )

    fig, axes = plt.subplots(
        2,
        len(functions),
        figsize=(6.5 * len(functions), 9),
        gridspec_kw={"height_ratios": [1.6, 1]},
    )

    t = np.arange(1, n_iter + 1)

    for col, fn in enumerate(functions):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # ── top: cumulative regret ──────────────────────────────────────────
        for algo in algos:
            d = results[fn][algo]
            mean = np.array(d["mean_cum_regret"])
            std = np.array(d["std_cum_regret"])
            style = ALGO_STYLE[algo]
            ax_top.plot(
                t,
                mean,
                color=style["color"],
                ls=style["ls"],
                lw=style["lw"],
                label=style["label"],
            )
            ax_top.fill_between(
                t, mean - std, mean + std, color=style["color"], alpha=0.15
            )

        ax_top.set_title(fn, fontsize=13, fontweight="bold")
        if col == len(functions) // 2:  # Middle subplot
            ax_top.set_xlabel("pulls t", fontsize=adaptive_label_fontsize(ax_top))
        if col == 0:
            ax_top.set_ylabel(
                "normalized\ncumulative regret", fontsize=adaptive_label_fontsize(ax_top)
            )
        ax_top.tick_params(labelsize=9)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.grid(alpha=0.25)
        ax_top.set_axisbelow(True)
        ax_top.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

        # ── bottom: simple regret strip ─────────────────────────────────────
        _strip(ax_bot, results[fn], algos)
        if col == 0:
            ax_bot.set_ylabel(
                "normalized\nsimple regret", fontsize=adaptive_label_fontsize(ax_bot)
            )

        # align y-axis right label annotation
        if col == len(functions) - 1:
            ax_top.annotate(
                "lower = better",
                xy=(1.01, 1.0),
                xycoords="axes fraction",
                fontsize=8,
                fontstyle="italic",
                color="#555555",
                ha="left",
                va="top",
            )

    # shared legend inside top-left axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        loc="upper left",
    )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()

    if save_fig:
        out = Path(__file__).parent / "dttts_compare.pdf"
        save_figure(out, bbox_inches="tight", mkdir=False)

    plt.close()


if __name__ == "__main__":
    plot_dttts_compare(save_fig=True)
