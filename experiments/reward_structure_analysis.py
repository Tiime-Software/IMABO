"""How separable is each benchmark's reward landscape? An offline diagnostic.

The discrete RF benchmarks give the mean reward of *every* configuration on the
grid, so the structure a bandit has to exploit can be measured directly, before
running any bandit at all. Two quantities, both computed on the 4-way reward
tensor (max_depth x max_features x min_samples_leaf x min_samples_split):

* **Additive share** -- the fraction of reward variance explained by main
  effects alone (a one-way ANOVA decomposition). This is exactly the assumption
  a factored, coordinate-at-a-time tuner such as AutoRAG-HP's Hier-MAB relies
  on: if the reward is additive across axes, crediting a scalar reward to the
  single perturbed coordinate is sound.
* **Multilinear (Tucker) rank** -- per-mode singular spectra of the unfolded
  tensor. Rank 1 on every mode means the landscape is essentially separable;
  rank 2+ means axes interact in a way an additive model cannot represent, but
  which a low-rank tensor bandit still could.

Why this matters: it turns "credit-g's good configurations depend on
combinations of hyperparameters" from a qualitative remark into a number, and
it predicts which competitor should do well on which task.

Usage (from repo root):
    python -m experiments.reward_structure_analysis
    python -m experiments.reward_structure_analysis --plot
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.benchmarks.rf_tabular_bandit import PARAM_NAMES, RFTabularFiniteBenchmark

RESULT_DIR = Path(__file__).parent.parent / "results" / "reward_structure"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {146822: "segment", 31: "credit-g", 167120: "numerai28.6"}


def reward_tensor(bm_id: int) -> tuple[np.ndarray, list[list]]:
    """Dense reward tensor over the benchmark's own discretised axes."""
    bench = RFTabularFiniteBenchmark(bm_id=bm_id, seed=0)
    axes = [bench.axes[name] for name in PARAM_NAMES]
    shape = tuple(len(a) for a in axes)
    tensor = np.zeros(shape)
    for idx in itertools.product(*[range(s) for s in shape]):
        tensor[idx] = bench.lookup[tuple(axes[d][i] for d, i in enumerate(idx))]
    return tensor, axes


def additive_share(tensor: np.ndarray) -> float:
    """Fraction of variance explained by main effects (ANOVA, no interactions)."""
    centered = tensor - tensor.mean()
    main = np.zeros_like(tensor)
    for d in range(tensor.ndim):
        others = tuple(i for i in range(tensor.ndim) if i != d)
        effect = tensor.mean(axis=others) - tensor.mean()
        shape = [1] * tensor.ndim
        shape[d] = len(effect)
        main = main + effect.reshape(shape)
    return float(1 - ((centered - main) ** 2).sum() / (centered**2).sum())


def mode_spectra(tensor: np.ndarray) -> dict[str, dict]:
    """Per-mode cumulative singular-value energy of the centered tensor."""
    centered = tensor - tensor.mean()
    out = {}
    for d, name in enumerate(PARAM_NAMES):
        unfolded = np.moveaxis(centered, d, 0).reshape(tensor.shape[d], -1)
        sv = np.linalg.svd(unfolded, compute_uv=False)
        energy = np.cumsum(sv**2) / np.sum(sv**2)
        out[name] = {
            "cumulative_energy": energy.tolist(),
            "rank_95": int(np.searchsorted(energy, 0.95) + 1),
            "dim": int(tensor.shape[d]),
        }
    return out


def analyse() -> dict:
    results = {}
    for bm_id, label in TASKS.items():
        tensor, _ = reward_tensor(bm_id)
        results[label] = {
            "additive_share": additive_share(tensor),
            "modes": mode_spectra(tensor),
            "shape": list(tensor.shape),
        }
    return results


def report(results: dict) -> None:
    print(f"{'task':13s}{'additive':>10s}{'interaction':>13s}   per-mode rank for 95% energy")
    print("-" * 78)
    for label, data in results.items():
        add = data["additive_share"] * 100
        ranks = ", ".join(
            f"{n.replace('min_samples_', 'ms_')}:{m['rank_95']}/{m['dim']}"
            for n, m in data["modes"].items()
        )
        print(f"{label:13s}{add:9.1f}%{100 - add:12.1f}%   {ranks}")


def plot(results: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.3 * len(results), 3.4))
    if len(results) == 1:
        axes = [axes]
    for ax, (label, data) in zip(axes, results.items()):
        for name, mode in data["modes"].items():
            energy = [0.0] + list(mode["cumulative_energy"])
            ax.plot(range(len(energy)), np.array(energy) * 100, marker="o",
                    markersize=3.5, linewidth=1.4, label=name)
        ax.axhline(95, color="0.4", linestyle=":", linewidth=1)
        ax.set_title(f"{label}\nadditive: {data['additive_share']*100:.1f}%")
        ax.set_xlabel("Components retained")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 103)
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Cumulative variance (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"figure saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    results = analyse()
    with open(RESULT_DIR / "reward_structure.json", "w") as fh:
        json.dump(results, fh, indent=1)
    report(results)
    if args.plot:
        plot(results, RESULT_DIR / "reward_structure.pdf")


if __name__ == "__main__":
    main()
