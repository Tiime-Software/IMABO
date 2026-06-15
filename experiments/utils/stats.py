"""Statistical helpers shared across experiment scripts."""

import csv
from pathlib import Path

import numpy as np


def calculate_statistics(all_results: list[dict]) -> dict:
    """Aggregate per-run regret data into mean/std statistics.

    Args:
        all_results: List of dicts, one per run.  Each dict maps
            algorithm_name -> {"regrets": list[float], "simple_regrets": float}.

    Returns:
        Dict mapping algorithm_name -> stats dict with keys:
            "regrets": {"mean", "std", "sum_regrets"},
            "simple_regrets": {"mean", "std", "simple_regrets"},
            "cov": covariance matrix of (sum_regret, simple_regret).
    """
    if not all_results:
        return {}

    algorithms = list(all_results[0].keys())
    statistics: dict = {}

    for algorithm in algorithms:
        all_regrets = np.asarray(
            [result[algorithm]["regrets"] for result in all_results]
        )
        all_simple_regrets = np.array(
            [result[algorithm]["simple_regrets"] for result in all_results]
        )

        sum_regrets = np.sum(all_regrets, axis=1)
        cov_regrets = np.cov(sum_regrets, all_simple_regrets)

        statistics[algorithm] = {
            "cov": cov_regrets,
            "regrets": {
                "mean": np.mean(all_regrets, axis=0),
                "std": np.std(all_regrets, axis=0),
                "sum_regrets": sum_regrets / all_regrets.shape[1],
            },
            "simple_regrets": {
                "mean": float(np.mean(all_simple_regrets)),
                "std": float(np.std(all_simple_regrets)),
                "simple_regrets": all_simple_regrets,
            },
        }
    return statistics


def save_results_to_csv(
    results_dict: dict,
    filename: str,
    exp_type: str = "toy",
    result_dir: Path | None = None,
) -> None:
    """Save per-algorithm summary statistics to a CSV file."""
    if result_dir is None:
        result_dir = Path("results")
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    summary_rows = []
    for key, stats in results_dict.items():
        parts = key.split("_")
        func_name = parts[0]
        dim = int(parts[1].replace("D", ""))
        n_iter = int(parts[2])

        for algorithm, data in stats.items():
            summary_rows.append(
                {
                    "function": func_name,
                    "dimension": dim,
                    "n_iterations": n_iter,
                    "algorithm": algorithm,
                    "simple_regret_mean": data["simple_regrets"]["mean"],
                    "simple_regret_std": data["simple_regrets"]["std"],
                    "total_regret_mean": float(data["regrets"]["sum_regrets"].mean()),
                    "total_regret_std": float(data["regrets"]["sum_regrets"].std()),
                }
            )

    path = result_dir / f"{filename}_{exp_type}_summary.csv"
    with open(path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary saved to {path}")


def save_iterations_to_csv(
    results_dict: dict,
    filename: str,
    exp_type: str = "toy",
    result_dir: Path | None = None,
) -> Path:
    """Save per-iteration regret statistics to a CSV file."""
    if result_dir is None:
        result_dir = Path("results")
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    rows = []
    for key, stats in results_dict.items():
        parts = key.split("_")
        func_name = parts[0]
        dim = int(parts[1].replace("D", ""))
        n_iter = int(parts[2])

        for algorithm, data in stats.items():
            regrets_mean = data["regrets"]["mean"]
            regrets_std = data["regrets"]["std"]
            for i, (mean, std) in enumerate(zip(regrets_mean, regrets_std)):
                rows.append(
                    {
                        "function": func_name,
                        "dimension": dim,
                        "n_iterations": n_iter,
                        "algorithm": algorithm,
                        "iteration": i + 1,
                        "regret_mean": mean,
                        "regret_std": std,
                    }
                )

    path = result_dir / f"{filename}_{exp_type}_iterations.csv"
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Iterations saved to {path}")
    return path
