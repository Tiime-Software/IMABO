"""Finite-armed HPO experiment on a real RF tabular Bernoulli bandit.

Ground truth: HPOBench's precomputed RandomForest accuracy grid on OpenML
task 9952 (phoneme), coarsened to a finite categorical search space and
converted to Bernoulli(accuracy) rewards -- see
experiments/benchmarks/tabular_finite.py for how the grid was built.

Since the arm set is finite and known, optimum and regret are exact (no
re-fitting): each optimizer runs directly against the lookup table.

Like experiments/hotpotqa_experiment.py, this runs ONE algorithm per
invocation and saves that algorithm's results to its own file (per-run JSON
checkpoints + a per-algorithm summary/iterations CSV). That way re-running or
adding an algorithm (e.g. the slow TabFM one) never requires re-running the
others -- just change `algorithm` in main() and re-invoke.

Usage (from repo root):
    python -m experiments.hpo_finite_experiment
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from tqdm import tqdm

from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import MOSSAIR, UCBAIR
from experiments.benchmarks.tabular_finite import RFTabularFiniteBenchmark
from experiments.utils.stats import (
    calculate_statistics,
    save_iterations_to_csv,
    save_results_to_csv,
)
from imabo import IMABO
from imabo.tabfm_optimizer import TabFMIMABO, load_tabfm

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5


class Algorithm(Enum):
    IMOSS_TPE = "IMOSS-TPE"
    IMOSS = "IMOSS"
    RANDOM = "Random Search"
    IMOSS_TABFM = "IMOSS-TabFM"
    UCB_AIR = "UCB-AIR"
    MOSS_AIR = "MOSS-AIR"


def algo_slug(algorithm: Algorithm) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames."""
    return algorithm.value.lower().replace(" ", "_").replace("-", "_")


def build_optimizer(
    algorithm: Algorithm,
    search_space: dict[str, Any],
    seed: int,
    tabfm_model: Any = None,
):
    if algorithm == Algorithm.IMOSS_TPE:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=BETA,
            tpe_split_bound="lcb",
        )
    elif algorithm == Algorithm.IMOSS:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            use_tpe=False,
            beta=BETA,
        )
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.IMOSS_TABFM:
        model = tabfm_model if tabfm_model is not None else load_tabfm()
        return TabFMIMABO(
            search_space=search_space, seed=seed, tabfm_model=model, beta=BETA
        )
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.MOSS_AIR:
        return MOSSAIR(search_space=search_space, seed=seed)


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    noise: bool = True,
) -> dict:
    """Run one seed of the experiment with a single algorithm.

    noise=True (default): Bernoulli(f(x)) reward, as originally designed.
    noise=False: the reward IS f(x) directly (no sampling) -- an ablation to
    check how much of the run-to-run spread/outliers comes from reward noise
    rather than the search-space/optimizer mechanics themselves.
    """
    opt = build_optimizer(algorithm, bench.get_search_space(), seed, tabfm_model)

    regrets = []
    for _ in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        x = opt.suggest()
        y = bench(x, noise=noise)
        opt.observe(y)
        regrets.append(bench.regret(x))  # noiseless regret

    best = opt.best_config
    simple_regret = bench.regret(best) if best is not None else bench.max_value
    best_reward = bench.mean_reward(best) if best is not None else None
    return {
        "regrets": regrets,
        "simple_regrets": simple_regret,
        "best_config": best,
        "best_reward": best_reward,
    }


def benchmark_tag(noise: bool) -> str:
    """Filename prefix -- keeps the noiseless ablation's files from ever
    colliding with (or overwriting) the normal Bernoulli-reward results."""
    return "rf9952" if noise else "rf9952noiseless"


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 20,
    base_seed: int = 42,
    tabfm_model: Any = None,
    noise: bool = True,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm.

    Each run is checkpointed to its own JSON file -- a run that's already on
    disk is loaded instead of re-executed, so re-running this (e.g. after
    adding more runs or budgets) never re-does completed work.
    """
    stem = f"{benchmark_tag(noise)}_{algo_slug(algorithm)}_{n_iterations}iters"
    all_results = []
    for i in tqdm(range(n_runs), desc=f"  T={n_iterations} runs", leave=True):
        seed = base_seed + i
        run_path = RESULT_DIR / f"{stem}_run{i}.json"
        if run_path.exists():
            with open(run_path) as f:
                all_results.append(json.load(f))
            tqdm.write(f"--- {stem}_run{i} already complete, skipping ---")
            continue
        result = run_single_experiment(
            bench,
            n_iterations,
            algorithm,
            seed=seed,
            tabfm_model=tabfm_model,
            noise=noise,
        )
        all_results.append(result)
        with open(run_path, "w") as f:
            json.dump(result, f)
    return all_results


def run_experiment(
    bench, n_runs, base_seed, n_iter, algorithm: Algorithm, noise: bool = True
):
    dim = len(bench.get_search_space())
    tag = benchmark_tag(noise)

    tabfm_model = load_tabfm() if algorithm == Algorithm.IMOSS_TABFM else None
    if tabfm_model is not None:
        print("Loaded TabFM model (once, reused across all runs/budgets).")

    results_dict = {}
    label = f"{algorithm.value}{'' if noise else ' (noiseless)'}"
    print(f"\n{label}: T={n_iter}, {n_runs} runs...")
    all_results = run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        tabfm_model=tabfm_model,
        noise=noise,
    )
    key = f"{tag}_{dim}D_{n_iter}"
    results_dict[key] = calculate_statistics(
        [{algorithm.value: r} for r in all_results]
    )

    filename = f"{tag}_{algo_slug(algorithm)}"
    save_results_to_csv(
        results_dict, filename, exp_type="hpo_finite", result_dir=RESULT_DIR
    )
    save_iterations_to_csv(
        results_dict, filename, exp_type="hpo_finite", result_dir=RESULT_DIR
    )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    n_iter = 5000
    bench = RFTabularFiniteBenchmark()
    dim = len(bench.get_search_space())
    print("Search space:")
    for name, values in bench.get_search_space().items():
        print(f"  {name}: {values['choices']}")
    print(
        f"RF tabular finite benchmark (OpenML task 9952, phoneme): "
        f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}, "
        f"best_config={bench.best_config}"
    )
    algorithms = [
        Algorithm.IMOSS_TPE,
        Algorithm.IMOSS,
        Algorithm.IMOSS_TABFM,
        Algorithm.UCB_AIR,
        Algorithm.MOSS_AIR,
    ]
    # Bernoulli-reward runs (noise=True) are already cached on disk from
    # before and get skipped; noise=False is the ablation checking how much
    # of the run-to-run spread/outliers comes from reward noise itself.
    for noise in (True, False):
        for algorithm in algorithms:
            run_experiment(bench, n_runs, base_seed, n_iter, algorithm, noise=noise)
