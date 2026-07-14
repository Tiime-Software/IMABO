"""Hyperparameter optimization as a finite-armed Bernoulli bandit.

Each arm is a RandomForest hyperparameter configuration from a real OpenML
tabular benchmark; pulling an arm draws a Bernoulli sample whose success
probability is that configuration's validation accuracy. Because the arm set
is finite and its accuracies are precomputed, both the optimum and the regret
of every pull are known exactly, with no model re-fitting required.

Compares the IMOSS bandit framework -- with a TPE, uniform, or TabFM proposal
oracle -- against classic infinite-armed bandit baselines, across benchmarks
spanning easy to hard reward-noise regimes. Each run is checkpointed to its
own file, so re-running only completes missing seeds or algorithms.

Usage (from repo root):
    python -m experiments.rf_tabular_bandit_experiment
"""

import copy
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from experiments.utils.stats import (
    calculate_statistics,
    save_iterations_to_csv,
    save_results_to_csv,
)
from imabo import IMABO
from imabo.memory import config_to_key
from imabo.optimizer import IMABOTabFM, load_tabfm

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5
N_JOBS = 8


class Algorithm(Enum):
    IMOSS_TPE = "IMOSS-TPE"
    IMOSS = "IMOSS"
    RANDOM = "Random Search"
    IMOSS_TABFM = "IMOSS-TabFM"
    UCB_AIR = "UCB-AIR"


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
        return IMABOTabFM(
            search_space=search_space,
            seed=seed,
            tabfm_model=model,
            beta=BETA,
        )
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(
            search_space=search_space,
            seed=seed,
        )


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    noise: bool = True,
) -> dict:
    """Run one seed of the experiment with a single algorithm.

    noise=True (default): Bernoulli(f(x)) reward.
    noise=False: the reward IS f(x) directly (no sampling) -- an ablation to
    check how much of the run-to-run spread/outliers comes from reward noise
    rather than the search-space/optimizer mechanics themselves.
    """
    opt = build_optimizer(algorithm, bench.get_search_space(), seed, tabfm_model)
    param_names = sorted(bench.get_search_space().keys())

    regrets = []
    # Anytime simple regret: at each step, the true regret of the config the
    # optimizer would RETURN as best if stopped now (its actual selection
    # strategy applied to the current state) -- lets us plot convergence of the
    # returned answer over the budget, not just the final scalar simple_regret.
    simple_regret_trace = []
    suggestion_counts: Counter = Counter()
    for _ in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        x = opt.suggest()
        y = bench(x, noise=noise)
        opt.observe(y)
        regrets.append(bench.regret(x))  # noiseless regret
        suggestion_counts[config_to_key(x, param_names)] += 1
        incumbent = opt.best_config
        simple_regret_trace.append(
            bench.regret(incumbent) if incumbent is not None else bench.max_value
        )

    best = opt.best_config
    simple_regret = bench.regret(best) if best is not None else bench.max_value
    best_reward = bench.mean_reward(best) if best is not None else None

    best_key = config_to_key(best, param_names) if best is not None else None
    most_suggested_key, most_suggested_count = (
        suggestion_counts.most_common(1)[0] if suggestion_counts else (None, 0)
    )
    return {
        "regrets": regrets,
        "simple_regret_trace": simple_regret_trace,
        "simple_regrets": simple_regret,
        "best_config": best,
        "best_reward": best_reward,
        "best_config_suggestions": (
            suggestion_counts[best_key] if best_key is not None else 0
        ),
        "most_suggested_count": most_suggested_count,
        "is_best_most_suggested": best_key is not None
        and best_key == most_suggested_key,
    }


def benchmark_tag(bm_id: int, noise: bool) -> str:
    """Filename prefix -- keeps different benchmarks (bm_id) and the noiseless
    ablation's files from ever colliding with (or overwriting) each other."""
    return f"rf{bm_id}" if noise else f"rf{bm_id}noiseless"


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 20,
    base_seed: int = 42,
    tabfm_model: Any = None,
    noise: bool = True,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm.

    Each run is checkpointed to its own JSON file -- a run that's already on
    disk is loaded instead of re-executed, so re-running this (e.g. after
    adding more runs or budgets) never re-does completed work. Runs still
    missing a checkpoint are executed concurrently via joblib (threading
    backend -- each run's compute is numpy/sklearn-heavy and releases the
    GIL). Each thread gets its own shallow copy of `bench` re-seeded via
    `reset_noise`, since `bench.rng` is mutable and shared across threads
    otherwise -- both a data race and a reproducibility bug (concurrent
    runs would consume from the same noise stream instead of their own
    per-seed one).
    """
    stem = f"{benchmark_tag(bench.bm_id, noise)}_{algo_slug(algorithm)}_{n_iterations}iters"

    all_results: list[dict | None] = [None] * n_runs
    pending = []
    for i in range(n_runs):
        run_path = RESULT_DIR / f"{stem}_run{i}.json"
        if run_path.exists():
            with open(run_path) as f:
                all_results[i] = json.load(f)
            tqdm.write(f"--- {stem}_run{i} already complete, skipping ---")
        else:
            pending.append(i)

    def _one_run(i: int) -> dict:
        seed = base_seed + i
        local_bench = copy.copy(bench)
        local_bench.reset_noise(seed)
        result = run_single_experiment(
            local_bench,
            n_iterations,
            algorithm,
            seed=seed,
            tabfm_model=tabfm_model,
            noise=noise,
        )
        with open(RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
            json.dump(result, f)
        return result

    if pending:
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_one_run)(i) for i in pending
        )
        for i, result in zip(pending, results):
            all_results[i] = result

    return all_results


def run_experiment(
    bench,
    n_runs,
    base_seed,
    n_iter,
    algorithm: Algorithm,
    noise: bool = True,
    n_jobs: int = N_JOBS,
):
    dim = len(bench.get_search_space())
    tag = benchmark_tag(bench.bm_id, noise)

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
        n_jobs=n_jobs,
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
    # OpenML task_ids to run, spanning reward-noise regimes (all built via
    # experiments.benchmarks.build_rf_tabular_grid): 146822 segment (clean/well-separated),
    # 31 credit-g (mid-range, noisy Bernoulli), 167120 numerai28.6 (near-random, hard).
    bm_ids = [146822, 31, 167120]
    algorithms = [
        Algorithm.IMOSS_TPE,
        Algorithm.IMOSS,
        Algorithm.IMOSS_TABFM,
        Algorithm.UCB_AIR,
    ]

    for bm_id in bm_ids:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        print("Search space:")
        for name, values in bench.get_search_space().items():
            print(f"  {name}: {values['choices']}")
        print(
            f"RF tabular finite benchmark (OpenML task {bm_id}): "
            f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}, "
            f"best_config={bench.best_config}"
        )
        for algorithm in algorithms:
            run_experiment(bench, n_runs, base_seed, n_iter, algorithm)
