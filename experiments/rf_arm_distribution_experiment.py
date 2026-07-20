"""Per-iteration distribution of the arm each oracle would suggest"""

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from experiments.rf_tabular_bandit_experiment import (
    Algorithm,
    algo_slug,
    benchmark_tag,
    build_optimizer,
)
from imabo.memory import config_to_key
from imabo.optimizer import load_tabfm

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite_arm_distribution"
RESULT_DIR.mkdir(exist_ok=True)

N_SHADOW = 10
N_JOBS = 8


def _shadow_copy(opt: Any) -> Any:
    """Deep-copy an optimizer's mutable state for a disposable "what would it
    suggest right now" probe, without deep-copying (or corrupting) heavy
    read-only attributes shared across instances.

    IMABOTabFM's `_tabfm_model` is a single frozen pretrained model reused
    across every run/budget (see rf_tabular_bandit_experiment.run_experiment);
    deep-copying its weights on every one of thousands of iterations would be
    both unnecessary (it's never mutated) and far too slow. Pre-seeding the
    deepcopy memo with its id makes copy.deepcopy skip it and reuse the same
    reference in the copy instead.
    """
    memo = {}
    model = getattr(opt, "_tabfm_model", None)
    if model is not None:
        memo[id(model)] = model
    return copy.deepcopy(opt, memo)


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    n_shadow: int = N_SHADOW,
) -> dict:
    """Run one seed, recording at every iteration the mean/std of the true
    reward of `n_shadow` independent draws from the optimizer's suggest()
    distribution at that state (see _shadow_copy), alongside the same
    real-trajectory fields as rf_tabular_bandit_experiment.run_single_experiment.
    """
    opt = build_optimizer(algorithm, bench.get_search_space(), seed, tabfm_model)
    param_names = sorted(bench.get_search_space().keys())

    regrets = []
    simple_regret_trace = []
    shadow_reward_mean = []
    shadow_reward_std = []
    suggestion_counts: Counter = Counter()
    for _ in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        # `shadow.suggest()` with no `observe()` in between is safe for our
        # switch_strategy="beta" optimizers: moss_anytime only reads
        # nb_pending in "delayed" mode, so the pending-count buildup across
        # these n_shadow calls never reaches the scoring formula. It DOES
        # bump the shared step_counter each call (see Memory.pull_arm), so by
        # the last of the n_shadow draws the shadow's internal t is inflated
        # by up to n_shadow-1 -- negligible against t once past the first
        # ~100 iterations of a 5000-iteration budget, and discarded with the
        # copy either way.
        shadow = _shadow_copy(opt)
        shadow_rewards = [bench.mean_reward(shadow.suggest()) for _ in range(n_shadow)]
        shadow_reward_mean.append(float(np.mean(shadow_rewards)))
        shadow_reward_std.append(float(np.std(shadow_rewards)))

        x = opt.suggest()
        y = bench(x, noise=True)
        opt.observe(y)
        regrets.append(bench.regret(x))
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
        "shadow_reward_mean": shadow_reward_mean,
        "shadow_reward_std": shadow_reward_std,
        "best_config": best,
        "best_reward": best_reward,
        "best_config_suggestions": (
            suggestion_counts[best_key] if best_key is not None else 0
        ),
        "most_suggested_count": most_suggested_count,
        "is_best_most_suggested": best_key is not None
        and best_key == most_suggested_key,
    }


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    tabfm_model: Any = None,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run (mirrors rf_tabular_bandit_experiment.run_multiple_experiments)."""
    stem = (
        f"{benchmark_tag(bench.bm_id, True)}_{algo_slug(algorithm)}_{n_iterations}iters"
    )

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
    n_jobs: int = N_JOBS,
) -> None:
    tabfm_model = load_tabfm() if algorithm == Algorithm.IMOSS_TABFM else None
    if tabfm_model is not None:
        print("Loaded TabFM model (once, reused across all runs/budgets).")

    print(f"\n{algorithm.value}: T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        tabfm_model=tabfm_model,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    n_iter = 5000
    # Same three benchmarks as rf_tabular_bandit_experiment.py, spanning
    # reward-noise regimes: segment (clean), credit-g (noisy), numerai28.6 (hard).
    bm_ids = [146822, 31, 167120]
    # The IMOSS proposal-oracle family only -- this experiment is about
    # contrasting how each oracle's live suggestion distribution evolves,
    # not about the IMOSS-vs-UCB-AIR framework comparison.
    algorithms = [
        Algorithm.IMOSS,
        Algorithm.IMOSS_TPE,
        Algorithm.IMOSS_TABFM,
        Algorithm.UCB_AIR,
    ]

    for bm_id in bm_ids:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        print(
            f"RF tabular finite benchmark (OpenML task {bm_id}): "
            f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}"
        )
        for algorithm in algorithms:
            run_experiment(bench, n_runs, base_seed, n_iter, algorithm)
