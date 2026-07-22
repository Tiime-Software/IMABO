"""Delayed and censored reward experiment.

Does IMABO's delay-aware switching rule (`switch_strategy="delayed"`, see
`imabo/moss.py`'s `moss_anytime`) actually help when reward feedback arrives
late or never arrives at all, compared to a delay-oblivious IMABO exposed to
the same delayed environment, and to a no-delay skyline?

Delay/censoring is calibrated from real Gleipnir production feedback timing
(see `experiments/benchmarks/delayed/delay_model.py` -- only the fitted
log-normal + censoring-frequency parameters are kept, never the raw per-user
timing log). The reward surface is the same real RF tabular HPOBench
benchmark used by `rf_arm_distribution_experiment.py`, with the same
`bm_id`s, so this experiment's figure sits alongside the existing paper
results.
"""

import copy
import json
from enum import Enum
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.random_search import RandomSearch
from experiments.benchmarks.delayed.delay_model import DelayModel
from experiments.benchmarks.delayed.simulator import run_baseline, run_delayed
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from imabo import IMABO

RESULT_DIR = Path(__file__).parent.parent / "results" / "delayed_feedback"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5
PATIENCE_STEPS = 72
N_JOBS = 8


class Algorithm(Enum):
    IMABO_DELAYED = "IMABO-Delayed"
    IMABO_NAIVE = "IMABO-Naive"
    IMABO_NO_DELAY = "IMABO-NoDelay"
    RANDOM = "Random Search"


def algo_slug(algorithm: Algorithm) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames."""
    return algorithm.value.lower().replace(" ", "_").replace("-", "_")


def benchmark_tag(bm_id: int) -> str:
    return f"rf{bm_id}"


def build_optimizer(algorithm: Algorithm, search_space: dict, seed: int):
    """Factory dispatching on the algorithm enum (mirrors the same pattern in
    rf_arm_distribution_experiment.py). IMABO_NAIVE and IMABO_NO_DELAY share
    the exact same optimizer config (switch_strategy="beta") -- they only
    differ in which loop runs them (run_delayed vs run_baseline, see
    run_single_experiment)."""
    if algorithm == Algorithm.IMABO_DELAYED:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=BETA,
            switch_strategy="delayed",
        )
    elif algorithm in (Algorithm.IMABO_NAIVE, Algorithm.IMABO_NO_DELAY):
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=BETA,
            switch_strategy="beta",
        )
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=search_space, seed=seed)
    raise ValueError(f"Invalid algorithm: {algorithm}")


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    patience_steps: int = PATIENCE_STEPS,
    delay_model: DelayModel | None = None,
) -> dict:
    """Run one seed of one algorithm, returning a flat JSON-serializable dict
    (regrets trace, pending/arrival/censoring traces, best config found).

    `delay_model` is only consulted for the two algorithms run through
    `run_delayed` (IMABO_DELAYED, IMABO_NAIVE); defaults to the real
    Gleipnir-calibrated model. IMABO_NO_DELAY/RANDOM always run through
    `run_baseline`, which never sees a delay model at all.
    """
    search_space = bench.get_search_space()
    opt = build_optimizer(algorithm, search_space, seed)

    if algorithm in (Algorithm.IMABO_DELAYED, Algorithm.IMABO_NAIVE):
        result = run_delayed(
            opt,
            bench,
            n_iterations,
            seed=seed,
            delay_model=delay_model or DelayModel(),
            patience_steps=patience_steps,
        )
    else:
        result = run_baseline(opt, bench, n_iterations, seed=seed)

    best = result["best_config"]
    result["simple_regret"] = (
        bench.regret(best) if best is not None else bench.max_value
    )
    return result


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run."""
    stem = f"{benchmark_tag(bench.bm_id)}_{algo_slug(algorithm)}_{n_iterations}iters"

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
        result = run_single_experiment(local_bench, n_iterations, algorithm, seed=seed)
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
    print(f"\n{algorithm.value}: T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    n_iter = 10000
    # Same OpenML tasks as rf_arm_distribution_experiment.py, so this
    # experiment's figures sit alongside the existing ones.
    bm_ids = [146822, 31, 167120]
    algorithms = [
        Algorithm.IMABO_DELAYED,
        Algorithm.IMABO_NAIVE,
        Algorithm.IMABO_NO_DELAY,
        Algorithm.RANDOM,
    ]

    for bm_id in bm_ids:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        print(
            f"RF tabular finite benchmark (OpenML task {bm_id}): "
            f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}"
        )
        for algorithm in algorithms:
            run_experiment(bench, n_runs, base_seed, n_iter, algorithm)
