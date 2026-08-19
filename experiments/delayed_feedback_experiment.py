"""Delayed and censored reward experiment.

Does IMABO's delay-aware switching rule (`IMOSS(delayed=True)`, see
`imabo/policies/imoss.py`'s `anytime_moss_index`) actually help when reward feedback arrives
late or never arrives at all, compared to a delay-oblivious IMABO exposed to
the same delayed environment, and to a no-delay skyline?

Benchmarks (set BENCHMARK below): the LCBench surrogate (YAHPO-Gym), a **mixed
continuous + finite** NN-HPO problem, and NAS-Bench-201, a **structured finite**
neural-architecture cell (6 edges x 5 ops). Both are structured spaces IMABO's
TPE oracle is built for -- distinct from the tabular RF tasks used elsewhere in
the paper. See `experiments/benchmarks/delayed/lcbench_bandit.py` /
`nasbench201_bandit.py` (one-time setup: `setup_lcbench.py` /
`setup_nasbench201.py`).
"""

import copy
import json
from enum import Enum
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.delayed.delay_model import RuntimeDelayModel
from experiments.benchmarks.delayed.simulator import (
    patience_for_quantile,
    run_baseline,
    run_delayed,
)
from imabo import IMABO, IMOSS, TPEOracle

# ---------------------------------------------------------------- benchmark
# Two reward surfaces for the delayed experiment, each with a runtime-driven
# delay (the delay IS the config's real training time):
#   "lcbench"     mixed continuous+finite NN HPO surrogate (default).
#   "nasbench201" structured finite NAS cell (6 edges x 5 ops), Dong & Yang 2020.
# Both are structured spaces where IMABO's TPE oracle generalizes across the
# structure -- the regime the delay mechanism is meant to show value in.
BENCHMARK = "lcbench"


# LCBench instances (tabular tasks).
LCBENCH_INSTANCES = ["167200", "168868", "189908"]  # higgs, APSFailure, Fashion-MNIST
# NAS-Bench-201 datasets (image classification).
NASBENCH201_INSTANCES = ["cifar100", "cifar10", "ImageNet16-120"]

_RESULT_SUBDIR = {
    "lcbench": "delayed_feedback_lcbench",
    "nasbench201": "delayed_feedback_nasbench201",
}
RESULT_DIR = Path(__file__).parent.parent / "results" / _RESULT_SUBDIR[BENCHMARK]
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Severity sweep results land in a separate, parallel directory tree so the
# main run's checkpoints are never touched by (or confused with) sweep points.
_SEVERITY_RESULT_SUBDIR = {
    "lcbench": "delayed_feedback_lcbench_severity",
    "nasbench201": "delayed_feedback_nasbench201_severity",
}
SEVERITY_RESULT_DIR = (
    Path(__file__).parent.parent / "results" / _SEVERITY_RESULT_SUBDIR[BENCHMARK]
)
SEVERITY_RESULT_DIR.mkdir(parents=True, exist_ok=True)

BETA = 0.5
# Patience window is set PER BENCHMARK as a shared quantile of each benchmark's
# own delay distribution (see simulator.patience_for_quantile). At quantile q, exactly (1 - q) of *observed* feedback is
# cut by the window in every benchmark, isolating the Bernoulli (never-arrives)
# censoring in cross-benchmark comparisons. PATIENCE_STEPS_FALLBACK is used only
# if a delay model produces no positive delays (e.g. feedback_freq == 0).
PATIENCE_QUANTILE = 0.95
PATIENCE_STEPS_FALLBACK = 72
# LCBench queries a shared ONNX surrogate session, and nats_bench holds a shared
# tabular API handle -- neither is guaranteed thread-safe under concurrent
# queries, so both run serial. Override after import if your yahpo/ORT or
# nats_bench build is confirmed thread-safe.
N_JOBS = 1


def make_benchmark(bench_id):
    """Construct the reward-surface benchmark for one task id."""
    if BENCHMARK == "lcbench":
        from experiments.benchmarks.delayed.lcbench_bandit import LCBenchMixedBenchmark

        return LCBenchMixedBenchmark(instance=str(bench_id))
    if BENCHMARK == "nasbench201":
        from experiments.benchmarks.delayed.nasbench201_bandit import (
            NASBench201Benchmark,
        )

        return NASBench201Benchmark(instance=str(bench_id))
    raise ValueError(
        f"Unknown BENCHMARK {BENCHMARK!r}; delayed experiment supports "
        f"'lcbench' and 'nasbench201'."
    )


DEFAULT_FEEDBACK_FREQ = (
    0.2  # 80% Bernoulli-censored, on top of patience-window censoring
)


def default_delay_model():
    """Delay = the config's real training time for both benchmarks
    (RuntimeDelayModel), so a slow config reports late and one exceeding the
    patience window is censored. Additionally, `feedback_freq=0.2` means only
    20% of pulls are ever eligible to arrive at all -- the other 80% are
    Bernoulli-censored upfront (a crashed/never-submitted job, independent of
    runtime), stacking with the patience-window censoring of the pulls that
    are eligible."""
    return RuntimeDelayModel(feedback_freq=DEFAULT_FEEDBACK_FREQ)


def patience_for_bench(bench, delay_model=None, q: float = PATIENCE_QUANTILE) -> int:
    """Per-benchmark patience window: the q-th percentile of THIS benchmark's
    own delay distribution (see simulator.patience_for_quantile). Computed once
    per benchmark from the BASE (delay_scale=1.0) delay model and held fixed
    across a run/sweep, so a larger delay_scale pushes more pulls past a fixed
    window instead of moving the window with the delays. Falls back to a fixed
    step count if no positive delays can be sampled (e.g. feedback_freq == 0)."""
    try:
        return patience_for_quantile(bench, delay_model or default_delay_model(), q=q)
    except ValueError:
        return PATIENCE_STEPS_FALLBACK


class Algorithm(Enum):
    IMABO_DELAYED = "IMABO-Delayed"
    IMABO_NAIVE = "IMABO-Naive"
    IMABO_NO_DELAY = "IMABO-NoDelay"
    UCB_AIR = "UCB-AIR"


def algo_slug(algorithm: Algorithm) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames."""
    return algorithm.value.lower().replace(" ", "_").replace("-", "_")


def benchmark_tag(bm_id) -> str:
    """Filename prefix, distinct per benchmark family so result files never
    collide (`lc126026_...` vs `nb201cifar100_...`)."""
    prefix = {"lcbench": "lc", "nasbench201": "nb201"}[BENCHMARK]
    return f"{prefix}{bm_id}"


def build_optimizer(algorithm: Algorithm, search_space: dict, seed: int):
    """Factory dispatching on the algorithm enum (mirrors the same pattern in
    rf_arm_distribution_experiment.py).

    IMABO_NAIVE and IMABO_NO_DELAY deliberately share the exact same optimizer
    config (`IMOSS(delayed=False)`); this is NOT a redundant double-run. They
    differ in the *environment* they are run in, which is what each is meant to
    measure (see run_single_experiment):
        - IMABO_NAIVE   -> run_delayed:  the delay-blind optimizer exposed to
                          delayed/censored feedback -> the cost of ignoring delay.
        - IMABO_NO_DELAY-> run_baseline: the same optimizer with instant feedback
                          -> the no-delay skyline (best achievable, upper bound).
    Same code, different simulator, different regret trace -- so both are needed;
    they cannot be collapsed into one run. IMABO_DELAYED (IMOSS(delayed=True)) is
    the delay-aware rule whose whole point is to recover the NO_DELAY skyline while
    running in the same delayed environment as NAIVE.
    """
    if algorithm == Algorithm.IMABO_DELAYED:
        return IMABO(
            search_space,
            IMOSS(beta=BETA, delayed=True),
            TPEOracle(multivariate=True),
            seed=seed,
        )
    elif algorithm in (Algorithm.IMABO_NAIVE, Algorithm.IMABO_NO_DELAY):
        return IMABO(
            search_space,
            IMOSS(beta=BETA),
            TPEOracle(multivariate=True),
            seed=seed,
        )
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(search_space=search_space, seed=seed, beta=BETA)


def run_single_experiment(
    bench,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    patience_steps: int = PATIENCE_STEPS_FALLBACK,
    delay_model=None,
) -> dict:
    """Run one seed of one algorithm, returning a flat JSON-serializable dict
    (regrets trace, pending/arrival/censoring traces, best config found).

    `delay_model` is only consulted for the two algorithms run through
    `run_delayed` (IMABO_DELAYED, IMABO_NAIVE); defaults to the model matching
    the active benchmark (runtime-driven for LCBench, log-normal for RF).
    IMABO_NO_DELAY/UCB_AIR always run through `run_baseline`, which never sees a
    delay model at all.
    """
    search_space = bench.get_search_space()
    opt = build_optimizer(algorithm, search_space, seed)

    if algorithm in (Algorithm.IMABO_DELAYED, Algorithm.IMABO_NAIVE):
        result = run_delayed(
            opt,
            bench,
            n_iterations,
            seed=seed,
            delay_model=delay_model or default_delay_model(),
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
    bench,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
    patience_steps: int | None = None,
    delay_model=None,
    stem_suffix: str = "",
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run. `patience_steps` is the per-benchmark window (computed once by the
    caller); if None it is derived here from this benchmark's delay quantile.

    `delay_model` overrides the delay model passed to `run_single_experiment`
    for the two algorithms that consult one (IMABO_DELAYED, IMABO_NAIVE);
    ignored for IMABO_NO_DELAY/UCB_AIR, which never see one. `stem_suffix`
    tags the checkpoint filename for a non-default condition (e.g. "_ff100"
    for a feedback_freq=1.0 run) so it can't collide with the default
    condition's files, which keep the original untagged stem.
    """
    if patience_steps is None:
        patience_steps = patience_for_bench(bench)
    stem = (
        f"{benchmark_tag(bench.bm_id)}_{algo_slug(algorithm)}_{n_iterations}iters"
        f"{stem_suffix}"
    )

    all_results: list[dict | None] = [None] * n_runs
    pending = []
    for i in tqdm(range(n_runs), desc=f"Running {n_runs} experiments"):
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
            patience_steps=patience_steps,
            delay_model=delay_model,
        )
        result["patience_steps"] = patience_steps  # provenance for the run
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
    patience_steps: int | None = None,
    delay_model=None,
    stem_suffix: str = "",
) -> None:
    print(f"\n{algorithm.value}: T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        n_jobs=n_jobs,
        patience_steps=patience_steps,
        delay_model=delay_model,
        stem_suffix=stem_suffix,
    )


# ============================================================ severity sweep
# How does IMABO's delay-aware correction hold up as delay/censoring get more
# or less severe than the calibrated default above? The main run above uses a
# single, fixed severity level; this sweep varies severity along two
# independent axes and looks at the FINAL regret (a shorter horizon than the
# main run, since only the endpoint matters, not the full trajectory shape):
#   - delay severity: multiplies the expected delay, with the Bernoulli
#     censoring rate held at the same calibrated `feedback_freq=0.2` as the
#     main experiment (`DEFAULT_FEEDBACK_FREQ`), so this sweep matches the
#     main run's censoring regime and varies delay length only.
#   - censoring severity: overrides the feedback frequency directly, delay
#     distribution fixed at the real calibrated shape.
# Only IMABO_DELAYED and IMABO_NAIVE actually respond to either sweep -- they
# are the two algorithms run through `run_delayed` with the swept delay model.
# IMABO_NO_DELAY/UCB_AIR run through `run_baseline`, which never consults a
# delay model, so their regret is severity-invariant by construction; they are
# run once (not re-swept) as flat reference lines -- see
# `experiments/utils/plots/delayed_feedback_plot.py`.


def make_severity_delay_model(delay_scale: float, feedback_freq: float | None):
    """One severity-point delay model. Both benchmarks use the runtime-driven
    RuntimeDelayModel: `delay_scale` multiplies every config's runtime-derived
    delay (delay-length axis) and `feedback_freq` sets the Bernoulli arrival
    probability (censoring axis)."""
    return RuntimeDelayModel.at_severity(
        delay_scale=delay_scale, feedback_freq=feedback_freq
    )


SEVERITY_N_ITERATIONS = 10000

# Multiplies expected delay; 1.0 is the real calibrated severity.
DELAY_SEVERITIES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CENSORING_SEVERITIES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

SWEPT_ALGORITHMS = [Algorithm.IMABO_DELAYED, Algorithm.IMABO_NAIVE]
REFERENCE_ALGORITHMS = [Algorithm.IMABO_NO_DELAY, Algorithm.UCB_AIR]


def _sweep_stem(bm_id, algorithm: Algorithm, sweep_name: str, severity: float) -> str:
    return (
        f"{benchmark_tag(bm_id)}_{algo_slug(algorithm)}_{sweep_name}_{severity}"
        f"_{SEVERITY_N_ITERATIONS}iters"
    )


def _reference_stem(bm_id, algorithm: Algorithm) -> str:
    return (
        f"{benchmark_tag(bm_id)}_{algo_slug(algorithm)}_reference_"
        f"{SEVERITY_N_ITERATIONS}iters"
    )


def run_severity_sweep(
    bench,
    sweep_name: str,
    severities: list[float],
    delay_model_for,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
    patience_steps: int = PATIENCE_STEPS_FALLBACK,
    patience_for=None,
) -> None:
    """Run SWEPT_ALGORITHMS at every value of `severities`, checkpointed per
    (algorithm, severity, run), into SEVERITY_RESULT_DIR. `delay_model_for(severity)
    -> DelayModel` builds the model for one sweep point."""
    for severity in severities:
        delay_model = delay_model_for(severity)
        point_patience = patience_for(severity) if patience_for else patience_steps
        for algorithm in SWEPT_ALGORITHMS:
            stem = _sweep_stem(bench.bm_id, algorithm, sweep_name, severity)
            pending = [
                i
                for i in range(n_runs)
                if not (SEVERITY_RESULT_DIR / f"{stem}_run{i}.json").exists()
            ]
            if not pending:
                tqdm.write(f"--- {stem}: all {n_runs} runs already complete ---")
                continue

            def _one_run(
                i,
                algorithm=algorithm,
                delay_model=delay_model,
                stem=stem,
                point_patience=point_patience,
            ) -> dict:
                seed = base_seed + i
                local_bench = copy.copy(bench)
                local_bench.reset_noise(seed)
                opt = build_optimizer(algorithm, local_bench.get_search_space(), seed)
                result = run_delayed(
                    opt,
                    local_bench,
                    SEVERITY_N_ITERATIONS,
                    seed=seed,
                    delay_model=delay_model,
                    patience_steps=point_patience,
                )
                result["patience_steps"] = point_patience  # provenance
                with open(SEVERITY_RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
                    json.dump(result, f)
                return result

            print(f"{stem}: running {len(pending)}/{n_runs} pending runs...")
            Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
                delayed(_one_run)(i) for i in pending
            )


def run_reference_algorithms(
    bench,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
) -> None:
    """Run REFERENCE_ALGORITHMS once each (severity-invariant, see module
    docstring), checkpointed under a fixed 'reference' tag reused by every
    sweep's plot as a flat line, into SEVERITY_RESULT_DIR."""
    for algorithm in REFERENCE_ALGORITHMS:
        stem = _reference_stem(bench.bm_id, algorithm)
        pending = [
            i
            for i in range(n_runs)
            if not (SEVERITY_RESULT_DIR / f"{stem}_run{i}.json").exists()
        ]
        if not pending:
            tqdm.write(f"--- {stem}: all {n_runs} runs already complete ---")
            continue

        def _one_run(i, algorithm=algorithm, stem=stem) -> dict:
            seed = base_seed + i
            local_bench = copy.copy(bench)
            local_bench.reset_noise(seed)
            opt = build_optimizer(algorithm, local_bench.get_search_space(), seed)
            result = run_baseline(opt, local_bench, SEVERITY_N_ITERATIONS, seed=seed)
            with open(SEVERITY_RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
                json.dump(result, f)
            return result

        print(f"{stem}: running {len(pending)}/{n_runs} pending runs...")
        Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_one_run)(i) for i in pending
        )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    n_iter = 10000
    bench_ids = {
        "lcbench": LCBENCH_INSTANCES,
        "nasbench201": NASBENCH201_INSTANCES,
    }[BENCHMARK]
    algorithms = [
        Algorithm.IMABO_DELAYED,
        Algorithm.IMABO_NAIVE,
        Algorithm.IMABO_NO_DELAY,
        Algorithm.UCB_AIR,
    ]

    for bench_id in bench_ids:
        bench = make_benchmark(bench_id)
        n_arms = getattr(bench, "n_arms", "inf (mixed continuous+finite)")

        # --- main experiment: single, calibrated severity level -------------
        # One patience window per benchmark: the PATIENCE_QUANTILE-th percentile
        # of this benchmark's own delay distribution, held fixed across all four
        # algorithms and all seeds for this benchmark.
        patience = patience_for_bench(bench)
        print(
            f"[{BENCHMARK}] task {bench_id}: {n_arms} arms, "
            f"best val={bench.max_value:.4f}, "
            f"patience={patience} steps (q={PATIENCE_QUANTILE})"
        )
        for algorithm in algorithms:
            run_experiment(
                bench, n_runs, base_seed, n_iter, algorithm, patience_steps=patience
            )

        # --- severity sweep: delay length x censoring rate ------------------
        print(f"\n=== [{BENCHMARK}] severity sweep: task {bench_id} ===")
        try:
            severity_base_model = make_severity_delay_model(
                delay_scale=1.0, feedback_freq=1.0
            )
            severity_patience = patience_for_quantile(
                bench, severity_base_model, q=PATIENCE_QUANTILE
            )
        except ValueError:
            severity_patience = PATIENCE_STEPS_FALLBACK
        print(
            f"    severity patience={severity_patience} steps "
            f"(q={PATIENCE_QUANTILE}, from base delay model)"
        )

        run_reference_algorithms(bench, n_runs=n_runs, base_seed=base_seed)

        print("--- Delay severity sweep ---")
        # feedback_freq held at the main experiment's calibrated
        # DEFAULT_FEEDBACK_FREQ (0.2) so this sweep varies delay length only,
        # under the same censoring regime as the main run. Censoring's own
        # effect is the sweep below.
        run_severity_sweep(
            bench,
            "delay",
            DELAY_SEVERITIES,
            delay_model_for=lambda s: make_severity_delay_model(
                delay_scale=s, feedback_freq=DEFAULT_FEEDBACK_FREQ
            ),
            n_runs=n_runs,
            base_seed=base_seed,
            patience_steps=severity_patience,
        )

        print("--- Censoring severity sweep ---")
        # CENSORING_SEVERITIES is the feedback frequency = 1 - censoring rate,
        # realized as the Bernoulli arrival probability at a fixed patience.
        run_severity_sweep(
            bench,
            "censor",
            CENSORING_SEVERITIES,
            delay_model_for=lambda s: make_severity_delay_model(
                delay_scale=1.0, feedback_freq=s
            ),
            n_runs=n_runs,
            base_seed=base_seed,
            patience_steps=severity_patience,
        )
