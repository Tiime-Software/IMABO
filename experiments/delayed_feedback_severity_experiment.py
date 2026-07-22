"""Sensitivity sweep: how does IMABO's delay-aware correction hold up as
delay/censoring get more or less severe than what we actually observed in
production?

`delayed_feedback_experiment.py` runs at a single, real severity level (the
Gleipnir-calibrated delay/censoring model). The standard way the delayed-
bandit literature shows *how much the delay-aware correction actually buys
you* is instead a sweep: final regret at the horizon as a function of a
delay-severity knob, one curve per algorithm -- e.g. Vernade et al., ICML
2020, "Linear Bandits with Stochastic Delayed Feedback" (regret-vs-time
panels at several expected-delay levels); Howson et al., 2023, "Delayed
Feedback in Generalised Linear Bandits Revisited" (Figs 3-4: final-round
regret directly as a function of expected delay). This module runs that
sweep, along two independent axes (our setting has two nuisance parameters,
not one):

  - delay severity: multiplies the expected delay, with censoring turned
    off entirely (`feedback_freq=1.0`) so this sweep isolates pure
    delay-length effects. Holding censoring at the real calibrated rate
    here instead would keep a constant ~59% of every pull permanently
    censored regardless of delay_scale (censored pulls always expire at
    `patience_steps+1`, independent of the sampled delay), which dominates
    how much pending backlog accumulates and washes out any dependence on
    delay length itself.
  - censoring severity: overrides the feedback frequency directly, delay
    distribution fixed at the real calibrated shape
    (`DelayModel.at_severity(feedback_freq=...)`).

Only IMABO_DELAYED and IMABO_NAIVE actually respond to either sweep -- they
are the two algorithms run through `run_delayed` with the swept delay model.
IMABO_NO_DELAY/RANDOM run through `run_baseline`, which never consults a
delay model, so their regret is severity-invariant by construction; they are
run once (not re-swept) as flat reference lines -- see
`experiments/utils/plots/delayed_feedback_plot.py`.
"""

import copy
import json
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.benchmarks.delayed.delay_model import DelayModel
from experiments.benchmarks.delayed.simulator import run_baseline, run_delayed
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from experiments.delayed_feedback_experiment import (
    Algorithm,
    algo_slug,
    benchmark_tag,
    build_optimizer,
)

RESULT_DIR = Path(__file__).parent.parent / "results" / "delayed_feedback_severity"
RESULT_DIR.mkdir(exist_ok=True)

# Shorter than delayed_feedback_experiment.py's n_iter=10000: only the final
# cumulative regret at this horizon is used, so the full trajectory shape
# doesn't need as many iterations, and this runs at 2 sweeps x 6 points x 2
# algorithms x n_runs -- keeping it short keeps the whole sweep tractable.
N_ITERATIONS = 2000
PATIENCE_STEPS = 72
N_JOBS = 8

# Multiplies expected delay; 1.0 is the real Gleipnir-calibrated severity.
DELAY_SEVERITIES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# Overrides feedback_freq directly; 0.407 is the real calibrated value.
CENSORING_SEVERITIES = [0.1, 0.2, 0.407, 0.6, 0.8, 1.0]

SWEPT_ALGORITHMS = [Algorithm.IMABO_DELAYED, Algorithm.IMABO_NAIVE]
REFERENCE_ALGORITHMS = [Algorithm.IMABO_NO_DELAY, Algorithm.RANDOM]


def _sweep_stem(bm_id: int, algorithm: Algorithm, sweep_name: str, severity: float) -> str:
    return (
        f"{benchmark_tag(bm_id)}_{algo_slug(algorithm)}_{sweep_name}_{severity}"
        f"_{N_ITERATIONS}iters"
    )


def _reference_stem(bm_id: int, algorithm: Algorithm) -> str:
    return f"{benchmark_tag(bm_id)}_{algo_slug(algorithm)}_reference_{N_ITERATIONS}iters"


def run_severity_sweep(
    bench: RFTabularFiniteBenchmark,
    sweep_name: str,
    severities: list[float],
    delay_model_for,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
) -> None:
    """Run SWEPT_ALGORITHMS at every value of `severities`, checkpointed per
    (algorithm, severity, run). `delay_model_for(severity) -> DelayModel`
    builds the model for one sweep point."""
    for severity in severities:
        delay_model = delay_model_for(severity)
        for algorithm in SWEPT_ALGORITHMS:
            stem = _sweep_stem(bench.bm_id, algorithm, sweep_name, severity)
            pending = [
                i for i in range(n_runs) if not (RESULT_DIR / f"{stem}_run{i}.json").exists()
            ]
            if not pending:
                tqdm.write(f"--- {stem}: all {n_runs} runs already complete ---")
                continue

            def _one_run(
                i, algorithm=algorithm, delay_model=delay_model, stem=stem
            ) -> dict:
                seed = base_seed + i
                local_bench = copy.copy(bench)
                local_bench.reset_noise(seed)
                opt = build_optimizer(algorithm, local_bench.get_search_space(), seed)
                result = run_delayed(
                    opt,
                    local_bench,
                    N_ITERATIONS,
                    seed=seed,
                    delay_model=delay_model,
                    patience_steps=PATIENCE_STEPS,
                )
                with open(RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
                    json.dump(result, f)
                return result

            print(f"{stem}: running {len(pending)}/{n_runs} pending runs...")
            Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
                delayed(_one_run)(i) for i in pending
            )


def run_reference_algorithms(
    bench: RFTabularFiniteBenchmark,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = N_JOBS,
) -> None:
    """Run REFERENCE_ALGORITHMS once each (severity-invariant, see module
    docstring), checkpointed under a fixed 'reference' tag reused by every
    sweep's plot as a flat line."""
    for algorithm in REFERENCE_ALGORITHMS:
        stem = _reference_stem(bench.bm_id, algorithm)
        pending = [
            i for i in range(n_runs) if not (RESULT_DIR / f"{stem}_run{i}.json").exists()
        ]
        if not pending:
            tqdm.write(f"--- {stem}: all {n_runs} runs already complete ---")
            continue

        def _one_run(i, algorithm=algorithm, stem=stem) -> dict:
            seed = base_seed + i
            local_bench = copy.copy(bench)
            local_bench.reset_noise(seed)
            opt = build_optimizer(algorithm, local_bench.get_search_space(), seed)
            result = run_baseline(opt, local_bench, N_ITERATIONS, seed=seed)
            with open(RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
                json.dump(result, f)
            return result

        print(f"{stem}: running {len(pending)}/{n_runs} pending runs...")
        Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_one_run)(i) for i in pending
        )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    bm_ids = [146822, 31, 167120]

    for bm_id in bm_ids:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        print(f"\n=== RF tabular benchmark (OpenML task {bm_id}) ===")

        run_reference_algorithms(bench, n_runs=n_runs, base_seed=base_seed)

        print("--- Delay severity sweep ---")
        # feedback_freq=1.0 (no censoring) isolates pure delay-length effects
        # -- otherwise the real calibrated 59% censoring rate would stay
        # fixed across every point of this sweep and dominate how much
        # pending backlog accumulates regardless of delay_scale, washing out
        # the very effect this sweep is meant to show. Censoring's own
        # effect is covered by the sweep below instead.
        run_severity_sweep(
            bench,
            "delay",
            DELAY_SEVERITIES,
            delay_model_for=lambda s: DelayModel.at_severity(
                delay_scale=s, feedback_freq=1.0
            ),
            n_runs=n_runs,
            base_seed=base_seed,
        )

        print("--- Censoring severity sweep ---")
        run_severity_sweep(
            bench,
            "censor",
            CENSORING_SEVERITIES,
            delay_model_for=lambda s: DelayModel.at_severity(feedback_freq=s),
            n_runs=n_runs,
            base_seed=base_seed,
        )
