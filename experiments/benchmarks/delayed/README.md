# Delayed / censored feedback experiment

Does IMABO's delay-aware switching rule (`switch_strategy="delayed"`, see
`imabo/moss.py`) actually help when reward feedback arrives late or never
arrives at all, compared to a delay-oblivious IMABO exposed to the same
delayed environment, and to a no-delay skyline?

This file covers the whole pipeline (simulator, delay models, benchmarks,
severity sweep, plots) plus per-benchmark setup and run instructions.

## Pipeline at a glance

```
experiments/delayed_feedback_experiment.py            # main run + severity sweep (one file, one __main__)
experiments/utils/plots/delayed_feedback_plot.py      # figures, read from results/
experiments/benchmarks/delayed/
    simulator.py           # run_delayed / run_baseline event loop, patience_for_quantile
    delay_model.py          # RuntimeDelayModel
    lcbench_bandit.py       # LCBenchMixedBenchmark    (BENCHMARK = "lcbench", default)
    nasbench201_bandit.py   # NASBench201Benchmark      (BENCHMARK = "nasbench201")
    setup_lcbench.py        # one-time surrogate download + smoke test
    setup_nasbench201.py    # one-time archive verify + ref-cache build
```

`BENCHMARK` at the top of `delayed_feedback_experiment.py` picks the active
reward surface (only `"lcbench"` and `"nasbench201"` are supported — anything
else raises); the plot script imports it too, so changing it in one place
re-targets the whole pipeline (its own `results/delayed_feedback_<tag>*`
subdirectory, so runs never collide). The severity sweep is **not** a
separate script — it runs from the same `__main__` block, right after the
main experiment, for whichever benchmark is currently selected.

## The simulator (`simulator.py`)

A suggested config's reward is computed immediately but **delivered late**:
each pull's delay is drawn from a delay model and the observation is pushed
onto a min-heap keyed by arrival step. At every step, the loop:

1. `x = optimizer.suggest()`, evaluate `y = bench(x, noise=True)`.
2. Sample a delay (`delay_model.sample_delay_steps`, config-dependent via
   `bench.expected_runtime_steps(x)` for both current benchmarks), and
   schedule `(x, y)` for delivery at `step + delay`.
3. **Check expiry before arrival**: any pull still pending more than
   `patience_steps` after it was generated is dropped (censored) — one rule
   covering both "arrived too late" and "was never going to arrive" (a
   `None` delay from Bernoulli censoring is parked past the window so it
   flows through the same expiry path). This ordering matters: checking
   arrival first would deliver an already-expired pull instead of censoring
   it.
4. Pop and `optimizer.memory.observe(config, reward)` every remaining
   observation whose arrival step has passed.

`run_baseline` is the synchronous comparison loop (suggest -> evaluate ->
observe immediately, no heap) — used for the no-delay skyline and for
algorithms whose regret is delay-invariant by construction (UCB-AIR).

`patience_steps` is not a fixed constant: `simulator.patience_for_quantile`
computes it **per benchmark** as the `PATIENCE_QUANTILE`-th percentile
(0.95 by default) of that benchmark's own delay distribution, so the
patience-induced censoring rate is identical across benchmarks with very
different delay time-scales, instead of an arbitrary shared cutoff. It falls
back to `PATIENCE_STEPS_FALLBACK` (72) only if a delay model produces no
positive delays at all (e.g. `feedback_freq=0`).

## Algorithms compared

| Algorithm | `switch_strategy` | Simulator loop | Measures |
|---|---|---|---|
| **IMABO-Delayed** | `"delayed"` | `run_delayed` | the delay-aware correction, in the harsh environment |
| **IMABO-Naive** | `"beta"` | `run_delayed` | the cost of ignoring delay (same optimizer as NoDelay) |
| **IMABO-NoDelay** | `"beta"` | `run_baseline` | the no-delay skyline (upper bound) |
| **UCB-AIR** | — | `run_baseline` | severity-invariant reference line |

IMABO-Naive and IMABO-NoDelay deliberately share the exact same optimizer
config — what differs is the *environment* each is run in. IMABO-Delayed's
whole point is to recover IMABO-NoDelay's performance while running in
IMABO-Naive's environment. UCB-AIR never consults a delay model (always
`run_baseline`), so its regret is severity-invariant and it is run once, not
re-swept.

## Delay models (`delay_model.py`)

Both current benchmarks use the same model: delay *is* the config's own
training time, not an injected/exogenous nuisance.

| Model | Used by | Delay is | Censoring |
|---|---|---|---|
| `RuntimeDelayModel` | LCBench | the config's own surrogate-predicted training time x log-normal jitter (median 1) | endogenous (runtime > patience window) + Bernoulli on top |
| `RuntimeDelayModel` | NAS-Bench-201 | the architecture's own **real recorded** training time (no jitter needed — it's already real) | endogenous (runtime > patience window) + Bernoulli on top |

The class itself defaults to `feedback_freq=1.0` (no *extra* Bernoulli
censoring beyond the endogenous runtime one), but
`delayed_feedback_experiment.default_delay_model()` overrides this to the
calibrated `DEFAULT_FEEDBACK_FREQ = 0.2` for both the main run and the
severity sweep's delay-severity axis — i.e. 80% of pulls are Bernoulli-censored
upfront (crashed/never-submitted job), stacking with whatever the patience
window censors on top. `RuntimeDelayModel.at_severity(delay_scale,
feedback_freq)` builds a severity-sweep variant: `delay_scale` multiplies the
expected delay, `feedback_freq` overrides the Bernoulli rate.

## Benchmarks, one by one

Selected via `BENCHMARK` at the top of `experiments/delayed_feedback_experiment.py`.

| `BENCHMARK` | Reward surface | Arms | Delay model |
|-------------|----------------|------|-------------|
| `"lcbench"` (default) | LCBench NN-HPO surrogate | mixed continuous+finite, ∞ | endogenous: config's predicted training time |
| `"nasbench201"` | NAS-Bench-201 tabular NAS cell | **structured finite** (6 edges × 5 ops = 15625) | endogenous: architecture's real training time |

LCBench and NAS-Bench-201 stress different axes: LCBench = mixed/continuous
arms (the regime IMABO's TPE oracle is built for) + delay intrinsic to a
*surrogate-predicted* runtime; NAS-Bench-201 = a large **structured finite**
arm set (still IMABO's structured-oracle regime, but purely categorical) +
delay from **real recorded** training time.

### LCBench (mixed continuous+finite)

**LCBench** (YAHPO-Gym surrogate) is a mixed continuous+finite neural-network
HPO problem with a production-realistic, endogenous delay.

**Why this benchmark**

- **Mixed continuous + finite space** (7 HPs): `num_layers` is a finite
  categorical axis `{1..5}`; `batch_size`/`max_units` are integer (log);
  `learning_rate`/`momentum`/`weight_decay`/`max_dropout` are continuous.
  On the continuous axes the arm set is *infinite* — the regime IMABO's TPE
  oracle is built for.
- **Delay is endogenous, not injected.** The delay of each pull *is* that
  configuration's own surrogate-predicted training time (LCBench ships a
  `time` objective). An expensive net (many layers/units, small batch)
  reports its score late; a job whose runtime exceeds the patience window is
  *censored* (the crashed/preempted-job story). This is how the
  asynchronous-HPO literature builds delayed benchmarks (Watanabe et al.
  2024, arXiv:2403.01888; Zimmer et al. 2021, arXiv:2006.13799) and removes
  the dependence on an unverifiable external timing log.
- **Fast.** Surrogate lookup (ONNX), no LLM, no model training.
- **Ground truth:** YAHPO-Gym's surrogate for LCBench — a trained surrogate
  of an AutoPyTorch MLP's validation accuracy on OpenML tasks. Default
  instances (`LCBENCH_INSTANCES`): `167200` (higgs), `168868` (APSFailure),
  `189908` (Fashion-MNIST) — verified directly against OpenML task IDs (an
  earlier version of this mapping had 126026 mislabeled "higgs" and 167104
  mislabeled "Fashion-MNIST"; those were wrong and have been corrected in
  `LCBENCH_INSTANCE_NAMES`). `168330` (jannis) and `189354` (airlines) are
  also available but not used by default.
- **Reward:** surrogate `val_accuracy` (0-100) rescaled to `[0,1]` as a
  Bernoulli success probability; pulling an arm draws one Bernoulli sample.
  `max_value` (the reference optimum) has no closed form on a continuous
  space, so it's estimated once by dense random sampling (100k configs) and
  cached to `assets/lcbench_<instance>_<metric>_ref.json`.
- **Delay rescaling:** the endogenous training-time delay is rescaled so the
  *median* config's delay is ~6 steps (`_TARGET_MEDIAN_DELAY_STEPS`), paired
  with `RuntimeDelayModel`.

**One-time setup**

```bash
# into your env (yahpo pins the pre-1.0 ConfigSpace API):
pip install "yahpo-gym>=1.0" onnxruntime "ConfigSpace>=0.6,<1.0" pyarrow

# download the surrogate ONNX models + metadata and smoke-test:
python -m experiments.benchmarks.delayed.setup_lcbench
# (or: --data-dir /path/to/yahpo_data)
```

**Run**

```bash
# BENCHMARK = "lcbench" is the default at the top of
# experiments/delayed_feedback_experiment.py.

python -m experiments.delayed_feedback_experiment       # main run (T=10000) + severity sweep, 3 instances
python -m experiments.utils.plots.delayed_feedback_plot # all figures
```

Results land in `results/delayed_feedback_lcbench/` (main run) and
`results/delayed_feedback_lcbench_severity/` (severity sweep). Figures are
written to `results/paper_plots/`, named `<benchmark-tags>_delayed_feedback_
<figure>.pdf` (e.g. `lc167200_lc168868_lc189908_delayed_feedback_regret_grid.pdf`),
except the delay-distribution calibration figure, which is
`lcbench_delayed_feedback_delay_distribution.pdf`.

**Files**

| File | Role |
|------|------|
| `lcbench_bandit.py` | `LCBenchMixedBenchmark`: surrogate reward (Bernoulli(val_acc/100)) + per-config `expected_runtime_steps`. Duck-type compatible with the other benchmark. |
| `delay_model.py` | `RuntimeDelayModel`: delay = config runtime × log-normal jitter; Bernoulli censoring on top; `at_severity()` for the sweep. |
| `simulator.py` | `run_delayed` passes `bench.expected_runtime_steps(x)` to the delay model when available. |
| `setup_lcbench.py` | one-time surrogate download + registration + smoke test. |

**Notes / gotchas**

- **Parallelism.** `N_JOBS = 1` in `delayed_feedback_experiment.py` because
  the yahpo ONNX session (and, for the other benchmark, the `nats_bench`
  tabular API handle) may not be thread-safe under concurrent queries. The
  surrogate lookup is fast, so the wall-clock cost of serial execution is
  modest; raise `N_JOBS` only if your yahpo/ORT build is confirmed
  thread-safe.
- **Reference optimum.** Regret needs `p* = max accuracy`, which a continuous
  space has no closed form for; it is estimated by a dense (100k) random
  sample over the surrogate and cached to
  `assets/lcbench_<instance>_<metric>_ref.json`. The same sample sets the
  delay time-scale (median config ≈ 6 steps). Delete the cache to
  re-estimate.
- **`patience_steps`.** Per-benchmark, computed from
  `PATIENCE_QUANTILE = 0.95` (see the simulator section above), not a fixed
  72 — `PATIENCE_STEPS_FALLBACK = 72` is only the fallback if that
  computation fails.

### NAS-Bench-201 (structured finite, real runtime delay)

**NAS-Bench-201** (Dong & Yang 2020, arXiv:2001.00326; queried via
`nats_bench`, the NATS-Bench topology search space, Dong et al. 2021,
arXiv:2009.00437) is a tabular neural-architecture-search benchmark — the
structured, many-armed, real-delay benchmark for the delayed experiment.

**What it is**

- **Arm** = a cell of **6 edges**, each assigned one of **5 operations**
  (`none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`) →
  5⁶ = 15625 architectures. Exposed to IMABO as **6 categorical axes**, so
  its TPE oracle generalizes over good-vs-bad operations per edge instead of
  enumerating all 15625 — the structured many-arm regime IMABO targets.
- **Reward** = `Bernoulli(val_accuracy / 100)` → **0/1**, matching LCBench.
  Exact regret against `p* = max over all 15625 architectures` (enumerated
  once, cached — exact, not sampled, because the space is finite).
- **Delay** = `RuntimeDelayModel`: each architecture's **real recorded
  training time** is the delay (same conception as LCBench). Censoring = a
  job whose training time exceeds the patience window (crashed/preempted).
  No injected or surrogate-jittered model — reuses the LCBench delay
  machinery unchanged.
- **Datasets** (`NASBENCH201_INSTANCES`): `cifar100` (default), `cifar10`,
  `ImageNet16-120` — image classification, distinct from the OpenML tabular
  tasks used elsewhere.

**One-time setup**

```bash
pip install nats_bench
# Download the topology-search-space archive `NATS-tss-v1_0-3ffb9-simple`
# from the NATS-Bench release (github.com/D-X-Y/NATS-Bench; access-gated,
# not on any auto-download allowlist) and extract it, then:
python -m experiments.benchmarks.delayed.setup_nasbench201 \
    --nats-path /path/to/NATS-tss-v1_0-3ffb9-simple
#   (default path: experiments/benchmarks/delayed/assets/NATS-tss-v1_0-3ffb9-simple)
```

**Run**

```bash
# set BENCHMARK = "nasbench201" in delayed_feedback_experiment.py, then:
python -m experiments.delayed_feedback_experiment        # main run + severity sweep, 3 instances
python -m experiments.utils.plots.delayed_feedback_plot  # all figures
```

Results land in `results/delayed_feedback_nasbench201{,_severity}/`; figures
in `results/paper_plots/`, named `<benchmark-tags>_delayed_feedback_<figure>.pdf`
(e.g. `nb201cifar100_nb201cifar10_nb201ImageNet16-120_delayed_feedback_regret_grid.pdf`),
except the delay-distribution calibration figure, which is
`nasbench201_delayed_feedback_delay_distribution.pdf`.

**Files**

| File | Role |
|------|------|
| `nasbench201_bandit.py` | `NASBench201Benchmark`: 6×5 structured space, 0/1 reward, exact p*, `expected_runtime_steps`. Duck-type compatible with the other benchmark. |
| `setup_nasbench201.py` | one-time: verify the NATS-Bench archive + build the ref cache. |
| `delay_model.py` | reuses `RuntimeDelayModel` (delay = training time) — no NAS-specific delay model. |

## Severity sweep

Runs from the same `__main__` block as the main experiment, in
`experiments/delayed_feedback_experiment.py`, right after the fixed-severity
run for each benchmark instance. It varies severity along two independent
axes and looks at the **final** cumulative regret (`SEVERITY_N_ITERATIONS =
10000`):

- **Delay severity** (`DELAY_SEVERITIES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`):
  multiplies the expected delay, with the Bernoulli censoring rate held at
  the same calibrated `DEFAULT_FEEDBACK_FREQ = 0.2` as the main run, so this
  axis isolates pure delay-length effects under the main run's censoring
  regime.
- **Censoring severity** (`CENSORING_SEVERITIES = [0.1, 0.2, 0.4, 0.6, 0.8,
  1.0]`): overrides `feedback_freq` directly, delay shape fixed at
  `delay_scale=1.0`.

Only IMABO-Delayed/IMABO-Naive (`SWEPT_ALGORITHMS`) are re-run per severity
point — they're the only algorithms that consult a delay model.
IMABO-NoDelay/UCB-AIR (`REFERENCE_ALGORITHMS`) are severity-invariant and run
once (`run_reference_algorithms`) as flat reference lines. The patience
window for the sweep is computed once from the base (`delay_scale=1.0,
feedback_freq=1.0`) delay model and held fixed across every severity point,
so a larger `delay_scale` genuinely pushes more pulls past a fixed window
instead of moving the window with the delays.

## Plots (`experiments/utils/plots/delayed_feedback_plot.py`)

`DATA_DIR`/`SEVERITY_DATA_DIR`/figure filenames all key off the active
`BENCHMARK`, so the same module serves both benchmark families without
edits. The figures currently generated by the script's `__main__` block:

- **`plot_cumulative_regret_grid`** — the headline comparison: cumulative
  regret vs. iteration, one subplot per benchmark instance, one line per
  algorithm. Does IMABO-Delayed recover most of the no-delay skyline's
  performance while IMABO-Naive (same environment, ignores pending/censored
  pulls) lags behind it?
- **`plot_regret_vs_observations`** — simple regret of the best config found
  vs. the *cumulative number of rewards actually received* rather than
  iterations elapsed, normalizing away the sample-count gap caused by
  censoring (IMABO-Delayed needs ~5x more steps to collect the same number
  of observations as IMABO-NoDelay under the default 20% feedback rate): given
  the same amount of real feedback, does the delayed/censored learner climb
  toward the optimum as fast?
- **`plot_active_set_grid`** — rewards observed per active arm
  (`cumsum(num_arrived_this_step) / num_active`) vs. iteration: how much
  confirmed feedback backs up each arm the optimizer is currently tracking,
  as one combined statistic instead of two separately-scaled curves.
- **`plot_regret_vs_censoring_severity`** — final average regret vs.
  feedback frequency (linear x-axis), one subplot per benchmark instance, one
  line per swept algorithm plus flat reference lines for the skyline and
  UCB-AIR.

Defined but **not** called from `__main__` (call directly for these
diagnostics):

- **`plot_censoring_comparison_grid`** — isolates delay cost from censoring
  cost by comparing the default `feedback_freq=0.2` condition against a
  `feedback_freq=1.0` (no extra censoring) condition, checkpointed under
  `stem_suffix="_ff100"`.
- **`plot_pending_queue_grid`** — pending-feedback queue size over time.
- **`plot_reward_arrivals_histogram`** — distribution of how many rewards
  arrive per step, IMABO-NoDelay (always exactly 1) vs. IMABO-Delayed (sparse
  and bursty).
- **`plot_delay_distribution`** — calibration figure, not derived from any
  run checkpoint: histograms the active delay model's samples on a log-x
  axis, annotating the median delay, the patience window, and the censored
  fraction.
- **`plot_regret_vs_arrivals`** — instantaneous regret vs. how much feedback
  arrived that step.
- **`plot_cumulative_regret_by_severity`** — cumulative regret vs. iteration
  across an entire severity sweep (not just the final point), color-coded by
  severity.
- **`plot_regret_vs_delay_severity`** — final regret vs. delay severity
  (log-x), the delay-axis counterpart to `plot_regret_vs_censoring_severity`.
- **`plot_delay_effectiveness`** — two-panel headline effectiveness figure:
  final regret normalized by the no-delay skyline (=1.0), averaged across
  benchmark instances, vs. delay severity (left) and censoring severity
  (right).

## Running everything

```bash
# pick BENCHMARK in experiments/delayed_feedback_experiment.py first
# (and run the matching one-time setup above: setup_lcbench.py or setup_nasbench201.py)
python -m experiments.delayed_feedback_experiment        # main run + severity sweep
python -m experiments.utils.plots.delayed_feedback_plot  # figures
```
