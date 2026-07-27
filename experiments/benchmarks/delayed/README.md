# Delayed / censored feedback experiment

Does IMABO's delay-aware switching rule (`switch_strategy="delayed"`, see
`imabo/moss.py`) actually help when reward feedback arrives late or never
arrives at all, compared to a delay-oblivious IMABO exposed to the same
delayed environment, and to a no-delay skyline?

This file is the architecture overview for the whole pipeline (simulator,
delay models, benchmarks, severity sweep, plots). For one-time per-benchmark
setup commands (installing yahpo/downloading Criteo), see
[`README_lcbench.md`](README_lcbench.md).

## Pipeline at a glance

```
experiments/delayed_feedback_experiment.py            # main run: fixed, calibrated severity
experiments/delayed_feedback_severity_experiment.py   # sweep: delay length x censoring rate
experiments/utils/plots/delayed_feedback_plot.py      # figures, read from results/
experiments/benchmarks/delayed/
    simulator.py       # run_delayed / run_baseline event loop
    delay_model.py     # DelayModel, RuntimeDelayModel, CriteoDelayModel
    lcbench_bandit.py  # LCBenchMixedBenchmark   (BENCHMARK = "lcbench", default)
    criteo_bandit.py   # CriteoConversionBenchmark (BENCHMARK = "criteo")
    ../rf_tabular_bandit.py  # RFTabularFiniteBenchmark (BENCHMARK = "rf")
```

`BENCHMARK` at the top of `delayed_feedback_experiment.py` picks the active
reward surface; the severity experiment and plot script both import it, so
changing it in one place re-targets the whole pipeline (its own
`results/delayed_feedback_<tag>*` subdirectory, so runs never collide).

## The simulator (`simulator.py`)

A suggested config's reward is computed immediately but **delivered late**:
each pull's delay is drawn from a delay model and the observation is pushed
onto a min-heap keyed by arrival step. At every step, the loop:

1. `x = optimizer.suggest()`, evaluate `y = bench(x, noise=True)`.
2. Sample a delay (`delay_model.sample_delay_steps`, optionally config-dependent
   via `bench.expected_runtime_steps(x)`), and schedule `(x, y)` for delivery
   at `step + delay`.
3. Pop and `optimizer.memory.observe(config, reward)` every observation whose
   arrival step has passed.
4. Drop (censor) any pull still pending more than `patience_steps` (72) after
   it was generated — one rule covering both "arrived too late" and "was
   never going to arrive" (a `None` delay from Bernoulli censoring is just
   scheduled past the window so it flows through the same expiry path).

`run_baseline` is the synchronous comparison loop (suggest -> evaluate ->
observe immediately, no heap) — used for the no-delay skyline and for
algorithms whose regret is delay-invariant by construction (Random Search).

## Algorithms compared

| Algorithm | `switch_strategy` | Simulator loop | Measures |
|---|---|---|---|
| **IMABO-Delayed** | `"delayed"` | `run_delayed` | the delay-aware correction, in the harsh environment |
| **IMABO-Naive** | `"beta"` | `run_delayed` | the cost of ignoring delay (same optimizer as NoDelay) |
| **IMABO-NoDelay** | `"beta"` | `run_baseline` | the no-delay skyline (upper bound) |
| **Random Search** | — | `run_baseline` | severity-invariant reference line |

IMABO-Naive and IMABO-NoDelay deliberately share the exact same optimizer
config — what differs is the *environment* each is run in. IMABO-Delayed's
whole point is to recover IMABO-NoDelay's performance while running in
IMABO-Naive's environment.

## Delay models (`delay_model.py`)

| Model | Used by | Delay is | Censoring |
|---|---|---|---|
| `DelayModel` | RF | fitted log-normal (`mu=0.5795, sigma=2.3226`), config-independent | Bernoulli, `feedback_freq=0.407` (~59% never arrive) |
| `RuntimeDelayModel` | LCBench | the config's own predicted training time x log-normal jitter (median 1) | endogenous (runtime > patience window) + optional Bernoulli on top (`feedback_freq=1.0` by default = none) |
| `CriteoDelayModel` | Criteo | sampled from the **real** empirical `time_delay_for_conversion` array | Bernoulli at the real click->conversion rate |

All three expose `at_severity(delay_scale, feedback_freq)` for the severity
sweep: `delay_scale` multiplies the expected delay (`mu -> mu + log(scale)`
for the log-normal models), `feedback_freq` overrides the censoring rate.

## Benchmarks, one by one

### LCBench (`lcbench_bandit.py`) — default, `BENCHMARK = "lcbench"`

- **Ground truth:** YAHPO-Gym's surrogate for LCBench — a trained surrogate of
  an AutoPyTorch MLP's validation accuracy on OpenML tasks. Default instances:
  `126026` (higgs), `168330` (jannis), `167104` (Fashion-MNIST) — disjoint from
  the RF and LR/SVM tasks used elsewhere in the paper.
- **Search space** is genuinely **mixed continuous + finite** (7 HPs):
  `num_layers` is categorical `{1..5}`; `batch_size`/`max_units` are integer
  (log-scaled); `learning_rate`/`momentum`/`weight_decay`/`max_dropout` are
  continuous. On the continuous axes the arm set is infinite — the regime
  IMABO's TPE oracle targets, unlike the finite RF grid below.
- **Reward:** surrogate `val_accuracy` (0-100) rescaled to `[0,1]` as a
  Bernoulli success probability; pulling an arm draws one Bernoulli sample.
  `max_value` (the reference optimum) has no closed form on a continuous
  space, so it's estimated once by dense random sampling (100k configs) and
  cached to `assets/lcbench_<instance>_<metric>_ref.json`.
- **Delay is endogenous:** each pull's delay *is* that configuration's own
  surrogate-predicted training time (`expected_runtime_steps`), rescaled so
  the *median* config's delay is ~6 steps (`_TARGET_MEDIAN_DELAY_STEPS`) —
  an expensive net reports late, a job whose runtime exceeds the patience
  window is censored (the crashed/preempted-job story). Paired with
  `RuntimeDelayModel`.

### RF tabular (`../rf_tabular_bandit.py`) — `BENCHMARK = "rf"`

- **Ground truth:** precomputed RandomForestClassifier validation accuracy
  (HPOBench, max fidelity) over a discretized 4D grid (`max_depth x
  max_features x min_samples_leaf x min_samples_split`) on 3 OpenML tasks
  (`146822`, `31`, `167120`), coarsened to a few hundred arms per task.
- **Search space:** purely finite, 4 categorical dimensions.
- **Reward:** `Bernoulli(accuracy)`; exact regret since the grid is
  precomputed and finite.
- **Delay is exogenous/injected:** the fitted log-normal `DelayModel`,
  calibrated from real production ("Gleipnir") feedback timing — this is the
  original setting the delayed-feedback experiment was built around, kept for
  reproducibility.

### Criteo (`criteo_bandit.py`) — `BENCHMARK = "criteo"`

- **Ground truth:** the real Criteo Sponsored Search Conversion Log. Each row
  is a real ad click with a binary `Sale` and `time_delay_for_conversion`.
- **Arms:** a *segment* — a composite of low-cardinality categoricals
  (`product_category1 x device_type x product_age_group x product_gender x
  product_brand`), min-support filtered — a **large finite** arm set
  (thousands), exposed to IMABO as one categorical dimension.
- **Reward:** `Bernoulli(segment's empirical conversion rate)`; exact regret
  against `p* = max_x p_x`.
- **Delay is real, empirical data:** sampled directly from the log's
  `time_delay_for_conversion` field via `CriteoDelayModel`; censoring = the
  real fraction of clicks that never convert. This is the delay distribution
  the delayed-feedback bandit literature is built on (Chapelle 2014, KDD;
  Vernade, Cappe & Perchet 2017, UAI).
- Requires a one-time asset build from the raw log — see
  `build_criteo_asset.py` and [`README_lcbench.md`](README_lcbench.md) for the
  download + build steps.

The three benchmarks stress different axes: RF = finite arms + generic
injected delay; LCBench = mixed/continuous arms + delay intrinsic to the
config (realistic async-HPO); Criteo = large finite arms + delay that's real
production data end-to-end.

## Severity sweep (`delayed_feedback_severity_experiment.py`)

The main experiment runs at one fixed, calibrated severity level. The sweep
instead varies severity along two independent axes and looks at the **final**
cumulative regret (shorter horizon, `N_ITERATIONS = 2000`, since only the
endpoint matters):

- **Delay severity** (`DELAY_SEVERITIES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`):
  multiplies the expected delay, with censoring turned **off**
  (`feedback_freq=1.0`) so this isolates pure delay-length effects — a fixed
  censoring rate would otherwise dominate the pending backlog and wash out
  any dependence on delay length.
- **Censoring severity** (`CENSORING_SEVERITIES = [0.1, 0.2, 0.407, 0.6, 0.8,
  1.0]`): overrides `feedback_freq` directly, delay shape fixed at the real
  calibrated value.

Only IMABO-Delayed/IMABO-Naive are re-run per severity point (they're the
only algorithms that consult a delay model); IMABO-NoDelay/Random are
severity-invariant and run once as flat reference lines.

## Plots (`experiments/utils/plots/delayed_feedback_plot.py`)

`DATA_DIR`/`SEVERITY_DATA_DIR`/figure filenames all key off the active
`BENCHMARK`, so the same module serves all three benchmark families without
edits. The figures currently generated by the script's `__main__` block:

- **`plot_delay_distribution`** — calibration figure, not derived from any run
  checkpoint. Samples the active delay model directly and histograms delay on
  a log-x axis (heavy-tailed, so linear crushes it into one bin), annotating
  the median delay, the patience window (72 steps), and the censored
  fraction.
- **`plot_regret_vs_delay_severity`** / **`plot_regret_vs_censoring_severity`**
  — one subplot per benchmark instance, x-axis = severity (delay multiplier,
  log scale, or feedback frequency, linear), y-axis = final cumulative regret
  averaged over seeds (+/- std band). IMABO-Delayed/Naive as lines; flat
  dashed reference lines for IMABO-NoDelay and Random Search.
- **`plot_delay_effectiveness`** — the headline figure. Two panels (delay
  severity | censoring severity); regret is normalized per-benchmark by that
  benchmark's own IMABO-NoDelay skyline, then averaged across benchmark
  instances, so every task lands on one comparable "1.0 = matches skyline"
  scale. Shows in one glance whether IMABO-Delayed tracks the skyline as the
  regime worsens while IMABO-Naive drifts away from it.

`plot_cumulative_regret_grid`, `plot_pending_queue_grid`, and
`plot_regret_vs_arrivals` are still defined in the module (regret-vs-iteration,
pending-queue-size-vs-iteration, and regret-vs-arrivals-this-step,
respectively) but are no longer called from `__main__`; call them directly if
you need those diagnostics.

## Running everything

```bash
# pick BENCHMARK in experiments/delayed_feedback_experiment.py first
python -m experiments.delayed_feedback_experiment           # main run
python -m experiments.delayed_feedback_severity_experiment  # severity sweeps
python -m experiments.utils.plots.delayed_feedback_plot     # figures
```

See [`README_lcbench.md`](README_lcbench.md) for one-time setup
(`setup_lcbench.py`, `build_criteo_asset.py`) and per-benchmark result/figure
paths.
