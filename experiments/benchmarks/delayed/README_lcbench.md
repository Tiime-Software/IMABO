# Delayed/censored feedback benchmarks

The delayed-feedback experiment supports several reward surfaces, selected by
`BENCHMARK` at the top of `experiments/delayed_feedback_experiment.py`. Each is
paired with the delay model natural to it; all flow through the same simulator
and plot script.

| `BENCHMARK` | Reward surface | Arms | Delay model |
|-------------|----------------|------|-------------|
| `"lcbench"` (default) | LCBench NN-HPO surrogate | mixed continuous+finite, ∞ | endogenous: config's predicted training time |
| `"nasbench201"` | NAS-Bench-201 tabular NAS cell | **structured finite** (6 edges × 5 ops = 15625) | endogenous: architecture's real training time |
| `"rf"` | RF tabular HPOBench grid | finite (~2250) | fitted log-normal (injected) |
| `"criteo"` | Criteo ad-conversion log | large finite, **unstructured** (thousands of segments) | **real** click→conversion delay |

**On `"criteo"`:** its arms are atomic segment ids with no shared geometry, so
IMABO's structured TPE oracle cannot generalize across them and the base problem
is hard to learn within a normal budget (it underuses IMABO). Prefer
`"nasbench201"` for a structured, many-armed, real-delay benchmark; Criteo is
retained only as a real delayed-conversion-data reference at small arm counts.

The LCBench option (the default) is described first, then NAS-Bench-201, then
Criteo at the bottom.

---

## LCBench (mixed continuous+finite)

**LCBench** (YAHPO-Gym surrogate) is a mixed continuous+finite neural-network
HPO problem with a **production-realistic, endogenous delay**.

## Why this benchmark

- **Mixed continuous + finite space** (7 HPs): `num_layers` is a finite
  categorical axis `{1..5}`; `batch_size`/`max_units` are integer (log);
  `learning_rate`/`momentum`/`weight_decay`/`max_dropout` are continuous.
  On the continuous axes the arm set is *infinite* — the regime IMABO's TPE
  oracle is built for, and a strictly richer test than the RF grid's finite
  arms.
- **Delay is endogenous, not injected.** The delay of each pull *is* that
  configuration's own surrogate-predicted training time (LCBench ships a `time`
  objective). An expensive net (many layers/units, small batch) reports its
  score late; a job whose runtime exceeds the patience window is *censored*
  (the crashed/preempted-job story). This is how the asynchronous-HPO
  literature builds delayed benchmarks (Watanabe et al. 2024, arXiv:2403.01888;
  Zimmer et al. 2021, arXiv:2006.13799) and removes the dependence on an
  unverifiable external timing log.
- **Fast.** Surrogate lookup (ONNX), no LLM, no model training.
- **Not already used in the paper.** Default instances (`126026` higgs,
  `168330` jannis, `167104` Fashion-MNIST) are LCBench tasks disjoint from the
  RF tasks (146822, 31, 167120) and the LR/SVM task (167149).

## One-time setup

```bash
# into your env (yahpo pins the pre-1.0 ConfigSpace API):
pip install "yahpo-gym>=1.0" onnxruntime "ConfigSpace>=0.6,<1.0" pyarrow

# download the surrogate ONNX models + metadata and smoke-test:
python -m experiments.benchmarks.delayed.setup_lcbench
# (or: --data-dir /path/to/yahpo_data)
```

## Run

```bash
# BENCHMARK = "lcbench" is the default at the top of
# experiments/delayed_feedback_experiment.py. To reproduce the original finite
# RF setting instead, set BENCHMARK = "rf" there.

python -m experiments.delayed_feedback_experiment           # T=10000, 4 algos x 3 instances
python -m experiments.delayed_feedback_severity_experiment  # delay + censoring sweeps
python -m experiments.utils.plots.delayed_feedback_plot     # all figures
```

Results land in `results/delayed_feedback_lcbench/` and
`results/delayed_feedback_lcbench_severity/` (the RF results in
`results/delayed_feedback*/` are never touched). Figures are written to
`results/paper_plots/lcbench_delayed_feedback_*.pdf`.

## What each piece does

| File | Role |
|------|------|
| `lcbench_bandit.py` | `LCBenchMixedBenchmark`: surrogate reward (Bernoulli(val_acc/100)) + per-config `expected_runtime_steps`. Duck-type compatible with `RFTabularFiniteBenchmark`. |
| `delay_model.py` | adds `RuntimeDelayModel`: delay = config runtime × log-normal jitter; Bernoulli censoring on top; `at_severity()` for the sweep. |
| `simulator.py` | `run_delayed` now passes `bench.expected_runtime_steps(x)` to the delay model when available; RF path (no such method) is unchanged. |
| `setup_lcbench.py` | one-time surrogate download + registration + smoke test. |

## Notes / gotchas

- **Parallelism.** The RF experiment used `N_JOBS=8` threading. The yahpo ONNX
  session may not be thread-safe under concurrent `objective_function` calls; if
  you see nondeterministic surrogate values or ORT errors, set `N_JOBS=1` in
  `delayed_feedback_experiment.py` (the surrogate is fast, so the wall-clock
  cost is modest), or give each thread its own `benchmark_set.BenchmarkSet`.
- **Reference optimum.** Regret needs `p* = max accuracy`, which a continuous
  space has no closed form for; it is estimated by a dense (100k) random sample
  over the surrogate and cached to `assets/lcbench_<instance>_<metric>_ref.json`.
  The same sample sets the delay time-scale (median config ≈ 6 steps). Delete
  the cache to re-estimate.
- **`patience_steps`.** Kept at 72 (from the RF setup). With the runtime delay
  rescaled so the median config ≈ 6 steps, most configs arrive well inside the
  window and only the slow tail is censored — tune `target_median_delay_steps`
  in `lcbench_bandit.py` if you want a heavier endogenous-censoring regime.

---

## NAS-Bench-201 (structured finite, runtime delay)

**NAS-Bench-201** (Dong & Yang 2020, arXiv:2001.00326; queried via
`nats_bench`, the NATS-Bench topology search space, Dong et al. 2021,
arXiv:2009.00437) is a tabular neural-architecture-search benchmark. It is the
structured, many-armed, real-delay benchmark for the delayed experiment.

### What it is
- **Arm** = a cell of **6 edges**, each assigned one of **5 operations**
  (`none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`) →
  5⁶ = 15625 architectures. Exposed to IMABO as **6 categorical axes**, so its
  TPE oracle generalizes over good-vs-bad operations per edge instead of
  enumerating all 15625 — the structured many-arm regime IMABO targets (and the
  reason this works where Criteo's unstructured segment arms do not).
- **Reward** = `Bernoulli(val_accuracy / 100)` → **0/1**, matching LCBench/RF.
  Exact regret against `p* = max over all 15625 architectures` (enumerated once,
  cached — exact, not sampled, because the space is finite).
- **Delay** = `RuntimeDelayModel`: each architecture's **real recorded training
  time** is the delay (same conception as LCBench). Censoring = a job whose
  training time exceeds the patience window (crashed/preempted). No injected or
  conversion-signal model — reuses the LCBench delay machinery unchanged.
- **Datasets** (instances): `cifar100` (default), `cifar10`, `ImageNet16-120` —
  image classification, distinct from the OpenML tabular tasks used elsewhere.

### One-time setup
```bash
pip install nats_bench
# Download the topology-search-space archive `NATS-tss-v1_0-3ffb9-simple`
# from the NATS-Bench release (github.com/D-X-Y/NATS-Bench; access-gated,
# not on any auto-download allowlist) and extract it, then:
python -m experiments.benchmarks.delayed.setup_nasbench201 \
    --nats-path /path/to/NATS-tss-v1_0-3ffb9-simple
#   (default path: experiments/benchmarks/delayed/assets/NATS-tss-v1_0-3ffb9-simple)
```

### Run
```bash
# set BENCHMARK = "nasbench201" in delayed_feedback_experiment.py, then:
python -m experiments.delayed_feedback_experiment
python -m experiments.delayed_feedback_severity_experiment
python -m experiments.utils.plots.delayed_feedback_plot
```
Results land in `results/delayed_feedback_nasbench201{,_severity}/`; figures in
`results/paper_plots/nb201*_delayed_feedback_*.pdf`. Other benchmarks' results
are never touched.

### Files
| File | Role |
|------|------|
| `nasbench201_bandit.py` | `NASBench201Benchmark`: 6×5 structured space, 0/1 reward, exact p*, `expected_runtime_steps`. Duck-type compatible with the other benchmarks. |
| `setup_nasbench201.py` | one-time: verify the NATS-Bench archive + build the ref cache. |
| `delay_model.py` | reuses `RuntimeDelayModel` (delay = training time) — no NAS-specific delay model. |

---

## Criteo ad-conversion bandit (large finite, real delay)

A widely used delay benchmark. Two standard references for delayed conversion
feedback: Chapelle (2014, "Modeling Delayed Feedback in Display Advertising",
KDD '14, pp. 1097-1105, doi:10.1145/2623330.2623634) and Vernade, Cappe &
Perchet (2017, "Stochastic Bandit Models for Delayed Conversions", UAI 2017,
arXiv:1706.09186) -- the latter is a delayed-conversion bandit model built in
Chapelle's framework with censored late conversions.

### What it is
- **Dataset:** Criteo *Sponsored Search Conversion Log* (Criteo AI Lab,
  terms-of-use gated). Each row is a real ad click with product/context
  features, a binary `Sale` (did it convert), and `time_delay_for_conversion`
  (seconds click→conversion, `-1` if it never converted).
- **Arm** = a *segment* — a composite of low-cardinality categoricals
  (`product_category1 × device_type × product_age_group × product_gender ×
  product_brand`), min-support filtered. A **large finite** arm set (thousands),
  exposed to IMABO as one categorical dimension whose choices are the populated
  segment keys — so every proposable arm is real and well-populated.
- **Reward** = `Bernoulli(segment conversion rate)` → **0/1** (the `Sale`
  event), drawn by the benchmark. Exact regret against `p* = max_x p_x`. The
  conversion rate is low, so this is a **low-mean** bandit — not a heavily
  censored one.
- **Delay** = `CriteoDelayModel`, a **conversion-signal** model. A conversion
  (reward 1) arrives after a **real** `time_delay_for_conversion`; a
  non-conversion (reward 0) is an **observed 0** resolved at the patience
  deadline.
- **Censoring** = a real conversion whose delay exceeds the patience window.
  At the deadline the learner cannot tell a pending late conversion from a true
  non-conversion (no ping either way), so **any pull unresolved at the deadline
  is observed as reward 0** — a *wrong 0* for a late conversion. That
  late-conversion-as-0 **bias** (not missing data) is the delayed-conversion
  problem a delay-aware learner must correct (Chapelle 2014; Vernade et al.
  2017). A reward of 0 is always observed — never dropped. The
  censoring-severity sweep for Criteo therefore varies the **patience window**
  (a target quantile of the conversion delays, so a fraction `q` of conversions
  land in time), not a Bernoulli feedback rate. This differs from RF/LCBench,
  whose expired feedback is genuine missing data (`CENSOR_MODE="drop"`); Criteo
  is `CENSOR_MODE="observe_zero"`.

Note: because arms are atomic segment ids (no shared coordinate geometry),
IMABO's TPE oracle here behaves as smart resampling of good arms rather than
structural generalization. Criteo's contribution is the *large-arm + real-delay*
story; LCBench/RF carry the *structured-oracle* story.

### One-time setup
```bash
# 1. Download the Sponsored Search Conversion Log (CriteoSearchData) from
#    https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/
#    (accept the terms; it is not on any auto-download allowlist).

# 2. Build the compact assets (per-segment rates + empirical delay array):
python -m experiments.benchmarks.delayed.build_criteo_asset \
    --raw /path/to/CriteoSearchData
#    -> assets/criteo_sponsored_search_arms.csv   (segment, conversion_rate, count)
#    -> assets/criteo_sponsored_search_delay.npz  (real delays + conversion rate)
#    Options: --min-count (arm support floor, default 200),
#             --segment-cols (which categoricals define an arm),
#             --seconds-per-step (delay time unit, default 3600),
#             --max-rows (cap for a quick smaller build).
```

### Run
```bash
# set BENCHMARK = "criteo" in experiments/delayed_feedback_experiment.py, then:
python -m experiments.delayed_feedback_experiment
python -m experiments.delayed_feedback_severity_experiment
python -m experiments.utils.plots.delayed_feedback_plot
```
Results land in `results/delayed_feedback_criteo{,_severity}/`; figures in
`results/paper_plots/criteo*_delayed_feedback_*.pdf`. The LCBench and RF results
are never touched.

### Files
| File | Role |
|------|------|
| `build_criteo_asset.py` | one-time: raw log → per-segment rates CSV + empirical-delay npz. |
| `criteo_bandit.py` | `CriteoConversionBenchmark`: 0/1 reward from segment conversion rate; exact regret; duck-type compatible with the other benchmarks. |
| `delay_model.py` | `CriteoDelayModel`: samples the real empirical delay array; Bernoulli censoring at the real conversion rate; `at_severity()` for the sweep. |
