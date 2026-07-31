# IMABO: Infinite Multi-Armed Bandits with Oracles

IMABO is an online hyperparameter optimization algorithm that combines multi-armed bandit exploitation (MOSS) with Bayesian exploration (TPE) over a dynamically expanding configuration space.

This repository contains the `imabo` package itself, plus every experiment, baseline, and plotting script used to produce the results in the paper.

## Installation

```bash
pip install -e .
```

To also install the experiment dependencies (plotting, async server client, TabPFN, etc.):

```bash
pip install -e ".[experiments]"
```

To also install the development tools (tests, linter):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from imabo import IMABO

search_space = {
    "learning_rate": {"lower": 1e-5, "upper": 1.0, "log": True},
    "n_layers": {"lower": 1, "upper": 8, "int": True},
    "dropout": {"lower": 0.0, "upper": 0.5},
    "optimizer": {"choices": ["adam", "sgd", "adamw"]},
}

optimizer = IMABO(search_space=search_space, seed=42)

for step in range(200):
    config = optimizer.suggest()
    reward = train_and_evaluate(config)  # your evaluation function
    optimizer.observe(reward)

print(f"Best config: {optimizer.best_config}")
```

## Algorithm Overview

IMABO maintains a growing set of configurations M_t and alternates between two oracles:

1. **TPE Oracle** (exploration): proposes new configurations by fitting Parzen estimators on good/bad arms and maximizing $\frac{l(x)}{g(x)}$.
2. **MOSS Oracle** (exploitation): selects the best existing arm using the minimax-optimal anytime MOSS index.

The switching rule is controlled by a parameter beta in (0, 1):

- If $|M_t| < t^{\beta}$ → invoke TPE (explore)
- Otherwise → invoke MOSS (exploit)

This ensures the configuration space grows sublinearly with $O(t^\beta)$, maintaining computational efficiency while preserving near-optimal regret.

## API Reference

| Parameter          | Type       | Default           | Description                            |
| ------------------ | ---------- | ----------------- | -------------------------------------- |
| `search_space`     | `dict`     | required          | Parameter definitions (see below)      |
| `seed`             | `int \| None` | `42`           | Random seed                            |
| `n_startup_trials` | `int`      | `10`              | Random initial configurations          |
| `beta`             | `float`    | `0.8`             | Switching exponent                     |
| `switch_strategy`  | `str`      | `"beta"`          | `"beta"` (sync) or `"delayed"` (async) |
| `n_ei_candidates`  | `int`      | `24`              | EI candidates sampled from l(x)        |
| `gamma_func`       | `callable` | top 30% quantile  | Good/bad split function                |
| `multivariate`     | `bool`     | `True`            | Multivariate Parzen estimation         |
| `use_tpe`          | `bool`     | `True`            | Disable to fall back to random exploration (ablation) |
| `tpe_split_bound`  | `str`      | `"moss"`          | `"moss"` or `"lcb"` index used to rank arms when splitting good/bad |
| `memory`           | `Memory`   | `InMemoryStorage` | Custom storage                         |

### Search Space Format

```python
{
    "param_name": {"lower": float, "upper": float},                    # continuous
    "param_name": {"lower": float, "upper": float, "log": True},      # log-uniform
    "param_name": {"lower": int, "upper": int, "int": True},          # integer
    "param_name": {"choices": ["a", "b", "c"]},                       # categorical
}
```

### Methods

- `optimizer.suggest() -> dict` — Returns the next configuration to evaluate.
- `optimizer.observe(reward: float)` — Reports the reward for the last suggestion.
- `optimizer.best_config -> dict | None` — Best configuration found so far.

### Other Optimizer Variants

- `IMABOTabFM` — TPE oracle replaced by a Google TabFM foundation-model prior.
- `IMABOTabPFN` — TPE oracle replaced by a Prior-Labs TabPFN-3 foundation-model prior (requires the `experiments` extra, which includes `tabpfn`).

## Custom Memory

For production use with persistent storage (database, Redis, etc.), implement the `Memory` interface:

```python
from imabo import Memory, ArmStats, CurrentState

class RedisMemory(Memory):
    def set(self, key, stats): ...
    def get_reward_frequency(self) -> float: ...
    def increment_step_counter(self): ...
    def get_current_state(self) -> CurrentState: ...
    def pull_arm(self, arm_key): ...
    def observe(self, config, reward): ...

optimizer = IMABO(search_space=space, memory=RedisMemory(...))
```

## Reproducing Paper Experiments

Install the experiment dependencies first:

```bash
pip install -e ".[experiments]"
```

Every experiment follows the same pattern: run the script to (re)generate results under `results/` — runs are seed-checkpointed and resumable, so re-running skips completed seeds — then plot from those results (usually the same script, or a companion script in `experiments/utils/plots/`). Pass `--help` to any script for its options; most support `--plot-only` to replot without rerunning.

| Script | What it reproduces |
| --- | --- |
| `experiments.toy_experiment` | IMABO vs. tree baselines (StoSOO, HOO-T, Stroquool) on synthetic toy functions (sin1, garland, rastrigin). |
| `experiments.hpo_experiment` | IMABO vs. tree baselines on real HPO benchmarks (Logistic Regression, SVM via HPOBench). **Requires the HPO benchmark server** (see Docker below). |
| `experiments.rf_arm_distribution_experiment` | Per-iteration arm-choice distribution for every method, including the `IMOSS-TabFM` and `IMOSS-TabPFN` foundation-model oracles, on the RF tabular grid. |
| `experiments.ablation_experiment` | Ablations: (1) TPE oracle impact (`use_tpe=False`) across dimensions, (2) MOSS oracle / `k` impact vs. `OptunaBandit`. |
| `experiments.delayed_feedback_experiment` | Delay-aware switching (`switch_strategy="delayed"`) vs. delay-oblivious IMABO under censored/delayed rewards, on LCBench (YAHPO-Gym) and NAS-Bench-201. Needs one-time setup — see `experiments/benchmarks/delayed/README.md`. |
| `experiments.hotpotqa_experiment` | HotpotQA online-HPO figure; runs every method including the foundation-model oracles. Requires an `OPENROUTER_API_KEY` (in a `.env` file or the environment). |
| `experiments.factored_baseline_experiment` | Hier-MAB (AutoRAG-HP) as a factored baseline on the discrete RF grid — tests whether coordinate-at-a-time credit assignment is competitive. |
| `experiments.coordination_barrier_experiment` | Synthetic counterexamples where reaching the global mode requires moving every coordinate together, illustrating where factored methods like Hier-MAB stall. |
| `experiments.reward_structure_analysis` | Offline diagnostic: additive-variance share and multilinear (Tucker) rank of each RF benchmark's reward tensor, independent of any bandit run. |

Example invocations:

```bash
python -m experiments.toy_experiment
python -m experiments.rf_arm_distribution_experiment                          # all algorithms x all benchmarks
python -m experiments.rf_arm_distribution_experiment --algorithm IMOSS-TabPFN # single method
python -m experiments.hotpotqa_experiment --algorithm IMOSS-TABPFN            # or IMOSS-TABFM
python -m experiments.hotpotqa_experiment --algorithm IMOSS-TABPFN --plot-only
```

### HPO benchmark server (Docker)

`hpo_experiment` (and anything using HPOBench, e.g. Logistic Regression / SVM) needs the bundled HPOBench server running:

```bash
docker compose up hpo-server
```

This builds a container with a Python 3.7 conda env and the vendored `HPOBench/` source, and exposes it on `localhost:8901`. A `dev` service with the same image is also available for interactive use (`docker compose run dev`).

### Baselines implemented (`experiments/baselines/`)

- `stroquool.py` — StoSOO, HOO-T, StroquOOL, Sequool (tree-based continuous bandits) + `TimedOptimizer` wrapper.
- `optuna_bandit.py` — `OptunaBandit`, a k-averaging wrapper around Optuna's TPE sampler.
- `random_search.py` — uniform random search.
- `ucb_air.py` — UCB-AIR (Wang, Audibert & Munos), infinitely-many-armed bandit with the Arm-Increasing Rule.
- `hier_mab.py` — Hier-MAB, the two-level hierarchical bandit from AutoRAG-HP (Fu et al., EMNLP Findings 2024).

## Repository Layout

```
imabo/                              # core IMABO package
  optimizer.py                      # IMABO, FiniteIMABO, IMABOTabFM
  tabpfn_optimizer.py                # IMABOTabPFN
  moss.py                           # MOSS-anytime, UCB, KL-UCB indices
  tpe.py                            # TPE oracle (Parzen estimators, gamma/weight functions)
  memory.py                         # Memory interface, InMemoryStorage, ArmStats/CurrentState
  types.py                          # ArmKey, ArmConfig
experiments/
  toy_experiment.py                  # toy benchmark comparison
  hpo_experiment.py                  # real HPO benchmark comparison (needs Docker server)
  rf_arm_distribution_experiment.py  # arm-choice distribution incl. foundation-model oracles
  ablation_experiment.py             # TPE oracle + k-impact ablations
  delayed_feedback_experiment.py     # delayed/censored reward experiment
  hotpotqa_experiment.py             # HotpotQA online-HPO experiment
  factored_baseline_experiment.py    # Hier-MAB on the discrete RF grid
  coordination_barrier_experiment.py # synthetic factored-method counterexamples
  reward_structure_analysis.py       # offline reward-landscape diagnostics
  baselines/                        # StoSOO/HOO-T/Stroquool, OptunaBandit, Random, UCB-AIR, Hier-MAB
  benchmarks/
    config.py                       # BENCHMARKS dict (param specs, fidelity, metrics)
    hpo_wrapper.py                  # HPOBenchmark client (array_to_config, eval_config, …)
    toys/                           # sin1, garland, rastrigin objective functions
    hpo_bench/                      # HTTP server (Docker) + async client for HPOBench
    delayed/                        # LCBench / NAS-Bench-201 bandits, delay simulator, one-time setup scripts
    hotpotqa/                       # HotpotQA benchmark, embeddings, metrics
  utils/
    stats.py                        # calculate_statistics, CSV save helpers
    plots/                          # one plotting module per experiment, shared plot_configs.py (Wong palette, paper style)
results/                             # generated CSVs and PDFs (git-ignored)
HPOBench/                            # vendored HPOBench source (installed in the Docker image)
Dockerfile
docker-compose.yml
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
