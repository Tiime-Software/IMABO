# IMABO: Infinite Multi-Armed Bandits with Oracles

IMABO is an online hyperparameter optimization algorithm that combines multi-armed bandit exploitation (MOSS) with Bayesian exploration (TPE) over a dynamically expanding configuration space.

## Installation

```bash
pip install -e .
```

To also install experiment dependencies (plots, async server client, etc.):

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

1. **TPE Oracle** (exploration): proposes new configurations by fitting Parzen estimators on good/bad arms and maximizing l(x)/g(x).
2. **MOSS Oracle** (exploitation): selects the best existing arm using the minimax-optimal anytime MOSS index.

The switching rule is controlled by a parameter beta in (0, 1):
- If |M_t| < t^beta → invoke TPE (explore)
- Otherwise → invoke MOSS (exploit)

This ensures the configuration space grows sublinearly with O(t^beta), maintaining computational efficiency while preserving near-optimal regret.

## API Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_space` | `dict` | required | Parameter definitions (see below) |
| `seed` | `int \| None` | `42` | Random seed |
| `n_startup_trials` | `int` | `10` | Random initial configurations |
| `beta` | `float` | `0.8` | Switching exponent |
| `switch_strategy` | `str` | `"beta"` | `"beta"` (sync) or `"delayed"` (async) |
| `n_ei_candidates` | `int` | `24` | EI candidates sampled from l(x) |
| `gamma_func` | `callable` | 30% quantile | Good/bad split function |
| `multivariate` | `bool` | `True` | Multivariate Parzen estimation |
| `memory` | `Memory` | `InMemoryStorage` | Custom storage |

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

The `experiments/` directory contains all scripts used in the paper. Install the experiment dependencies first:

```bash
pip install -e ".[experiments]"
```

### Toy benchmarks

Compares IMABO against StoSOO, HOO-T, and StroquOOL on synthetic functions (Sin1, Garland, Rastrigin 4D) across budgets T ∈ {100, 500, 1000, 5000}:

```bash
python -m experiments.toy_experiment
```

### Real HPO benchmarks

Compares IMABO against StoSOO, HOO-T, and StroquOOL on Logistic Regression and SVM benchmarks from HPOBench across budgets T ∈ {1000, 3000, 5000, 10000}, 20 independent runs each.

Requires the HPO benchmark server running in Docker (see [HPO Benchmark Server](#hpo-benchmark-server) below):

```bash
python -m experiments.hpo_experiment
```

Results are written to `results/` as CSVs and PDF plots.

### Ablation study

Isolates the contribution of each oracle component (TPE oracle impact and MOSS oracle / k-averaging impact):

```bash
python -m experiments.ablation_experiment
```

---

## HPO Benchmark Server

The real HPO experiments require a Docker server that wraps [HPOBench](https://github.com/automl/HPOBench) in a Python 3.7 environment (required by HPOBench's dependencies).

### Build and start

```bash
docker compose up hpo-server
```

The server listens on `http://localhost:8000`. The first call to a benchmark loads it into memory (~30 s for SVM, ~10 s for LR).

### Supported benchmarks

| Key | Model | Dim | Fidelity |
|-----|-------|-----|---------|
| `lr` | Logistic Regression | 2 | iter=1000, subsample=1.0 |
| `svm` | SVM | 2 | subsample=0.5 |
| `rf` | Random Forest | 4 | n\_estimators=100, subsample=0.8 |
| `xgboost` | XGBoost | 4 | n\_estimators=100, subsample=0.8 |
| `histgb` | Hist. Gradient Boosting | 4 | n\_estimators=100, subsample=0.8 |
| `nn` | MLP | 5 | iter=100, subsample=0.8 |
| `pybnn` | PyBNN | 5 | — |

### How it works

The `HPOBench/` directory (included in this repo) contains the full HPOBench source. The `Dockerfile` installs it into a Python 3.7 conda environment (`hpo`). `benchmarks/hpo_bench/server.py` exposes a simple HTTP API; `benchmarks/hpo_wrapper.py` provides the `HPOBenchmark` client used by the experiment scripts.

---

## Repository Layout

```
imabo/               # core IMABO package
experiments/
  toy_experiment.py        # toy benchmark comparison (IMABO vs tree baselines)
  hpo_experiment.py        # real HPO benchmark comparison (IMABO vs tree baselines)
  ablation_experiment.py   # ablation study
  baselines/
    stroquool.py           # StoSOO, HOO-T, StroquOOL, Sequool + TimedOptimizer
    optuna_bandit.py       # OptunaBandit (k-averaging wrapper for Optuna TPE)
  utils/
    stats.py               # calculate_statistics, CSV helpers
    plot_functions.py      # publication-quality plots (Wong colorblind palette)
benchmarks/
  config.py                # BENCHMARKS dict (param specs, fidelity, metrics)
  hpo_wrapper.py           # HPOBenchmark client (array_to_config, eval_config, …)
  hpo_bench/
    server.py              # HTTP server (runs inside Docker)
    client.py              # async HTTP client (api_call, start/stop server)
HPOBench/                  # HPOBench source (installed in Docker image)
Dockerfile
docker-compose.yml
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
