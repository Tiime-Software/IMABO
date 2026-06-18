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

1. **TPE Oracle** (exploration): proposes new configurations by fitting Parzen estimators on good/bad arms and maximizing $\frac{l(x)}/{g(x)}$.
2. **MOSS Oracle** (exploitation): selects the best existing arm using the minimax-optimal anytime MOSS index.

The switching rule is controlled by a parameter beta in (0, 1):

- If $|M_t| < t^{\beta}$ → invoke TPE (explore)
- Otherwise → invoke MOSS (exploit)

This ensures the configuration space grows sublinearly with $O(t^\beta)$, maintaining computational efficiency while preserving near-optimal regret.

## API Reference


| Parameter          | Type       | Default           | Description                            |
| ------------------ | ---------- | ----------------- | -------------------------------------- |
| `search_space`     | `dict`     | required          | Parameter definitions (see below)      |
| `seed`             | `int       | None`             | `42`                                   |
| `n_startup_trials` | `int`      | `10`              | Random initial configurations          |
| `beta`             | `float`    | `0.8`             | Switching exponent                     |
| `switch_strategy`  | `str`      | `"beta"`          | `"beta"` (sync) or `"delayed"` (async) |
| `n_ei_candidates`  | `int`      | `24`              | EI candidates sampled from l(x)        |
| `gamma_func`       | `callable` | 30% quantile      | Good/bad split function                |
| `multivariate`     | `bool`     | `True`            | Multivariate Parzen estimation         |
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

Each experiment follows the same two-step workflow: **run** the experiment to generate CSVs under `results/`, then **plot** from those CSVs.

---

## Repository Layout

```
imabo/                         # core IMABO package
experiments/
  toy_experiment.py            # toy benchmark comparison (IMABO vs tree baselines)
  hpo_experiment.py            # real HPO benchmark comparison (IMABO vs tree baselines)
  ablation_experiment.py       # ablation study (TPE oracle + k impact)
  baselines/
    stroquool.py               # StoSOO, HOO-T, StroquOOL, Sequool + TimedOptimizer
    optuna_bandit.py           # OptunaBandit (k-averaging wrapper for Optuna TPE)
  benchmarks/
    config.py                  # BENCHMARKS dict (param specs, fidelity, metrics)
    hpo_wrapper.py             # HPOBenchmark client (array_to_config, eval_config, …)
    toys/
      toy_functions.py         # sin1, garland, rastrigin objective functions
    hpo_bench/
      server.py                # HTTP server (runs inside Docker)
      client.py                # async HTTP client
  utils/
    stats.py                   # calculate_statistics, CSV save helpers
    plot_configs.py            # Wong colorblind palette, set_research_style()
    toy_plot.py                # plots for toy_experiment results
    hpo_plot.py                # plots for hpo_experiment results
    ablation_plot.py           # plots for ablation_experiment results
results/                       # generated CSVs and PDFs (git-ignored)
HPOBench/                      # HPOBench source (installed in Docker image)
Dockerfile
docker-compose.yml
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

