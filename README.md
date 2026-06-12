# IMABO: Infinite Multi-Armed Bandits with Oracles

IMABO is an online hyperparameter optimization algorithm that combines multi-armed bandit exploitation (MOSS) with Bayesian exploration (TPE) over a dynamically expanding configuration space.

## Installation

```bash
pip install -e .
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

## Tests

```bash
pip install -e ".[dev]"
pytest
```
