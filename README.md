# IMABO: Infinite Multi-Armed Bandits with Oracles

IMABO is a Python framework for online hyperparameter optimization: choosing the configuration to serve on each request, and learning from the reward it returns. It combines any bandit policy for choosing among the configurations already tried with any oracle for proposing new ones.

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

optimizer = IMABO(search_space, seed=42)          # IMOSS-TPE

for step in range(200):
    config = optimizer.suggest()
    reward = train_and_evaluate(config)  # your evaluation function
    optimizer.observe(reward)

print(f"Best config: {optimizer.best_config}")
```

By default, IMABO uses `IMOSS(beta=0.5)` paired with `TPEOracle()`, which is IMOSS-TPE.

```python
from imabo import IMABO, IMOSS, RandomOracle, TPEOracle

IMABO(search_space, seed=42)                                  # IMOSS-TPE (the default)
IMABO(search_space, IMOSS(beta=0.5), TPEOracle(), seed=42)     # same
```

The four proposed instantiations from the paper are also available under their own names:

```python
from imabo import IMOSSRandom, IMOSSTPE, IMOSSMutateKLTPE, IMOSSTabPFN, load_tabpfn

IMOSSRandom(search_space, beta=0.5, seed=42)                          # IMOSS-Random
IMOSSTPE(search_space, beta=0.5, seed=42)                             # IMOSS-TPE
IMOSSMutateKLTPE(search_space, beta=0.5, seed=42)                     # IMOSS-mutate-KLxTPE
IMOSSTabPFN(search_space, beta=0.5, seed=42, model=load_tabpfn())     # IMOSS-TabPFN
```

## Algorithm Overview

IMABO maintains a growing active set of configurations M_t. Each round it makes one
binary choice:

```
for t = 1, 2, ...
    if policy.expand(): x = oracle.suggest(); admit x
    else:               x = policy.select()
    serve x, observe its reward, update x's statistics
```

The policy decides both when to admit a new arm and which known arm to serve
otherwise. The oracle only decides what a new arm should be.

`IMOSS` is the policy from the paper: it admits a new arm while $|M_t| < t^{\beta}$ and
otherwise serves the arm maximizing the anytime MOSS index. The active set therefore
grows as $O(t^\beta)$, which keeps the per-round cost bounded while preserving a
sublinear cumulative quantile-regret guarantee.

## API Reference

### `IMABO(search_space, policy, oracle, *, seed=None, memory=None)`

| Parameter      | Type                | Default           | Description                              |
| -------------- | ------------------- | ----------------- | ---------------------------------------- |
| `search_space` | `SearchSpace \| dict \| callable` | required | Parameter definitions (see below) |
| `policy`       | `AllocationPolicy`  | `IMOSS()`         | Where every pull goes                    |
| `oracle`       | `Oracle`            | `TPEOracle()`     | What to admit when the policy asks        |
| `seed`         | `int \| None`       | `None`            | Seeds one RNG shared by all three        |
| `memory`       | `Memory`            | `InMemoryStorage` | Where the active set lives               |

### `IMOSS(beta=0.5, alpha=0.1, n_warmup=10, max_pending=20, min_rewards=1, delayed=False)`

| Parameter     | Default | Description                                                             |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `beta`        | `0.5`   | Active-set growth exponent, in (0, 1)                                   |
| `alpha`       | `0.1`   | Confidence parameter scaling the MOSS exploration bonus                 |
| `n_warmup`    | `10`    | Configurations drawn from P0 before the index takes over (`Ns`)         |
| `max_pending` | `20`    | Re-serves allowed for an arm that has never returned a reward           |
| `min_rewards` | `1`     | Rewards an arm needs before `best_arm` will report it                   |
| `delayed`     | `False` | Use the delay-aware rules of Appendix C.1                               |

### `TPEOracle(n_candidates=24, gamma_func=None, weights_func=None, prior_weight=1.0, multivariate=True, split_index="policy", categorical_distance_func=None)`

| Parameter       | Default    | Description                                                          |
| --------------- | ---------- | -------------------------------------------------------------------- |
| `n_candidates`  | `24`       | Candidates drawn from l per call, ranked by l/g                      |
| `gamma_func`    | top 30%    | Good/bad split size                                                  |
| `multivariate`  | `True`     | One joint Parzen density rather than one per parameter               |
| `split_index`   | `"policy"` | Rank the split by the policy's index, or `"lcb"` pessimistically     |

Other policies and oracles: `BudgetedUCB` (fixed horizon, not in the paper),
`RandomOracle`, `MutateKLTPEOracle`, `TabPFNOracle`, `TabFMOracle`. Each carries its
own documented arguments.

### Search Space Format

A space can be given two ways: a dict or a suggestion function.

A dict, one entry per parameter. Any mix of the four kinds:

```python
{
    "param_name": {"lower": float, "upper": float},                    # continuous
    "param_name": {"lower": float, "upper": float, "log": True},      # log-uniform
    "param_name": {"lower": int, "upper": int, "int": True},          # integer
    "param_name": {"choices": ["a", "b", "c"]},                       # categorical
}
```

```python
from imabo import IMABO

SPACES = {
    "grid": {f"x{i}": {"choices": list(range(6))} for i in range(4)},
    "mixed": {
        "num_layers": {"choices": [1, 2, 3, 4, 5]},
        "batch_size": {"lower": 16, "upper": 512, "int": True, "log": True},
        "learning_rate": {"lower": 1e-4, "upper": 1e-1, "log": True},
        "momentum": {"lower": 0.1, "upper": 0.99},
        "max_dropout": {"lower": 0.0, "upper": 1.0},
    },
}

optimizer = IMABO(SPACES["mixed"], seed=42)
```

A suggestion function, which asks a `Trial` for its parameters instead of declaring them as
data:

```python
def space(trial):
    trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    trial.suggest_int("n_layers", 1, 8)
    trial.suggest_float("dropout", 0.0, 0.5)
    trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])

optimizer = IMABO(space, seed=42)
```

Each `suggest_*` call does two things: it declares a parameter — name, type, bounds — and
it returns a value drawn for it.

### Methods

- `optimizer.suggest() -> dict` — Returns the next configuration to serve.
- `optimizer.observe(reward, config=None)` — Reports a reward. Defaults to the last
  suggestion; pass `config=` when feedback arrives out of order.
- `optimizer.best_config -> dict | None` — The configuration the policy reports.
- `optimizer.run(objective, n_rounds) -> dict` — Drives the loop for offline use.
- `optimizer.state -> CurrentState` — Snapshot of the active set.
- `optimizer.propose() -> dict` — What the oracle would admit right now, for
  diagnostics. Advances the oracle's state, so call it on a `copy.deepcopy`.

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

optimizer = IMABO(space, memory=RedisMemory(...))
```

The oracles keep nothing of their own -- whatever they accumulate lives in the memory --
so a restarted process resumes as soon as your `Memory` reloads. Nothing has to be saved
around a crash.

## Writing Your Own Instantiation

An oracle is one method. `suggest` is handed everything any oracle could need -- the
active set, the arms that have returned a reward, and how the policy ranks an arm -- and
returns a configuration to admit.

```python
from imabo import IMABO, IMOSS, Oracle

class NearestGridOracle(Oracle):
    """Admit the first point of a coarse grid that is not already an arm."""

    def suggest(self, state, rewarded_arms, score):
        for value in (0.25, 0.5, 0.75):
            config = {name: value for name in self.space.names}
            if self.space.encode(config) not in state.arms:
                return config
        return self.space.sample(self.rng)          # P0, the baseline distribution

optimizer = IMABO(search_space, IMOSS(beta=0.5), NearestGridOracle(), seed=0)
```

A policy is two: when to admit, and which known arm to serve.

```python
from imabo import AllocationPolicy, ArmStats

class RoundRobin(AllocationPolicy):
    def setup(self, space, rng, memory):
        super().setup(space, rng, memory)
        for _ in range(5):                          # an initial design, drawn from P0
            memory.set(space.encode(space.sample(rng)), ArmStats())

    def expand(self, state, rewarded_arms):
        return len(state.arms) < 20

    def select(self, state, rewarded_arms):
        return rewarded_arms[state.nb_steps % len(rewarded_arms)][0]
```

Both are bound to the run by `setup`, which gives them the search space and the run's
shared RNG; neither may draw from it during `setup` except to seed an initial design.
Override `AllocationPolicy.score` to change how oracles rank the active set, and
`AllocationPolicy.best_arm` to change what the optimizer reports.

## Baselines

The baselines the paper compares against ship with the library and drive the same loop as
`IMABO` — `suggest()`, then `observe(reward)`, with `best_config` for the configuration to
report:

```python
from imabo import IMABO, QRM2, UCBAIR, RandomSearch

for optimizer in (IMABO(space), QRM2(space), UCBAIR(space), RandomSearch(space)):
    for _ in range(1000):
        config = optimizer.suggest()
        optimizer.observe(serve(config))
    print(optimizer.best_config)
```

| Baseline | What it is |
|---|---|
| `RandomSearch` | Uniform draws from `P0`. |
| `QRM2` | QRM2 (Roy Chaudhuri & Kalyanakrishnan): MOSS on a growing pool, restarted on a doubling schedule. |
| `UCBAIR`, `MOSSAIR` | UCB-AIR (Wang, Audibert & Munos), with the Arm-Increasing Rule; `MOSSAIR` swaps in the MOSS index. |
| `HierMAB` | Hier-MAB, the two-level hierarchical bandit of AutoRAG-HP (Fu et al., EMNLP Findings 2024). |
| `OptunaBandit` | A k-averaging wrapper around Optuna's TPE sampler. |
| `stroquool`, `stosoo`, `hoo_t` | The tree-search bandits, as generators over the unit cube; drive them through `TimedOptimizer`. |

The tree-search algorithms are exported as the generators they have always been —
`stroquool`, `stosoo`, `hoo_t` — plus the `TimedOptimizer` wrapper. They work on
`[0, 1]**d` rather than on a search space, so mapping their coordinates to configurations
is the caller's job, exactly as the paper's experiments do it:

```python
from imabo import TimedOptimizer, stosoo

optimizer = TimedOptimizer(stosoo, budget, dim)
while not optimizer.done:
    x = optimizer.suggest()                  # a point of [0, 1]**dim
    optimizer.observe(x, serve(to_config(x)))
```

## Reproducing Paper Experiments

Install the experiment dependencies first:

```bash
pip install -e ".[experiments]"
```

Every experiment follows the same pattern: run the script to (re)generate results under `results/` — runs are seed-checkpointed and resumable, so re-running skips completed seeds — then plot from those results (usually the same script, or a companion script in `experiments/utils/plots/`).

Command-line options are not uniform across the scripts:

| Script | `--help` | plots directly | notes |
| --- | --- | --- | --- |
| `coordination_barrier_experiment` | yes | `--plot` | `--n-seeds`, `--n-iterations`, `--n-jobs`, `--quick` |
| `factored_baseline_experiment` | yes | `--plot` | |
| `rf_arm_distribution_experiment` | yes | `--plot` | `--plot-only`, `--algorithm` |
| `hotpotqa_experiment` | yes | `--plot` | `--plot-only`, `--algorithm` |
| `reward_structure_analysis` | yes | `--plot` | |
| `toy_experiment` | no | via `utils/plots/` | no options; edit the constants at the top |
| `ablation_experiment` | no | via `utils/plots/` | idem |
| `delayed_feedback_experiment` | no | via `utils/plots/` | idem |
| `hpo_experiment` | no | via `utils/plots/` | idem |

`coordination_barrier_experiment --plot` reproduces Figure 4 (the two-dimensional Gaussian
counter-example) and is the only paper figure whose experiment needs no external data at
all: the landscapes are synthetic. Note that `--quick` shortens the run but does not skip
`imoss_tabpfn_tuned`, so it still loads TabPFN.

| Script | What it reproduces |
| --- | --- |
| `experiments.toy_experiment` | IMABO vs. tree baselines (StoSOO, HOO-T, Stroquool) on synthetic toy functions (sin1, garland, rastrigin). |
| `experiments.hpo_experiment` | IMABO vs. tree baselines on real HPO benchmarks (Logistic Regression, SVM via HPOBench). **Requires the HPO benchmark server** (see Docker below). |
| `experiments.rf_arm_distribution_experiment` | Per-iteration arm-choice distribution for every method, including the `IMOSS-TabFM` and `IMOSS-TabPFN` foundation-model oracles, on the RF tabular grid. |
| `experiments.ablation_experiment` | Ablations: (1) TPE oracle impact (`IMOSSTPE` vs `IMOSSRandom`) across dimensions, (2) MOSS oracle / `k` impact vs. `OptunaBandit`. |
| `experiments.delayed_feedback_experiment` | Delay-aware switching (`IMOSS(delayed=True)`) vs. delay-oblivious IMOSS under censored/delayed rewards, on LCBench (YAHPO-Gym) and NAS-Bench-201. Needs one-time setup — see `experiments/benchmarks/delayed/README.md`. |
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


## Tests

```bash
pip install -e ".[dev]"
pytest
```
