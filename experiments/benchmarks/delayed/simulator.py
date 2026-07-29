"""Delayed/censored-feedback simulation loop for IMABO-family optimizers.

Feedback for a suggested configuration doesn't arrive immediately: it's
scheduled some number of steps in the future (drawn from a delay model, e.g.
:class:`~experiments.benchmarks.delayed.delay_model.DelayModel`) and
delivered out of order via a min-heap, mirroring how real asynchronous
feedback (e.g. a user vote on a generated item) actually reaches the
optimizer. A pull not observed within ``patience_steps`` of being suggested
is dropped without ever being observed -- one rule covering both "arrived
too late" and "was never going to arrive" (censored).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.benchmarks.delayed.delay_model import DelayModel
from imabo.types import ArmConfig


def _active_set_size(optimizer: Any) -> int:
    """Number of distinct arms `optimizer` has drawn so far -- the realized
    |M_t| the switching rule's admission ceiling (`len(state.arms) <
    effective_t**beta`, see `imabo/optimizer.py`) is compared against.

    Duck-typed: IMABO-family optimizers expose `.memory.memory` (an
    `InMemoryStorage` dict, see `imabo/memory.py`), but UCB-AIR
    (`experiments/baselines/ucb_air.py`) has no `.memory` at all -- it keeps
    its own active set in `.arms`.
    """
    memory = getattr(optimizer, "memory", None)
    if memory is not None:
        return len(memory.memory)
    return len(optimizer.arms)


def patience_for_quantile(
    bench: Any,
    delay_model: Any,
    q: float = 0.95,
    n_samples: int = 20000,
    seed: int = 0,
) -> int:
    """Per-benchmark patience window = the q-th percentile of *this* benchmark's
    own positive (would-arrive) delay distribution, in simulator steps.

    A fixed `patience_steps` (e.g. 72) means completely different things across
    benchmarks whose delay time-scales differ: at 72 it cuts ~0.5% of observed
    feedback on LCBench, ~5.6% on the RF log-normal, and ~28% on Criteo's
    heavy-tailed real delays. Setting patience to a shared *quantile* instead
    makes the patience-induced censoring identical (= 1 - q) across benchmarks,
    so cross-benchmark comparisons isolate the Bernoulli (never-arrives)
    censoring rather than confounding it with an arbitrary shared cutoff.

    The delay distribution is sampled exactly as the simulator generates delays
    (`delay_model.sample_delay_steps(rng, expected_steps=...)`), so this is
    consistent with whatever model/benchmark pairing is in use:
      - RF (log-normal DelayModel): expected_steps is None; the quantile is of
        the fitted log-normal.
      - Criteo (empirical CriteoDelayModel): quantile of the real delay array.
      - LCBench (RuntimeDelayModel): delay = config runtime x jitter, so configs
        are sampled from the benchmark and their per-config delays are pooled.
    Only positive (non-None) delays define patience -- a `None` is a
    Bernoulli-censored pull that never arrives regardless of the window, so it
    must not shift the cutoff.

    Compute this ONCE per benchmark from the base (delay_scale=1.0) delay model
    and hold it fixed across a severity sweep, so a larger delay_scale genuinely
    pushes more pulls past a fixed window (the effect the sweep measures) rather
    than moving the goalposts with it.
    """
    rng = np.random.default_rng(seed)
    has_runtime = hasattr(bench, "expected_runtime_steps")

    configs = None
    if has_runtime:
        # Sample configs from the benchmark's own space to get its per-config
        # runtime (hence delay) distribution. Prefer the benchmark's sampler;
        # fall back to a uniform draw over the declared search space.
        if hasattr(bench, "_sample_opt_configs"):
            configs = bench._sample_opt_configs(n_samples)
        else:
            configs = [
                _uniform_config(bench.get_search_space(), rng) for _ in range(n_samples)
            ]

    # reward=1 requests the positive-signal delay (matters only for the
    # conversion-signal CriteoDelayModel, whose patience-relevant distribution
    # is the conversion delays; value-agnostic models ignore it). We want the
    # distribution of delays for pulls that DO produce a delayed signal, which
    # is what patience trims.
    delays = []
    for i in range(n_samples):
        es = bench.expected_runtime_steps(configs[i]) if has_runtime else None
        d = delay_model.sample_delay_steps(rng, expected_steps=es, reward=1)
        if d is not None:
            delays.append(d)

    if not delays:
        raise ValueError(
            "No positive delays sampled -- cannot set a patience quantile "
            "(is feedback_freq 0?). Fall back to an explicit patience_steps."
        )
    return int(np.ceil(np.quantile(delays, q)))


def _uniform_config(search_space: dict, rng: np.random.Generator) -> dict:
    """Uniform draw over an IMABO-format search space (choices / {lower,upper,
    log,int}) -- fallback config sampler for the patience calc."""
    cfg = {}
    for name, spec in search_space.items():
        if "choices" in spec:
            cfg[name] = spec["choices"][rng.integers(len(spec["choices"]))]
        else:
            lo, hi = spec["lower"], spec["upper"]
            if spec.get("log", False):
                v = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                v = float(rng.uniform(lo, hi))
            cfg[name] = int(round(v)) if spec.get("int", False) else v
    return cfg


@dataclass(order=True)
class PendingObservation:
    """Heap item: a suggested config awaiting delayed delivery of its reward."""

    arrival_step: int
    config: ArmConfig = field(compare=False)
    reward: float = field(compare=False)
    generated_step: int = field(compare=False)


def run_delayed(
    optimizer: Any,
    bench: Any,
    n_iterations: int,
    seed: int = 42,
    delay_model: DelayModel | None = None,
    patience_steps: int = 72,
) -> dict:
    """Run `optimizer` against `bench` for `n_iterations` steps, with reward
    feedback delayed/censored per `delay_model`.

    `optimizer` must be an IMABO-family instance: delivery of a delayed
    reward is done via `optimizer.memory.observe(config, reward)` directly
    (a public method on the `Memory` ABC, `imabo/memory.py`), since it can
    record a reward against an arbitrary earlier config -- unlike
    `optimizer.observe(reward)`, which only supports the most recently
    suggested one. `bench` follows this repo's benchmark duck-type:
    `bench(x, noise=True) -> float`, `bench.regret(x) -> float`.
    """
    delay_model = delay_model or DelayModel()
    rng = np.random.default_rng(seed)

    heap: list[PendingObservation] = []
    regrets: list[float] = []
    simple_regret_trace: list[float] = []
    num_pending: list[int] = []
    num_arrived: list[int] = []
    num_censored: list[int] = []
    num_active: list[int] = []
    rewards_delivered: list[list[float]] = []  # raw reward values delivered per step

    for step in range(n_iterations):
        x = optimizer.suggest()
        y = bench(x, noise=True)
        regrets.append(bench.regret(x))

        # `expected_steps` lets a config-dependent delay model (e.g.
        # RuntimeDelayModel, whose delay IS the config's predicted training
        # time) derive this pull's delay from `x`. Delay models that ignore it
        # (the log-normal DelayModel) accept and discard the kwarg, and
        # benchmarks without a runtime estimate (RFTabularFiniteBenchmark) pass
        # None -- so both the runtime-driven and the injected-delay settings
        # flow through this one loop unchanged.
        expected_steps = (
            bench.expected_runtime_steps(x)
            if hasattr(bench, "expected_runtime_steps")
            else None
        )

        delay: int | None = delay_model.sample_delay_steps(
            rng, expected_steps=expected_steps, reward=y, patience_steps=patience_steps
        )
        arrival_step: int = (
            step + delay if delay is not None else step + patience_steps + 1
        )
        heapq.heappush(heap, PendingObservation(arrival_step, x, y, step))

        # Expiry (censoring) is checked BEFORE arrival: a pull that has sat in
        # the queue longer than the patience window has exceeded the deadline
        # and must NOT be delivered with its true reward on its scheduled
        # arrival step. A Bernoulli-censored pull (delay=None) is parked at
        # generated_step + patience_steps + 1 so that by the step it would
        # "arrive" it has already exceeded patience and is caught here first
        # (checking arrival first was an off-by-one that delivered censored
        # pulls late instead of censoring them).
        censored = 0
        if heap:
            survivors = []
            for pending in heap:
                if step - pending.generated_step <= patience_steps:
                    survivors.append(pending)
                else:
                    censored += 1
            if censored:
                heap = survivors
                heapq.heapify(heap)

        arrived = 0
        step_rewards: list[float] = []
        while heap and heap[0].arrival_step <= step:
            pending = heapq.heappop(heap)
            optimizer.memory.observe(pending.config, pending.reward)
            step_rewards.append(pending.reward)
            arrived += 1

        num_pending.append(len(heap))
        num_arrived.append(arrived)
        num_censored.append(censored)
        # Post-delivery, like num_pending: pull_arm() inserts into memory
        # synchronously inside suggest() (imabo/optimizer.py), so this step's
        # newly suggested arm is already counted -- no lag.
        num_active.append(_active_set_size(optimizer))
        rewards_delivered.append(step_rewards)

        incumbent = optimizer.best_config
        simple_regret_trace.append(
            bench.regret(incumbent) if incumbent is not None else bench.max_value
        )

    return {
        "regrets": regrets,
        "simple_regret_trace": simple_regret_trace,
        "num_pending": num_pending,
        "num_arrived_this_step": num_arrived,
        "num_censored_this_step": num_censored,
        "num_active": num_active,
        "rewards_delivered_this_step": rewards_delivered,
        "best_config": optimizer.best_config,
    }


def run_baseline(optimizer: Any, bench: Any, n_iterations: int, seed: int = 42) -> dict:
    """Synchronous no-delay loop: suggest -> evaluate -> observe immediately.

    Used both as the no-delay skyline and for algorithms (e.g. RandomSearch)
    whose regret trace is delay-invariant by construction: a context-free
    uniform sampler's next pick never depends on when past feedback arrived,
    so running it through the delayed heap machinery above would produce the
    identical `regrets` trace at extra cost (see
    experiments/delayed_feedback_experiment.py, `Algorithm.RANDOM`).
    """
    regrets: list[float] = []
    simple_regret_trace: list[float] = []
    num_active: list[int] = []
    rewards_delivered: list[list[float]] = []

    for _ in range(n_iterations):
        x = optimizer.suggest()
        y = bench(x, noise=True)
        optimizer.observe(y)
        regrets.append(bench.regret(x))
        rewards_delivered.append([y])
        # Not hard-coded from t**beta: the skyline's/UCB-AIR's realized active
        # set can lag the admission ceiling too (e.g. before enough arms have
        # been drawn), so we log the actual count here as well.
        num_active.append(_active_set_size(optimizer))

        incumbent = optimizer.best_config
        simple_regret_trace.append(
            bench.regret(incumbent) if incumbent is not None else bench.max_value
        )

    return {
        "regrets": regrets,
        "simple_regret_trace": simple_regret_trace,
        "num_pending": [0] * n_iterations,
        "num_arrived_this_step": [1] * n_iterations,
        "num_censored_this_step": [0] * n_iterations,
        "num_active": num_active,
        "rewards_delivered_this_step": rewards_delivered,
        "best_config": optimizer.best_config,
    }
