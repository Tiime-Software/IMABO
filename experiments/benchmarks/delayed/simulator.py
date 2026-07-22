"""Delayed/censored-feedback simulation loop for IMABO-family optimizers.

Feedback for a suggested configuration doesn't arrive immediately: it's
scheduled some number of steps in the future (drawn from a delay model, e.g.
:class:`~experiments.benchmarks.delayed.delay_model.GleipnirDelayModel`) and
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

    for step in range(n_iterations):
        x = optimizer.suggest()
        y = bench(x, noise=True)
        regrets.append(bench.regret(x))

        delay = delay_model.sample_delay_steps(rng)
        # A `None` delay (Bernoulli-censored -- never going to arrive) is
        # scheduled just past the patience window rather than skipped
        # outright, so it flows through the same expiry path below as a
        # "arrived too late" pull and is counted in `num_censored_this_step`
        # -- one accounting path for both flavors of censoring.
        arrival_step = step + delay if delay is not None else step + patience_steps + 1
        heapq.heappush(heap, PendingObservation(arrival_step, x, y, step))

        arrived = 0
        while heap and heap[0].arrival_step <= step:
            pending = heapq.heappop(heap)
            optimizer.memory.observe(pending.config, pending.reward)
            arrived += 1

        censored = 0
        if heap:
            survivors = [
                pending
                for pending in heap
                if step - pending.generated_step <= patience_steps
            ]
            censored = len(heap) - len(survivors)
            if censored:
                heap = survivors
                heapq.heapify(heap)

        num_pending.append(len(heap))
        num_arrived.append(arrived)
        num_censored.append(censored)

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

    for _ in range(n_iterations):
        x = optimizer.suggest()
        y = bench(x, noise=True)
        optimizer.observe(y)
        regrets.append(bench.regret(x))

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
        "best_config": optimizer.best_config,
    }
