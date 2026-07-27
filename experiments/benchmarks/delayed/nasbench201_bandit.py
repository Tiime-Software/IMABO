"""Structured finite bandit built from the NAS-Bench-201 tabular benchmark.

Ground truth is NAS-Bench-201 / NATS-Bench topology search space (Dong & Yang
2020, "NAS-Bench-201", arXiv:2001.00326; Dong et al. 2021, "NATS-Bench",
arXiv:2009.00437): a tabular neural-architecture-search benchmark that records
the trained accuracy AND the real training time of every architecture in a
fixed cell space, on cifar10 / cifar100 / ImageNet16-120.

Why this benchmark for the *delayed* experiment. Like LCBench, the feedback
delay here is not injected -- it *is* the architecture's real training time
(a bigger/slower cell reports its validation accuracy later; a crashed/preempted
job never reports). This is the runtime-as-delay conception the asynchronous-HPO
literature uses (Watanabe et al. 2024, "Fast Benchmarking of Asynchronous
Multi-Fidelity Optimization on Zero-Cost Benchmarks", arXiv:2403.01888), and the
same one ``lcbench_bandit.py`` uses -- so NAS-Bench-201 drops into the existing
delayed pipeline with the same ``RuntimeDelayModel`` and no new censoring model.

Why it suits IMABO where a flat conversion-log arm set does not. The space is a
cell of **6 edges, each assigned one of 5 operations** (none, skip_connect,
nor_conv_1x1, nor_conv_3x3, avg_pool_3x3), i.e. 5**6 = 15625 architectures.
Although that is a large arm count, it is *structured*: IMABO exposes the 6 edges
as 6 categorical axes and its TPE oracle generalizes over good-vs-bad operations
per edge, concentrating the budget on promising cells rather than enumerating all
15625 -- the structured many-arm regime IMABO targets. (Contrast an unstructured
segment-id arm set, where TPE has no geometry to exploit.)

Reward semantics match ``lcbench_bandit.py`` / ``rf_tabular_bandit.py`` so the
benchmarks are interchangeable in the delayed simulator: the tabular validation
accuracy (0--100) becomes a Bernoulli success probability in [0, 1], and pulling
an arm draws one Bernoulli sample. Because the space is finite and tabular, the
reference optimum p* is the exact maximum over all architectures (enumerated
once and cached), not a sampled estimate.

Setup (one-time): download the NATS-Bench topology-search-space archive and run
``experiments/benchmarks/delayed/setup_nasbench201.py`` (mirrors setup_lcbench).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REF_CACHE_DIR = Path(__file__).parent / "assets"
_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# The 5 candidate operations on each of the 6 cell edges (NAS-Bench-201 order).
NB201_OPS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
_N_EDGES = 6

# Datasets available in NAS-Bench-201. Distinct from the OpenML tasks used by
# the RF/LCBench benchmarks (image classification, not tabular).
NB201_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")

# Rescale real training seconds to simulator "steps" so the median architecture's
# delay is this many steps (same knob as LCBench). Tunable per experiment.
_TARGET_MEDIAN_DELAY_STEPS = 6.0

# NATS-Bench full-training budget: "200" epochs for the topology search space.
_HP = "200"

# NATS-Bench quirk: plain "cifar10" only exposes train-/test-accuracy (it's the
# final full-train-data regime); a genuine held-out validation accuracy for
# CIFAR-10 requires querying the separate "cifar10-valid" split instead.
# cifar100/ImageNet16-120 already expose "valid-accuracy" under their own
# dataset name (they use a proper train/valid/test split), so this is a no-op
# for them. `self.instance` itself is left alone (filenames, search space,
# `bm_id`) -- only the string passed to `get_more_info` changes.
_QUERY_DATASET = {
    "cifar10": "cifar10-valid",
    "cifar100": "cifar100",
    "ImageNet16-120": "ImageNet16-120",
}


def arch_str_from_ops(ops: list[str]) -> str:
    """Build the NAS-Bench-201 architecture string from 6 edge operations.

    Cell wiring (node->node): edges are ordered
        (1<-0), (2<-0), (2<-1), (3<-0), (3<-1), (3<-2)
    giving the canonical string
        |op0~0|+|op1~0|op2~1|+|op3~0|op4~1|op5~2|
    """
    o = ops
    return (
        f"|{o[0]}~0|+"
        f"|{o[1]}~0|{o[2]}~1|+"
        f"|{o[3]}~0|{o[4]}~1|{o[5]}~2|"
    )


class NASBench201Benchmark:
    """NAS-Bench-201 as a structured finite bandit with runtime-derived delay.

    Duck-type compatible with ``RFTabularFiniteBenchmark`` /
    ``LCBenchMixedBenchmark`` (the delayed simulator needs
    ``bench(x, noise=True) -> float``, ``bench.regret(x) -> float``,
    ``bench.get_search_space()``, ``bench.max_value``, ``bench.reset_noise(seed)``),
    plus ``expected_runtime_steps(x)`` consumed by ``run_delayed`` when a
    ``RuntimeDelayModel`` is used.
    """

    def __init__(
        self,
        instance: str = "cifar100",
        seed: int = 0,
        nats_path: str | None = None,
        target_median_delay_steps: float = _TARGET_MEDIAN_DELAY_STEPS,
    ):
        if instance not in NB201_DATASETS:
            raise ValueError(
                f"instance {instance!r} not in NAS-Bench-201 datasets {NB201_DATASETS}"
            )
        self.instance = str(instance)
        self.bm_id = self.instance  # filename tag
        self.rng = np.random.default_rng(seed)

        # Lazily import nats_bench so importing this module never requires it;
        # only constructing the benchmark does (matches lcbench_bandit).
        from nats_bench import create

        path = nats_path or str(_REF_CACHE_DIR / "NATS-tss-v1_0-3ffb9-simple")
        self._api = create(path, "tss", fast_mode=True, verbose=False)

        # One categorical axis per edge; choices are the 5 operations.
        self.param_names = [f"edge_{i}" for i in range(_N_EDGES)]
        self._search_space = {
            name: {"choices": list(NB201_OPS)} for name in self.param_names
        }

        self._mean_cache: dict[tuple, float] = {}
        self._runtime_cache: dict[tuple, float] = {}

        self.max_value, self._median_runtime_seconds = self._reference_stats()
        self._time_unit = max(
            1e-9, self._median_runtime_seconds / target_median_delay_steps
        )

    # ------------------------------------------------------------------ space

    def get_search_space(self) -> dict[str, dict]:
        """6 categorical edge axes over the 5 NAS-Bench-201 operations."""
        return self._search_space

    def _ops(self, x: dict) -> list[str]:
        return [x[f"edge_{i}"] for i in range(_N_EDGES)]

    def _key(self, x: dict) -> tuple:
        return tuple(self._ops(x))

    # ------------------------------------------------------------- tabular query

    def _query(self, x: dict) -> tuple[float, float]:
        """Return (success_prob in [0,1], training_time_seconds) for config ``x``,
        memoized. Success prob = validation accuracy / 100."""
        key = self._key(x)
        if key in self._mean_cache:
            return self._mean_cache[key], self._runtime_cache[key]
        arch = arch_str_from_ops(list(key))
        idx = self._api.query_index_by_arch(arch)
        info = self._api.get_more_info(
            idx, _QUERY_DATASET[self.instance], hp=_HP, is_random=False
        )
        acc = float(info.get("valid-accuracy", info.get("valtest-accuracy")))
        prob = min(1.0, max(0.0, acc / 100.0))
        # total training time (seconds) -- the delay a real job would incur.
        runtime = float(
            info.get("train-all-time", info.get("train-per-time", 0.0))
        )
        self._mean_cache[key] = prob
        self._runtime_cache[key] = runtime
        return prob, runtime

    def mean_reward(self, x: dict) -> float:
        return self._query(x)[0]

    def __call__(self, x: dict, noise: bool = True) -> float:
        p = self.mean_reward(x)
        if not noise:
            return p
        return float(self.rng.binomial(1, p))

    def regret(self, x: dict) -> float:
        return self.max_value - self.mean_reward(x)

    def expected_runtime_seconds(self, x: dict) -> float:
        return self._query(x)[1]

    def expected_runtime_steps(self, x: dict) -> float:
        """Deterministic per-config delay in simulator steps, before censoring
        and multiplicative jitter (applied by ``RuntimeDelayModel``)."""
        return self.expected_runtime_seconds(x) / self._time_unit

    def reset_noise(self, seed: int) -> None:
        """Re-seed the Bernoulli reward stream for an independent run. The
        tabular mean/runtime caches are deterministic and safely shared."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------- reference optimum

    def _reference_stats(self) -> tuple[float, float]:
        """Exact p* = max validation accuracy over ALL architectures, plus the
        median training time (for the delay time-scale). Enumerated once over
        the finite tabular space and cached to assets/."""
        cache = _REF_CACHE_DIR / f"nasbench201_{self.instance}_ref.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            return d["max_value"], d["median_runtime_seconds"]

        n = len(self._api)  # 15625 for the topology search space
        accs = np.empty(n)
        runtimes = np.empty(n)
        for idx in range(n):
            info = self._api.get_more_info(
                idx, _QUERY_DATASET[self.instance], hp=_HP, is_random=False
            )
            accs[idx] = float(info.get("valid-accuracy", info.get("valtest-accuracy")))
            runtimes[idx] = float(
                info.get("train-all-time", info.get("train-per-time", 0.0))
            )
        max_value = float(min(1.0, accs.max() / 100.0))
        median_runtime = float(np.median(runtimes))
        cache.write_text(
            json.dumps(
                {
                    "max_value": max_value,
                    "median_runtime_seconds": median_runtime,
                    "n_archs": int(n),
                }
            )
        )
        return max_value, median_runtime
