from experiments.benchmarks.config import BENCHMARKS
from typing import Any
import math
import numpy as np
import random
from pathlib import Path
from joblib import Memory

from experiments.benchmarks.hpo_bench.client import api_call

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hpo_wrapper")


def convert_numpy_to_python(
    x: dict[str, Any], param_specs: dict[str, Any]
) -> dict[str, Any]:
    """Convert numpy scalar values in a config dict to plain Python types."""
    converted: dict[str, Any] = {}
    for name, value in x.items():
        if hasattr(value, "item"):
            value = value.item()
        elif isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        spec = param_specs.get(name, {})
        if spec.get("choices"):
            converted[name] = value
        elif spec.get("int", False):
            converted[name] = int(value)
        else:
            converted[name] = float(value)
    return converted


location = Path(__file__).parent.parent / ".cache"
memory = Memory(location=location, verbose=0)


@memory.cache
async def _api_call_objective_function(
    config: dict[str, Any], fidelity: dict[str, Any], rng: int
) -> dict[str, Any]:
    result = await api_call(
        "objective_function", {"config": config, "fidelity": fidelity, "rng": rng}
    )
    if result is None:
        # Raise rather than return None: this is joblib.Memory-cached, and a
        # cached None from a transient failure (server hiccup, timeout) would
        # permanently poison this (config, fidelity, rng) cache key -- every
        # future run re-derives the same deterministic configs and would keep
        # getting None back without ever retrying the API call.
        raise RuntimeError(
            f"objective_function API call failed for config={config}, "
            f"fidelity={fidelity}, rng={rng}"
        )
    return result


class HPOBenchmark:
    def __init__(self, benchmark_name: str, seed: int):
        self.benchmark_name = benchmark_name
        self.rng = random.Random(seed)

        info = BENCHMARKS.get(benchmark_name)
        if not info or "param_specs" not in info:
            raise ValueError(
                f"Unsupported benchmark or missing param_specs: {benchmark_name}"
            )

        # Metrics
        benchmark_info = BENCHMARKS[self.benchmark_name]
        self.sample_metric = benchmark_info["sample_metric"]
        self.avg_metric = benchmark_info["avg_metric"]
        self.test_metric = benchmark_info["test_metric"]

        self.param_specs = info["param_specs"]
        self.param_names = list(sorted(self.param_specs.keys()))
        self.config_cache: dict[tuple, float] = {}

        self.fidelity = benchmark_info.get("fidelity", {})
        self._sorted_fidelity_names = tuple(sorted(self.fidelity.keys()))
        self.fidelity_tuple = tuple(
            self.fidelity[name] for name in self._sorted_fidelity_names
        )
        self.api_rng = 1
        self.dim = len(self.param_names)

        self._build_search_space()

        self._config_cache: dict[tuple, dict[str, Any]] = {}
        self._reset_cache_hit_count()

    def _build_search_space(self) -> dict[str, Any]:
        self.search_space = {}

        for name, spec in self.param_specs.items():
            if spec.get("choices", None):
                self.search_space[name] = list(spec["choices"])
            else:
                if spec.get("log", False):
                    lower = math.log(spec["lower"])
                    upper = math.log(spec["upper"])
                else:
                    lower = spec["lower"]
                    upper = spec["upper"]
                self.search_space[name] = (lower, upper)  # type: ignore
        return self.search_space

    def _process_result_metrics(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract and process metrics from result"""
        # Validate required metrics exist
        if not (
            self.sample_metric in result
            and result[self.sample_metric]
            and self.avg_metric in result
            and self.test_metric in result
        ):
            raise ValueError(
                f"No sample/avg/test metric found for {self.benchmark_name} in result"
            )

        samples = result[self.sample_metric]
        sample_size = len(samples)
        sample = self.rng.choice(samples)
        avg_sample = result.get(self.avg_metric, float("inf"))
        test_result = result.get(self.test_metric, float("inf"))

        return {
            "sample_result": sample,
            "avg_result": avg_sample,
            "test_result": test_result,
            "val_size": sample_size,
        }

    def _config_to_key(self, config_dict: dict[str, Any]) -> tuple:
        return tuple((name, config_dict[name]) for name in self.param_names)

    async def eval_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a configuration"""
        config_ = convert_numpy_to_python(config, self.param_specs)
        config_tuple = self._config_to_key(config_)
        cache_key = config_tuple
        result = await _api_call_objective_function(
            config_, self.fidelity, self.api_rng
        )
        processed_result = self._process_result_metrics(result)
        self._config_cache[cache_key] = processed_result
        return processed_result

    def save_cache(self) -> None:
        """Save the cache to disk at the end of the run"""
        pass
        # save_cache_to_disk(self._api_cache)

    def array_to_config(self, x: np.ndarray) -> dict[str, Any]:
        """Convert array coordinates to configuration dictionary
        Use to convert XArm suggestions to configuration dictionary
        Does not work for categorical parameters
        """
        config = {}
        for i, name in enumerate(self.param_names):
            spec = self.param_specs[name]
            lower, upper = self.search_space[name]
            value = lower + x[i] * (upper - lower)
            if spec.get("log", False):
                value = math.exp(value)
            value = max(spec["lower"], min(spec["upper"], value))
            if spec.get("int", False):
                value = int(round(value))

            config[name] = value
        return config

    def _reset_cache_hit_count(self) -> None:
        self._cache_hit_count = 0

    def get_config_value(self, config: dict[str, Any]) -> float:
        config_ = convert_numpy_to_python(config, self.param_specs)
        cache_key = self._config_to_key(config_)
        cached_result = self._config_cache.get(cache_key)
        return (
            cached_result.get("avg_result", float("-inf"))
            if cached_result
            else float("-inf")
        )

    @property
    def cache_hit_rate(self) -> float:
        return self._cache_hit_count


async def load_benchmark(benchmark: str) -> bool:
    """Load benchmark"""
    logger.info(f"Loading {benchmark}...")
    result = await api_call("load", {"benchmark": benchmark})
    if not result:
        logger.error("Failed to load benchmark")
        return False
    return True
