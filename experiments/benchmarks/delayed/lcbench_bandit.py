"""Mixed continuous+finite bandit built from the LCBench surrogate benchmark.

Ground truth is the YAHPO-Gym surrogate for LCBench (Zimmer et al. 2021,
Auto-PyTorch Tabular, arXiv:2006.13799; surrogate via
Pfisterer et al. 2022): a random-forest/neural surrogate trained on the learning
curves of an AutoPyTorch MLP on OpenML datasets. Unlike the finite RF tabular
grid (``rf_tabular_bandit.py``), the LCBench search space is genuinely
**mixed continuous + finite** and, on the continuous axes, **infinite** -- which
is the regime IMABO's TPE oracle is built for:

    num_layers      categorical/finite   {1, ..., 5}
    batch_size      integer (log)        [16, 512]
    max_units       integer (log)        [64, 1024]
    learning_rate   continuous (log)     [1e-4, 1e-1]
    momentum        continuous           [0.1, 0.99]
    weight_decay    continuous           [1e-5, 1e-1]
    max_dropout     continuous           [0.0, 1.0]

(exact bounds/log-flags are read from the live ConfigSpace at construction, not
hardcoded, so this stays correct across yahpo_gym versions.)

Why this benchmark for the *delayed/censored* experiment. In production HPO the
feedback delay is not an exogenous nuisance to be injected -- it *is* the
training time of the configuration: an expensive net (many layers/units, small
batch) reports its validation score late, and a job that crashes or is preempted
never reports at all. LCBench ships a per-configuration ``time`` objective from
the same surrogate, so we derive each pull's delay from its own predicted
runtime (see :meth:`expected_runtime_steps` and
``experiments/benchmarks/delayed/delay_model.RuntimeDelayModel``). This mirrors
how the asynchronous-HPO literature builds delayed benchmarks -- predicting the
runtime and letting the objective "sleep" for it (e.g. Watanabe et al. 2024,
"Fast Benchmarking of Asynchronous Multi-Fidelity Optimization on Zero-Cost
Benchmarks", arXiv:2403.01888; Zimmer et al. 2021, Auto-PyTorch Tabular,
arXiv:2006.13799) -- rather than sampling delay from a fitted external log.

Reward semantics match ``rf_tabular_bandit.py`` so the two benchmarks are
interchangeable in the delayed simulator: the surrogate's validation accuracy
(0--100) is turned into a Bernoulli success probability in [0, 1] and pulling an
arm draws one Bernoulli sample. Binary feedback is also what makes *censoring*
meaningful -- an un-arrived reward is indistinguishable from a zero, the core
difficulty studied in the delayed-conversion bandit literature (Vernade, Cappe
& Perchet 2017, UAI, arXiv:1706.09186; Chapelle 2014, KDD).

Setup (one-time): the surrogate ONNX files + metadata must be present locally.
See ``experiments/benchmarks/delayed/setup_lcbench.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Cache of the reference optimum (p* = max attainable accuracy) per instance, so
# the expensive dense surrogate search that estimates it runs once, not per run.
_REF_CACHE_DIR = Path(__file__).parent / "assets"
_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OpenML task_id -> readable dataset name, for figure titles. These are LCBench
# instances (yahpo lists task ids as instance strings) that do NOT overlap with
# the RF tabular tasks already used in the paper (146822 segment, 31 credit-g,
# 167120 numerai28.6) or the continuous LR/SVM task (167149). Verified against
# OpenML directly (task -> dataset id -> name), since yahpo's own instance
# strings give no indication of which is which and it is easy to mismatch (an
# earlier version of this dict had 126026 labeled "higgs" and 167104 labeled
# "Fashion-MNIST" -- both wrong; the real higgs is 167200 and the real
# Fashion-MNIST is 189908). Extend as needed; unknown ids fall back to the raw
# id in `_bench_title`.
LCBENCH_INSTANCE_NAMES = {
    "167200": "higgs",
    "168330": "jannis",
    "189908": "Fashion-MNIST",
    "168868": "APSFailure",
    "189354": "airlines",
}

# Integer hyperparameters with at most this many distinct values are exposed to
# IMABO as a *categorical* (finite) axis rather than a bounded integer, so the
# search space is genuinely mixed. num_layers ([1,5]) becomes {1,...,5}.
_CATEGORICAL_INT_THRESHOLD = 8

# The surrogate `time` objective is a raw training-time estimate (seconds). We
# rescale it to simulator "steps" so the *median* configuration's delay is this
# many steps -- large enough that fast configs arrive within the patience window
# while expensive ones lag or get censored. Tunable per experiment.
_TARGET_MEDIAN_DELAY_STEPS = 6.0


def _as_result_dict(raw: Any) -> dict:
    """yahpo `objective_function` returns a list of result dicts (one per input
    config) or, in some versions, a single dict. Normalize to one dict."""
    if isinstance(raw, list):
        return raw[0]
    return raw


class LCBenchMixedBenchmark:
    """LCBench surrogate as a mixed-space bandit with runtime-derived delay.

    Duck-type compatible with ``RFTabularFiniteBenchmark`` (the delayed
    simulator only needs ``bench(x, noise=True) -> float``,
    ``bench.regret(x) -> float``, ``bench.get_search_space()``,
    ``bench.max_value``, ``bench.reset_noise(seed)``), plus the extra
    ``expected_runtime_steps(x)`` consumed by ``run_delayed`` when a
    ``RuntimeDelayModel`` is used.
    """

    def __init__(
        self,
        instance: str | int = "126026",
        metric: str = "val_accuracy",
        seed: int = 0,
        n_ref_samples: int = 100_000,
        categorical_int_threshold: int = _CATEGORICAL_INT_THRESHOLD,
        target_median_delay_steps: float = _TARGET_MEDIAN_DELAY_STEPS,
    ):
        # Imported lazily so importing this module never requires yahpo_gym
        # (only constructing the benchmark does) -- matches how optimizer.py
        # treats the optional TabFM dependency.
        import yahpo_gym.benchmarks.lcbench  # noqa: F401  (registers the scenario)
        from yahpo_gym import benchmark_set

        self.instance = str(instance)
        self.bm_id = self.instance  # filename tag
        self.metric = metric
        self.rng = np.random.default_rng(seed)

        self._bench = benchmark_set.BenchmarkSet("lcbench")
        self._bench.set_instance(self.instance)

        # A full valid config template (params + task-id constant + fidelity),
        # so eval configs always carry every key objective_function requires;
        # we only overwrite the tunable params and pin the fidelity to max.
        self._template = (
            self._bench.config_space.sample_configuration().get_dictionary()
        )

        # Max fidelity (train to the last epoch): full validation score + full
        # training time = the largest, most production-like delay.
        fspace = self._bench.get_fidelity_space()
        fid_hp = fspace.get_hyperparameters()[0]
        self._fidelity_name = fid_hp.name
        self._max_fidelity = int(fid_hp.upper)

        opt_space = self._bench.get_opt_space()
        self._search_space, self.param_names = self._build_search_space(
            opt_space, categorical_int_threshold
        )
        self.dim = len(self.param_names)

        # Per-config memoization: the surrogate mean and runtime are
        # deterministic, so cache them (keyed by the rounded config) -- MOSS
        # re-pulls the same exploited arm many times.
        self._mean_cache: dict[tuple, float] = {}
        self._runtime_cache: dict[tuple, float] = {}

        self.max_value = self._estimate_reference_optimum(n_ref_samples)

        # Set the seconds-per-step scale from the median runtime over the same
        # reference sample, so expected_runtime_steps has the target median.
        self._time_unit = max(
            1e-9, self._median_runtime_seconds / target_median_delay_steps
        )

    # ------------------------------------------------------------------ space

    def _build_search_space(
        self, opt_space, categorical_int_threshold: int
    ) -> tuple[dict[str, dict], list[str]]:
        """Convert the yahpo ConfigSpace to IMABO's search-space dict format."""
        import ConfigSpace.hyperparameters as csh

        space: dict[str, dict] = {}
        for hp in opt_space.get_hyperparameters():
            name = hp.name
            if name == self._fidelity_name:
                # yahpo's lcbench opt space includes the fidelity (epoch) as a
                # tunable, but _to_full_config pins it to max fidelity AFTER
                # applying the optimizer's values -- so as a search axis it is
                # DEAD: zero effect on reward or runtime (measured eta^2 = 0.000
                # over 20k surrogate samples). Exposing it would make every
                # optimizer spend exploration budget on a no-op coordinate.
                continue
            if isinstance(hp, csh.CategoricalHyperparameter):
                space[name] = {"choices": list(hp.choices)}
            elif isinstance(hp, csh.OrdinalHyperparameter):
                space[name] = {"choices": list(hp.sequence)}
            elif isinstance(hp, csh.UniformIntegerHyperparameter):
                n_values = int(hp.upper) - int(hp.lower) + 1
                if n_values <= categorical_int_threshold:
                    # Small integer range -> genuine finite/categorical axis.
                    space[name] = {
                        "choices": list(range(int(hp.lower), int(hp.upper) + 1))
                    }
                else:
                    space[name] = {
                        "lower": int(hp.lower),
                        "upper": int(hp.upper),
                        "int": True,
                        "log": bool(getattr(hp, "log", False)),
                    }
            elif isinstance(hp, csh.UniformFloatHyperparameter):
                space[name] = {
                    "lower": float(hp.lower),
                    "upper": float(hp.upper),
                    "log": bool(getattr(hp, "log", False)),
                }
            elif isinstance(hp, csh.Constant):
                # Fixed per instance (e.g. OpenML_task_id); kept in _template.
                continue
            else:
                raise TypeError(
                    f"Unsupported hyperparameter type for {name}: {type(hp)}"
                )
        return space, sorted(space.keys())

    def get_search_space(self) -> dict[str, dict]:
        """Mixed search space in IMABO format (choices / {lower,upper,log,int})."""
        return self._search_space

    # ------------------------------------------------------------- surrogate

    def _key(self, x: dict) -> tuple:
        return tuple((n, x[n]) for n in self.param_names)

    def _to_full_config(self, x: dict) -> dict:
        cfg = dict(self._template)
        for n in self.param_names:
            v = x[n]
            # optuna/IMABO hands back numpy scalars; yahpo wants plain types.
            if hasattr(v, "item"):
                v = v.item()
            cfg[n] = v
        cfg[self._fidelity_name] = self._max_fidelity
        return cfg

    def _query(self, x: dict) -> tuple[float, float]:
        """Return (success_prob in [0,1], runtime_seconds) for config ``x``,
        memoized. Success prob = surrogate val_accuracy / 100."""
        key = self._key(x)
        if key in self._mean_cache:
            return self._mean_cache[key], self._runtime_cache[key]
        res = _as_result_dict(self._bench.objective_function(self._to_full_config(x)))
        acc = float(res[self.metric])
        prob = min(1.0, max(0.0, acc / 100.0))
        runtime = float(res["time"])
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
        """Re-seed the Bernoulli reward stream for an independent run (mirrors
        ``RFTabularFiniteBenchmark.reset_noise``). The surrogate mean/runtime
        caches are deterministic and safely shared across runs."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------- reference optimum

    def _sample_opt_configs(self, n: int) -> list[dict]:
        opt_space = self._bench.get_opt_space()
        cfgs = opt_space.sample_configuration(n)
        cfgs = cfgs if isinstance(cfgs, list) else [cfgs]
        return [c.get_dictionary() for c in cfgs]

    def _estimate_reference_optimum(self, n_ref_samples: int) -> float:
        """Estimate p* = max attainable accuracy by a dense random search over
        the surrogate (cached to assets/). Regret on a continuous space needs a
        reference optimum; the surrogate has no closed-form max, so we
        approximate it by the best of a large batched sample -- the standard
        approach for surrogate/continuous HPO benchmarks. Also records the
        median runtime used to set the delay time-scale."""
        cache = _REF_CACHE_DIR / f"lcbench_{self.instance}_{self.metric}_ref.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            self._median_runtime_seconds = d["median_runtime_seconds"]
            return d["max_value"]

        rng = np.random.default_rng(0)  # fixed: p* must not depend on run seed
        # Seed yahpo's ConfigSpace sampler deterministically where possible.
        try:
            self._bench.get_opt_space().seed(0)
        except Exception:
            pass

        configs = self._sample_opt_configs(n_ref_samples)
        full = [self._to_full_config(c) for c in configs]

        accs: list[float] = []
        runtimes: list[float] = []
        CHUNK = 5000
        for i in range(0, len(full), CHUNK):
            batch = full[i : i + CHUNK]
            raw = self._bench.objective_function(batch)
            results = raw if isinstance(raw, list) else [raw]
            for r in results:
                accs.append(float(r[self.metric]) / 100.0)
                runtimes.append(float(r["time"]))

        max_value = float(min(1.0, max(accs)))
        self._median_runtime_seconds = float(np.median(runtimes))
        cache.write_text(
            json.dumps(
                {
                    "max_value": max_value,
                    "median_runtime_seconds": self._median_runtime_seconds,
                    "n_ref_samples": len(accs),
                }
            )
        )
        return max_value
