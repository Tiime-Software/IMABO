"""Large finite-armed conversion bandit built from the real Criteo click logs.

Ground truth is the Criteo Sponsored Search Conversion Log (Criteo AI Lab): a
90-day sample of live ad clicks, each row a click with product/context features,
a binary ``Sale`` (did it convert), and ``time_delay_for_conversion`` (seconds
from click to conversion, ``-1`` if it never converted). Criteo conversion logs
are a standard basis for delayed-feedback bandit work (Chapelle 2014, KDD,
doi:10.1145/2623330.2623634; Vernade, Cappe & Perchet 2017, UAI,
arXiv:1706.09186).

We turn it into a finite-armed Bernoulli bandit the same way
``rf_tabular_bandit.py`` turns HPOBench into one:

    arm  x   = a *segment*: a composite of a few product/context categoricals
               (default product_category1 x device_type x product_age_group x
               product_gender x product_brand), enumerated over the segments
               that occur often enough in the log (min-support filtered). The
               segment set is large (thousands) -- the many-armed regime IMABO
               targets -- and finite/discrete, exposed to IMABO as a single
               categorical dimension whose choices are the segment keys, so
               every arm the optimizer can propose is a real, populated segment
               with a reliable rate (no sparse-cell extrapolation).

    mean p_x = the segment's empirical conversion rate over the log.
    pull     = draw Bernoulli(p_x) -> reward in {0, 1} (the Sale event).
    regret   = p* - p_x, with p* = max_x p_x (known exactly, finite arm set).

Delay is handled separately by ``CriteoDelayModel`` (delay_model.py), a
conversion-signal model whose delay distribution is the same log's real
``time_delay_for_conversion`` values -- so both the reward surface and the delay
come from one real dataset, plugged into the two slots the delayed simulator
already has (reward via ``bench(x)``, delay via the delay model).

Censoring here means the outcome is NEVER OBSERVED -- specifically a real
conversion (reward 1) whose click->conversion delay exceeds the patience window,
so its positive signal is lost. It is NOT the base non-conversion rate: a
reward of 0 (no conversion) is an OBSERVED low reward that always arrives (you
wait out the window and record a 0), not a censored pull. Low conversion rate
therefore makes this a low-mean bandit, not a heavily censored one; censoring is
governed by delay-vs-patience on the positive signals.

Setup (one-time): download the log from Criteo AI Lab and run
``experiments/benchmarks/delayed/build_criteo_asset.py`` to produce the compact
per-segment asset this class reads (the raw log is large; it is processed once).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).parent / "assets"


class CriteoConversionBenchmark:
    """Finite segment-armed Bernoulli conversion bandit (real Criteo rates).

    Duck-type compatible with ``RFTabularFiniteBenchmark`` -- the delayed
    simulator only needs ``bench(x, noise=True) -> float``,
    ``bench.regret(x) -> float``, ``bench.get_search_space()``,
    ``bench.max_value``, ``bench.reset_noise(seed)`` -- so it drops into the
    same experiment/plot pipeline. Deliberately exposes NO
    ``expected_runtime_steps`` (the delay is not a per-arm runtime here but a
    pooled empirical distribution owned by CriteoDelayModel), so the simulator
    passes ``expected_steps=None`` and the delay model samples from the log.
    """

    def __init__(
        self,
        instance: str = "sponsored_search",
        seed: int = 0,
    ):
        self.instance = str(instance)
        self.bm_id = self.instance  # filename tag, mirrors RFTabular.bm_id
        self.rng = np.random.default_rng(seed)

        asset = ASSETS_DIR / f"criteo_{self.instance}_arms.csv"
        meta_path = ASSETS_DIR / f"criteo_{self.instance}_arms_meta.json"
        if not asset.exists():
            raise FileNotFoundError(
                f"{asset} not found -- run "
                f"`python -m experiments.benchmarks.delayed.build_criteo_asset` "
                f"on the downloaded Criteo Sponsored Search Conversion Log first."
            )
        table = pd.read_csv(asset, dtype={"segment": str})
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # segment key -> empirical conversion rate (the arm's true mean p_x).
        self.lookup: dict[str, float] = dict(
            zip(table["segment"], table["conversion_rate"])
        )
        self.counts: dict[str, int] = dict(zip(table["segment"], table["count"]))
        self.segments = list(self.lookup.keys())
        self.n_arms = len(self.segments)

        best_seg = max(self.lookup, key=self.lookup.get)
        self.best_config = {"segment": best_seg}
        self.max_value = self.lookup[best_seg]

    def get_search_space(self) -> dict[str, dict]:
        """A single categorical dimension over the populated segment keys."""
        return {"segment": {"choices": list(self.segments)}}

    def mean_reward(self, x: dict) -> float:
        return self.lookup[x["segment"]]

    def __call__(self, x: dict, noise: bool = True) -> float:
        p = self.mean_reward(x)
        if not noise:
            return p
        return float(self.rng.binomial(1, p))

    def regret(self, x: dict) -> float:
        return self.max_value - self.mean_reward(x)

    def reset_noise(self, seed: int) -> None:
        """Re-seed the Bernoulli reward stream for an independent run (mirrors
        RFTabularFiniteBenchmark.reset_noise). The per-arm rate lookup is
        deterministic and safely shared across runs."""
        self.rng = np.random.default_rng(seed)
