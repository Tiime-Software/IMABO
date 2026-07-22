"""Delay/censoring model calibrated from real production feedback timing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_FIT = {
    "mu": 0.5795,
    "sigma": 2.3226,
    "feedback_freq": 0.40698,
}


@dataclass
class DelayModel:
    """Samples per-pull feedback delay in simulator steps, calibrated from
    real production feedback timing.

    1 step is treated as 1 hour, matching the unit the log-normal fit was
    computed in -- the same simplifying assumption used by the original
    gungnir port.
    """

    mu: float = _FIT["mu"]
    sigma: float = _FIT["sigma"]
    feedback_freq: float = _FIT["feedback_freq"]

    def sample_delay_steps(self, rng: np.random.Generator) -> int | None:
        """Return the delay (in steps) until feedback arrives, or ``None``
        if this pull is censored (never receives feedback)."""
        if rng.random() >= self.feedback_freq:
            return None
        delay_hours = rng.lognormal(mean=self.mu, sigma=self.sigma)
        return max(0, int(round(delay_hours)))

    @classmethod
    def at_severity(
        cls, delay_scale: float = 1.0, feedback_freq: float | None = None
    ) -> "DelayModel":
        """Build a variant of the calibrated model for a severity sweep.

        `delay_scale` multiplies the expected delay (median * delay_scale):
        since delay ~ LogNormal(mu, sigma), scaling the underlying variable
        by a constant `s` is exactly `mu -> mu + log(s)` (sigma unchanged).
        `feedback_freq` overrides the censoring rate directly if given,
        otherwise keeps the calibrated real value. Used to isolate "what if
        delay/censoring were worse (or better) than what we actually
        observed" -- see experiments/delayed_feedback_severity_experiment.py.
        """
        return cls(
            mu=_FIT["mu"] + math.log(delay_scale),
            sigma=_FIT["sigma"],
            feedback_freq=feedback_freq if feedback_freq is not None else _FIT["feedback_freq"],
        )
