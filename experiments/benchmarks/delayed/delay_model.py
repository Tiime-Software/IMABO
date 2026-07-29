"""Delay/censoring model calibrated from an empirical feedback-timing distribution."""

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
    an empirical feedback-timing distribution.

    1 step is treated as 1 hour, matching the unit the log-normal fit was
    computed in -- the same simplifying assumption used by the original
    calibration this was ported from.
    """

    mu: float = _FIT["mu"]
    sigma: float = _FIT["sigma"]
    feedback_freq: float = _FIT["feedback_freq"]

    def sample_delay_steps(
        self, rng: np.random.Generator, expected_steps: float | None = None, **kwargs
    ) -> int | None:
        """Return the delay (in steps) until feedback arrives, or ``None``
        if this pull is Bernoulli-censored (never receives feedback).

        ``expected_steps`` (a config-dependent delay hint) and any other
        simulator kwargs (``reward``, ``patience_steps`` -- used only by
        CriteoDelayModel) are accepted for a uniform interface and ignored:
        this model's delay is a fitted log-normal, independent of the config or
        the reward value. Censoring here is value-agnostic (a coin flip on
        arrival), which is correct: it does not depend on whether the reward is
        0 or 1. :class:`RuntimeDelayModel` is the model that uses expected_steps.
        """
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
        observed" -- see experiments/delayed_feedback_experiment.py (severity sweep section).
        """
        return cls(
            mu=_FIT["mu"] + math.log(delay_scale),
            sigma=_FIT["sigma"],
            feedback_freq=(
                feedback_freq if feedback_freq is not None else _FIT["feedback_freq"]
            ),
        )


@dataclass
class RuntimeDelayModel:
    """Config-dependent delay: the delay of a pull *is* the predicted training
    time of that configuration.

    This is the production-realistic delay for HPO. When you launch a training
    job for a configuration, its validation score comes back after the job
    finishes; an expensive configuration (deep/wide net, small batch) reports
    late, and a job that crashes or is preempted never reports at all. The
    asynchronous-HPO literature builds delayed benchmarks exactly this way --
    predict the runtime from a surrogate and let the objective "sleep" for it
    (Watanabe et al. 2024, arXiv:2403.01888; Zimmer et al. 2021,
    Auto-PyTorch, arXiv:2006.13799). We use LCBench's
    per-configuration ``time`` objective as that runtime
    (see ``lcbench_bandit.LCBenchMixedBenchmark.expected_runtime_steps``).

    Delay = ``expected_steps`` (the config's predicted runtime, already rescaled
    to steps by the benchmark) times a multiplicative log-normal jitter of
    median 1 (system/queueing noise on top of the deterministic surrogate
    estimate). ``delay_scale`` multiplies the whole thing for the severity
    sweep. Censoring is Bernoulli with rate ``1 - feedback_freq``, independent
    of the config -- the "job crashed / user never converted" component -- on
    top of the deterministic expiry of any pull whose runtime exceeds the
    patience window (handled by the simulator).

    ``feedback_freq=1.0`` (default) means *no* Bernoulli censoring: the only
    censoring is then endogenous -- configs whose predicted runtime exceeds the
    patience window -- which is the natural, benchmark-driven censoring story.
    """

    delay_scale: float = 1.0
    jitter_sigma: float = 0.5
    feedback_freq: float = 1.0

    def sample_delay_steps(
        self, rng: np.random.Generator, expected_steps: float | None = None, **kwargs
    ) -> int | None:
        """Return the delay (in steps) until feedback arrives, or None if the feedback is censored."""
        if rng.random() >= self.feedback_freq:
            return None
        if expected_steps is None:
            raise ValueError(
                "RuntimeDelayModel requires a benchmark exposing "
                "expected_runtime_steps(x); got expected_steps=None. Use it "
                "with LCBenchMixedBenchmark, not RFTabularFiniteBenchmark."
            )
        # Multiplicative log-normal jitter with median 1 (mean of the log is 0).
        jitter = rng.lognormal(mean=0.0, sigma=self.jitter_sigma)
        delay = self.delay_scale * expected_steps * jitter
        return max(0, int(round(delay)))

    @classmethod
    def at_severity(
        cls, delay_scale: float = 1.0, feedback_freq: float | None = None
    ) -> "RuntimeDelayModel":
        """Severity-sweep variant, mirroring ``DelayModel.at_severity``.

        ``delay_scale`` multiplies every config's runtime-derived delay (the
        delay-length axis); ``feedback_freq`` overrides the extra Bernoulli
        censoring rate (the censoring axis), defaulting to 1.0 = no extra
        censoring beyond patience-window expiry.
        """
        return cls(
            delay_scale=delay_scale,
            feedback_freq=1.0 if feedback_freq is None else feedback_freq,
        )
