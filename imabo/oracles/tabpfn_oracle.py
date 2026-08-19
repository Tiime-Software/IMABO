from __future__ import annotations

import contextlib
import threading
from statistics import NormalDist
from typing import Any, Callable, Literal

import numpy as np

from imabo.imabo import IMABO
from imabo.memory import ArmStats, CurrentState, key_to_config
from imabo.oracle import Oracle
from imabo.oracles.candidate_pool import (
    CandidatePool,
    categorical_indices,
    rank,
    to_frame,
)
from imabo.policies.imoss import IMOSS
from imabo.types import ArmConfig, ArmKey

# TabPFN's MPS backend is not safe to enter concurrently.
_MPS_LOCK = threading.Lock()


def _serialization_lock() -> Any:
    """The MPS lock when running on Apple GPUs, otherwise a no-op."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return _MPS_LOCK
    except Exception:
        pass
    return contextlib.nullcontext()


def quantile_to_kappa(quantile: float) -> float:
    """The UCB weight matching a quantile level under a Gaussian: ``Phi^-1(q)``."""
    return NormalDist().inv_cdf(min(max(quantile, 1e-6), 1 - 1e-6))


def load_tabpfn(
    model_type: str = "regression",
    device: str = "auto",
    model_path: str = "auto",
    warmup: bool = True,
) -> dict[str, Any]:
    """Resolve TabPFN's settings once, so every run shares one cached checkpoint.

    Returns the ``{"model_path", "device"}`` dict to hand to :class:`TabPFNOracle` as
    ``model``. With ``warmup`` it also runs a throwaway fit to force the download up front
    rather than inside the first optimization round.
    """
    from tabpfn import TabPFNRegressor

    assert model_type == "regression", "only the regression variant is used"

    if warmup:
        import pandas as pd

        regressor = TabPFNRegressor(
            n_estimators=1, model_path=model_path, device=device
        )
        regressor.fit(
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}), np.array([0.0, 1.0, 0.0, 1.0])
        )
        regressor.predict(pd.DataFrame({"x": [0.5]}))

    return {"model_path": model_path, "device": device}


class TabPFNOracle(Oracle):
    """Score a candidate pool with TabPFN-3 and admit the most promising one.

    Example:
        >>> model = load_tabpfn()
        >>> optimizer = IMABO(space, IMOSS(beta=0.5), TabPFNOracle(model=model))

    Args:
        pool: The candidate pool to rank. Defaults to 100-candidate mixture.
        min_arms: Rewarded arms required before fitting at all; below it the oracle draws
            from ``P0``, since a handful of points is not a table worth conditioning on.
        acquisition: ``"quantile"`` ranks on the ``quantile`` level of the posterior;
            ``"ucb"`` ranks on ``mean + kappa * std`` at the equivalent Gaussian weight.
        quantile: The single exploration knob, in (0, 1), shared by both acquisitions.
        n_estimators: Ensemble size. Each member conditions on the same table under a
            different feature ordering and preprocessing, and is one forward pass -- this
            is the dominant cost.
        max_num_rows: Cap on in-context rows per call, bounding cost as the run grows.
            ``None`` uses every row.
        refit_every: Calls served per fit. At 1 (the default) the model is refit every
            call; above 1 the top candidates are cached and served without refitting, which
            is cheaper but ranks them against arms that did not exist yet.
        fit_granularity: ``"arm"`` fits one row per rewarded arm at its mean reward;
            ``"pull"`` fits one row per individual pull, doing no averaging.
        fit_mode: TabPFN's ``fit_mode``. ``None`` leaves it to TabPFN in arm mode and
            selects the KV cache in pull mode, where the bigger table makes caching pay.
        model: Output of :func:`load_tabpfn`. Falls back to ``model_path``/``device``.
        model_kwargs: Extra keyword arguments for ``TabPFNRegressor``.
        on_suggestion: Diagnostics hook, called with ``(config, mean, score)`` every time
            the oracle returns a configuration. No effect on behaviour.
        on_candidates_scored: Diagnostics hook, called with ``(candidates, means)`` on every
            real fit, giving the whole scored pool. No effect on behaviour.
    """

    def __init__(
        self,
        pool: CandidatePool | None = None,
        min_arms: int = 10,
        acquisition: Literal["ucb", "quantile"] = "quantile",
        quantile: float = 0.975,
        n_estimators: int = 4,
        max_num_rows: int | None = 200,
        refit_every: int = 1,
        fit_granularity: Literal["arm", "pull"] = "arm",
        fit_mode: str | None = None,
        model: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        model_path: str = "auto",
        on_suggestion: Callable[[ArmConfig, float, float], None] | None = None,
        on_candidates_scored: (
            Callable[[list[ArmConfig], np.ndarray], None] | None
        ) = None,
    ):
        if acquisition not in ("ucb", "quantile"):
            raise ValueError(f"invalid acquisition: {acquisition!r}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be at least 1, got {refit_every}")
        self.pool = pool if pool is not None else CandidatePool()
        self.min_arms = min_arms
        self.acquisition = acquisition
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        self.fit_granularity = fit_granularity
        # The KV cache pays off only for the big per-pull table; leave arm mode on
        # TabPFN's own default.
        self.fit_mode = (
            fit_mode
            if fit_mode is not None
            else ("fit_with_cache" if fit_granularity == "pull" else None)
        )
        self.model_kwargs = model_kwargs or {}

        settings = model if model is not None else {}
        self._model = settings
        self._model_path = settings.get("model_path", model_path)
        self._device = settings.get("device", device)

        self.on_suggestion = on_suggestion
        self.on_candidates_scored = on_candidates_scored
        self._shortlist: list[tuple[ArmConfig, float, float]] = []
        # Per-arm log of individual rewards, populated only in "pull" mode.

    def suggest(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        score: Callable[[ArmKey], float],
    ) -> ArmConfig:
        if len(rewarded_arms) < self.min_arms:
            return self.space.sample(self.rng)

        if self._shortlist:
            return self._serve(self._shortlist.pop(0))

        candidates = self.pool.build(rewarded_arms, self.space, self.rng)
        candidates = self.pool.drop_duplicates_and_open(
            candidates, state, self.space, self.rng
        )
        frame = to_frame(self.space, candidates)

        with _serialization_lock():
            model = self._fit(rewarded_arms)
            if self.acquisition == "quantile":
                means, scores = self._predict(model, frame)
            else:
                means, stds = self._predict(model, frame, spread=True)
                scores = means + quantile_to_kappa(self.quantile) * stds

        if self.on_candidates_scored is not None:
            self.on_candidates_scored(candidates, means)

        ranked = [
            (candidates[i], float(means[i]), float(scores[i]))
            for i in rank(scores, self.refit_every)
        ]
        chosen, self._shortlist = ranked[0], ranked[1:]
        return self._serve(chosen)

    def _serve(self, entry: tuple[ArmConfig, float, float]) -> ArmConfig:
        config, mean, score = entry
        if self.on_suggestion is not None:
            self.on_suggestion(config, mean, score)
        return config

    def setup(self, space, rng, memory) -> None:
        super().setup(space, rng, memory)
        if self.fit_granularity == "pull" and not callable(
            getattr(memory, "get_rewards", None)
        ):
            raise TypeError(
                f'{type(memory).__name__} has no get_rewards(). fit_granularity="pull" '
                "fits one row per individual pull, which ArmStats does not keep -- see "
                "the oracle blocks in InMemoryStorage."
            )

    def _rows(
        self, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> tuple[list[ArmConfig], np.ndarray]:
        """Assemble the (configs, rewards) table the model is fit on for this step.

        ``fit_granularity="arm"`` (default) -> one row per rewarded arm at its
        running mean reward. ``"pull"`` -> one row per individual pull, expanding
        each arm via the memory's reward log so no averaging is done; an arm without
        a logged pull (e.g. rewarded before pull-logging was active) falls back to a
        single mean-reward row.
        """
        if self.fit_granularity == "pull":
            configs: list[ArmConfig] = []
            rewards: list[float] = []
            for key, stats in rewarded_arms:
                config = key_to_config(key, self.space.names)
                pulls = self.memory.get_rewards(key)
                if pulls:
                    configs.extend([config] * len(pulls))
                    rewards.extend(pulls)
                else:
                    configs.append(config)
                    rewards.append(float(stats.mean_reward))
            return configs, np.asarray(rewards, dtype=float)

        return (
            [key_to_config(k, self.space.names) for k, _ in rewarded_arms],
            np.array([stats.mean_reward for _, stats in rewarded_arms]),
        )

    def _fit(self, rewarded_arms: list[tuple[ArmKey, ArmStats]]) -> Any:
        from tabpfn import TabPFNRegressor

        configs, rewards = self._rows(rewarded_arms)
        if self.max_num_rows is not None and len(configs) > self.max_num_rows:
            keep = self.rng.sample(range(len(configs)), self.max_num_rows)
            configs = [configs[i] for i in keep]
            rewards = rewards[keep]

        extra = dict(self.model_kwargs)
        if self.fit_mode is not None and "fit_mode" not in extra:
            extra["fit_mode"] = self.fit_mode

        regressor = TabPFNRegressor(
            n_estimators=self.n_estimators,
            categorical_features_indices=categorical_indices(self.space) or None,
            random_state=self.rng.randint(0, 2**32 - 1),
            model_path=self._model_path,
            device=self._device,
            # Arm-mode tables are tiny; pull-mode ones can exceed TabPFN's pretraining
            # size. Either way this silences the >1000-samples-on-CPU guard.
            ignore_pretraining_limits=True,
            **extra,
        )
        regressor.fit(to_frame(self.space, configs), rewards)
        return regressor

    def _predict(
        self, model: Any, frame: Any, spread: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean, and either its standard deviation or its ``quantile`` level.

        Both come from one ``predict(output_type="full")`` call, which returns the
        ensemble already aggregated into a single calibrated distribution.
        """
        full = model.predict(frame, output_type="full")
        logits, criterion = full["logits"], full["criterion"]
        mean = criterion.mean(logits).cpu().detach().numpy().astype(float)
        if spread:
            variance = criterion.variance(logits).cpu().detach().numpy().astype(float)
            return mean, np.sqrt(np.clip(variance, 0.0, None))
        upper = criterion.icdf(logits, self.quantile).cpu().detach().numpy()
        return mean, upper.astype(float)


class IMOSSTabPFN(IMABO):
    """IMOSS paired with the TabPFN oracle."""

    def __init__(
        self,
        search_space,
        *,
        beta: float = 0.5,
        seed: int | None = None,
        **oracle: Any,
    ):
        super().__init__(
            search_space, IMOSS(beta=beta), TabPFNOracle(**oracle), seed=seed
        )
