
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from imabo.imabo import IMABO
from imabo.memory import ArmStats, CurrentState, key_to_config
from imabo.oracle import Oracle
from imabo.oracles.candidate_pool import CandidatePool, rank, to_frame
from imabo.policies.imoss import IMOSS
from imabo.types import ArmConfig, ArmKey


def load_tabfm(model_type: str = "regression") -> Any:
    """Load TabFM's weights.

    Works around a bug in ``tabfm==1.0.0``'s own ``load()``, which looks for
    ``pytorch_model.bin`` while the released repo only ships ``model.safetensors``.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import (
        HF_REPO_ID,
        ClassificationConfig,
        RegressionConfig,
    )

    config = (
        RegressionConfig() if model_type == "regression" else ClassificationConfig()
    )
    model = TabFM(**config.to_dict())
    checkpoint = Path(snapshot_download(repo_id=HF_REPO_ID)) / model_type
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.eval()
    return model


class TabFMOracle(Oracle):
    """Score a candidate pool with TabFM and admit the most promising one.

    Example:
        >>> optimizer = IMABO(space, IMOSS(beta=0.5), TabFMOracle(model=load_tabfm()))

    Args:
        pool: The candidate pool to rank. Defaults to a purely uniform pool, which is what
            this oracle has always used; pass ``CandidatePool()`` for Sec. 4.1.5's
            mixture around the best arm.
        min_arms: Rewarded arms required before fitting at all.
        suggest_method: ``"ucb"`` ranks on ``mean + kappa * std`` across ensemble members,
            ``"max"`` on their maximum.
        kappa: Exploration weight for ``suggest_method="ucb"``.
        n_estimators: Ensemble size, and the dominant cost -- one forward pass each.
        max_num_rows: Cap on in-context rows per call. ``None`` uses every row.
        refit_every: Calls served per fit; above 1 the top candidates are cached.
        model: A loaded model from :func:`load_tabfm`, shared across runs.
        model_kwargs: Extra keyword arguments for ``TabFMRegressor``.
        on_suggestion: Diagnostics hook, ``(config, mean, ranked_value)``. No effect on
            behaviour.
        on_candidates_scored: Diagnostics hook, ``(candidates, means)``, on every real fit.
    """

    def __init__(
        self,
        pool: CandidatePool | None = None,
        min_arms: int = 10,
        suggest_method: Literal["ucb", "max"] = "ucb",
        kappa: float = 1.0,
        n_estimators: int = 4,
        max_num_rows: int | None = 200,
        refit_every: int = 10,
        model: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        on_suggestion: Callable[[ArmConfig, float, float], None] | None = None,
        on_candidates_scored: (
            Callable[[list[ArmConfig], np.ndarray], None] | None
        ) = None,
    ):
        if suggest_method not in ("ucb", "max"):
            raise ValueError(f"invalid suggest_method: {suggest_method!r}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be at least 1, got {refit_every}")
        self.pool = (
            pool
            if pool is not None
            else CandidatePool(source="uniform", scale=None, filter_open=False)
        )
        self.min_arms = min_arms
        self.suggest_method = suggest_method
        self.kappa = kappa
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        self.model_kwargs = model_kwargs or {}
        self._model = model
        self.on_suggestion = on_suggestion
        self.on_candidates_scored = on_candidates_scored
        self._shortlist: list[tuple[ArmConfig, float, float]] = []

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

        # Fits before drawing the pool, unlike TabPFNOracle. Both orders consume the
        # shared RNG, so they are not interchangeable.
        model = self._fit(rewarded_arms)
        candidates = self.pool.build(rewarded_arms, self.space, self.rng)
        frame = to_frame(self.space, candidates)

        # `_predict_internal` returns per-member predictions in TabFM's *standardized*
        # y-space. Ranking is invariant to that affine transform so scoring there is fine,
        # but the values handed to the hooks must be transformed back, or a
        # predicted-vs-true diagnostic compares standardized numbers against [0, 1]
        # rewards.
        predictions = np.asarray(model._predict_internal(frame))
        mean = predictions.mean(axis=0)
        highest = predictions.max(axis=0)
        if self.suggest_method == "ucb":
            scores = mean + self.kappa * predictions.std(axis=0)
        else:
            scores = highest

        means = model._inverse_transform_y(mean)
        highests = model._inverse_transform_y(highest)

        if self.on_candidates_scored is not None:
            self.on_candidates_scored(candidates, means)

        # The hook reports the ensemble max, which is what `suggest_method="max"` ranks by.
        ranked = [
            (candidates[i], float(means[i]), float(highests[i]))
            for i in rank(scores, self.refit_every)
        ]
        chosen, self._shortlist = ranked[0], ranked[1:]
        return self._serve(chosen)

    def _serve(self, entry: tuple[ArmConfig, float, float]) -> ArmConfig:
        config, mean, value = entry
        if self.on_suggestion is not None:
            self.on_suggestion(config, mean, value)
        return config

    def _fit(self, rewarded_arms: list[tuple[ArmKey, ArmStats]]) -> Any:
        from tabfm import TabFMRegressor

        configs = [key_to_config(k, self.space.names) for k, _ in rewarded_arms]
        rewards = np.array([stats.mean_reward for _, stats in rewarded_arms])
        regressor = TabFMRegressor(
            model=self._model,
            random_state=self.rng.randint(0, 2**32 - 1),
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            **self.model_kwargs,
        )
        regressor.fit(to_frame(self.space, configs), rewards)
        return regressor


class IMOSSTabFM(IMABO):
    """IMOSS paired with the TabFM oracle. Not part of the paper."""

    def __init__(
        self,
        search_space,
        *,
        beta: float = 0.5,
        seed: int | None = None,
        **oracle: Any,
    ):
        super().__init__(
            search_space, IMOSS(beta=beta), TabFMOracle(**oracle), seed=seed
        )
