"""TabFM-based explore oracle for IMABO.

Replaces IMABO's TPE oracle (used to propose brand-new configurations) with
a TabFM (tabular foundation model) regression surrogate. The MOSS-anytime
bandit allocation policy -- both the explore/exploit switching rule and the
exploitation of already-tried arms -- is inherited unchanged from
:class:`imabo.optimizer.IMABO`.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tabfm import TabFMRegressor
from tabfm.src.pytorch.model import TabFM
from tabfm.src.pytorch.tabfm_v1_0_0 import (
    HF_REPO_ID,
    ClassificationConfig,
    RegressionConfig,
)

from imabo.memory import ArmStats, CurrentState, Memory, key_to_config
from imabo.optimizer import IMABO
from imabo.types import ArmConfig, ArmKey


def load_tabfm(model_type: str = "regression") -> TabFM:
    """Load TabFM weights via safetensors.

    Works around a bug in tabfm==1.0.0's own ``load()``: it hardcodes
    looking for ``pytorch_model.bin``, but the HF repo
    ``google/tabfm-1.0.0-pytorch`` currently only ships ``model.safetensors``,
    so ``load()`` raises ``FileNotFoundError``.
    """
    config = (
        RegressionConfig() if model_type == "regression" else ClassificationConfig()
    )
    model = TabFM(**config.to_dict())
    base_path = snapshot_download(repo_id=HF_REPO_ID)
    checkpoint_file = Path(base_path) / model_type / "model.safetensors"
    model.load_state_dict(load_file(checkpoint_file), strict=True)
    model.eval()
    return model


class TabFMIMABO(IMABO):
    """IMABO variant where the explore oracle is TabFM instead of TPE.

    Same switching rule as :class:`IMABO` (beta/delayed) and the same
    MOSS-anytime exploitation of known arms via :meth:`suggest_existing`.
    Only the "propose a brand-new configuration" step changes: instead of
    TPE's good/bad Parzen-estimator split, a TabFM regressor is fit on all
    observed (config, mean_reward) pairs and used to score a pool of
    randomly sampled candidate configurations with a UCB-style acquisition
    (predicted ensemble mean + ``kappa`` * ensemble std). The candidate with
    the highest score is proposed.

    Example:
        >>> optimizer = TabFMIMABO(
        ...     search_space={"x0": {"choices": [0, 1, 2, 3, 4, 5]}},
        ...     min_arms_for_fit=10,
        ... )
        >>> for _ in range(100):
        ...     config = optimizer.suggest()
        ...     reward = evaluate(config)
        ...     optimizer.observe(reward)
        >>> print(optimizer.best_config)
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        seed: int | None = 42,
        n_min_rewarded: int = 1,
        max_nb_pending_per_unrewarded_arm: int = 20,
        n_startup_trials: int = 10,
        switch_strategy: Literal["beta", "delayed"] = "beta",
        beta: float = 0.8,
        memory: Memory | None = None,
        min_arms_for_fit: int = 10,
        n_candidates: int = 100,
        kappa: float = 1.0,
        n_estimators: int = 4,
        max_num_rows: int | None = 200,
        refit_every: int = 10,
        model_type: str = "regression",
        tabfm_model: Any | None = None,
        tabfm_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize TabFMIMABO.

        Args:
            search_space: Dictionary defining the search space (same format
                as :class:`IMABO`).
            seed: Random seed for reproducibility.
            n_min_rewarded: Minimum observations before an arm is considered.
            max_nb_pending_per_unrewarded_arm: Max pending pulls for unrewarded arms.
            n_startup_trials: Number of random initial configurations.
            switch_strategy: "beta" (synchronous) or "delayed" (asynchronous).
            beta: Switching exponent controlling exploration rate.
            memory: Custom memory backend (defaults to InMemoryStorage).
            min_arms_for_fit: Minimum rewarded arms before TabFM is used;
                below this, new configs are sampled uniformly at random.
            n_candidates: Number of random candidates scored per TabFM call.
                TabFM's cost is dominated by ``n_estimators`` (one transformer
                forward pass per ensemble member); scoring more candidates in
                the same batched forward pass is comparatively cheap, so this
                can be large without much extra cost.
            kappa: UCB exploration weight (score = mean + kappa * std).
            n_estimators: TabFM ensemble size. This is the dominant cost
                driver (each member is a separate forward pass) -- kept small
                by default since we only need enough members for a rough
                std estimate, not state-of-the-art point accuracy.
            max_num_rows: Caps how much observation history is used as
                in-context data per TabFM call, bounding predict cost as the
                run grows. ``None`` uses all rewarded arms (unbounded).
            refit_every: Number of `suggest_new` calls served per TabFM
                fit+predict call. Since candidates are cheap to batch, one
                call ranks `n_candidates` and caches the top `refit_every` of
                them; subsequent calls pop from that cache instead of
                re-invoking TabFM, amortizing its cost across several steps
                at the price of a slightly stale ranking.
            model_type: TabFM model type ("regression" is the only mode used).
            tabfm_model: Pre-loaded TabFM model to reuse across optimizer
                instances (avoids reloading weights each time). Defaults to
                loading via :func:`load_tabfm`.
            tabfm_kwargs: Extra keyword arguments forwarded to
                ``TabFMRegressor``.
        """
        super().__init__(
            search_space=search_space,
            seed=seed,
            n_min_rewarded=n_min_rewarded,
            max_nb_pending_per_unrewarded_arm=max_nb_pending_per_unrewarded_arm,
            n_startup_trials=n_startup_trials,
            switch_strategy=switch_strategy,
            beta=beta,
            # `use_tpe` gates whether IMABO.suggest() calls `suggest_new` at
            # all (vs. falling back to a uniform random draw); it must stay
            # True here so our overridden `suggest_new` below actually runs.
            use_tpe=True,
            memory=memory,
        )
        self.min_arms_for_fit = min_arms_for_fit
        self.n_candidates = n_candidates
        self.kappa = kappa
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        self.tabfm_kwargs = tabfm_kwargs or {}
        self._tabfm_model = (
            tabfm_model if tabfm_model is not None else load_tabfm(model_type)
        )
        self._pending_candidates: list[ArmConfig] = []

    def _configs_to_frame(self, configs: list[ArmConfig]) -> pd.DataFrame:
        """Build a DataFrame from configs, tagging categorical columns."""
        df = pd.DataFrame(configs, columns=self.param_names)
        for name in self.param_names:
            if self.param_types[name] == "categorical":
                df[name] = df[name].astype("category")
        return df

    def _fit_surrogate(
        self, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> TabFMRegressor:
        """Fit a fresh TabFM regressor on all observed (config, reward) pairs."""
        configs = [key_to_config(k, self.param_names) for k, _ in rewarded_arms]
        rewards = np.array([stats.mean_reward for _, stats in rewarded_arms])

        X = self._configs_to_frame(configs)
        reg = TabFMRegressor(
            model=self._tabfm_model,
            random_state=self.rng.randint(0, 2**32 - 1),
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            **self.tabfm_kwargs,
        )
        reg.fit(X, rewards)
        return reg

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new configuration using the TabFM oracle (explore).

        TabFM's cost is dominated by the ensemble forward pass (one per
        `n_estimators`), not by how many candidates are scored in it. So
        instead of calling TabFM once per explore step, one call ranks
        `n_candidates` and the top `refit_every` are cached and dispensed
        across subsequent calls, amortizing the expensive part.
        """
        if len(rewarded_arms) < self.min_arms_for_fit:
            return self.generate_random_config()

        if self._pending_candidates:
            return self._pending_candidates.pop(0)

        surrogate = self._fit_surrogate(rewarded_arms)

        candidates = [self.generate_random_config() for _ in range(self.n_candidates)]
        X_candidates = self._configs_to_frame(candidates)

        try:
            preds = np.asarray(surrogate._predict_internal(X_candidates))
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            scores = mean + self.kappa * std
        except Exception:
            # `_predict_internal` is a private API; fall back to a plain
            # (uncertainty-free) mean prediction if it ever breaks.
            scores = np.asarray(surrogate.predict(X_candidates))

        ranked_idx = np.argsort(scores)[::-1]
        ranked = [candidates[i] for i in ranked_idx[: self.refit_every]]

        chosen, self._pending_candidates = ranked[0], ranked[1:]
        return chosen
