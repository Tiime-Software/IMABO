"""IMABO with a TabPFN-3 tabular foundation model as the exploration oracle.

:class:`IMABOTabPFN` keeps IMABO's bandit exploit phase unchanged and drives the
exploration proposal with a TabPFN-3 regressor: at each explore step it fits
TabPFN on the observed ``(config, reward)`` table, scores a pool of random
candidate configs, and proposes the best one under an acquisition rule
(``suggest_method`` -- ``ucb``/``max``/``mean``). A single fit+predict is
amortized over ``refit_every`` candidates so the foundation-model call is shared
across several explore steps.

Ensemble read-out
    TabPFN's public ``predict`` returns one predictive distribution per
    candidate that already averages its ``n_estimators`` internal forward
    passes, hiding the individual members. Instead of that aggregate,
    :meth:`IMABOTabPFN._predict_members` reconstructs each ensemble member's own
    mean prediction from TabPFN's internal per-estimator passes, and the
    acquisition reduces over members with ``mean``/``std``/``max`` -- so
    ``std`` is the true spread across ensemble members and
    ``suggest_method="max"`` is a genuine max over them. This relies on TabPFN
    internal APIs; if a version bump removes them, :meth:`suggest_new` raises
    (there is no distribution/quantile fallback).

Training-table granularity (``fit_granularity``)
    * ``"arm"`` (default): one row per rewarded arm, at its running mean reward.
    * ``"pull"``: one row per individual observation (no per-arm averaging),
      built from a per-arm log of raw rewards recorded in :meth:`observe`. The
      table then grows with the number of pulls, so TabPFN's KV-cache
      (``fit_mode="fit_with_cache"``) is enabled by default and ``max_num_rows``
      caps how many rows are passed in context.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from imabo.memory import (
    ArmStats,
    CurrentState,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.optimizer import IMABO
from imabo.types import ArmConfig, ArmKey


def load_tabpfn(
    model_type: str = "regression",
    device: str = "auto",
    model_path: str = "auto",
    warmup: bool = True,
) -> dict[str, Any]:
    from tabpfn import TabPFNRegressor

    assert model_type == "regression", "IMABOTabPFN only uses the regressor."

    if warmup:
        import pandas as pd

        X = pd.DataFrame({"_w": pd.Categorical([0, 1, 0, 1])})
        y = np.array([0.0, 1.0, 0.0, 1.0])
        reg = TabPFNRegressor(
            n_estimators=1,
            random_state=0,
            model_path=model_path,
            device=device,
            ignore_pretraining_limits=True,
        )
        reg.fit(X, y)
        reg.predict(X)  # force the full download + license path once, up front.

    return {"model_path": model_path, "device": device}


class IMABOTabPFN(IMABO):
    """
    Example:
        >>> optimizer = IMABOTabPFN(
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
        fit_granularity: Literal["arm", "pull"] = "arm",
        fit_mode: str | None = None,
        model_type: str = "regression",
        tabpfn_model: dict[str, Any] | None = None,
        tabpfn_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        model_path: str = "auto",
        suggest_method: Literal["ucb", "max", "mean"] = "ucb",
        on_suggestion: Callable[[ArmConfig, float, float], None] | None = None,
        on_candidates_scored: (
            Callable[[list[ArmConfig], np.ndarray], None] | None
        ) = None,
    ):
        """Initialize IMABOTabPFN.

        Args:
            tabpfn_model: Output of :func:`load_tabpfn` (a ``{"model_path",
                "device"}`` dict), threaded through so every run/budget shares
                the same checkpoint (cached in memory by TabPFN). If ``None``,
                falls back to the ``model_path``/``device`` args below.
            tabpfn_kwargs: Extra keyword arguments forwarded to
                ``TabPFNRegressor`` (e.g. ``inference_precision``,
                ``fit_mode``).
            device: TabPFN inference device ("auto" -> cuda if available else
                cpu). Ignored if ``tabpfn_model`` is given.
            model_path: TabPFN checkpoint path ("auto" downloads/uses the
                default regressor checkpoint). Ignored if ``tabpfn_model`` is
                given.
            n_estimators: TabPFN ensemble size. This is the number of
                per-member predictions the acquisition reduces over with
                ``mean``/``std``/``max`` (see :meth:`_predict_members`).
            fit_granularity: What rows the surrogate is fit on (see the module
                docstring). ``"arm"`` (default) = one row per rewarded arm at
                its mean reward. ``"pull"`` = one row per individual observation
                (no per-arm averaging); this requires per-pull rewards, which
                this class records by overriding :meth:`observe`.
            fit_mode: TabPFN ``fit_mode`` forwarded to every ``TabPFNRegressor``
                (``"low_memory"``/``"fit_preprocessors"``/``"fit_with_cache"``).
                If ``None`` (default), it is left to TabPFN's default for
                ``fit_granularity="arm"`` and set to ``"fit_with_cache"`` (KV
                cache) for ``fit_granularity="pull"``, where the much larger
                per-pull table makes caching the training representation worth
                it. An explicit value here (or ``fit_mode`` in
                ``tabpfn_kwargs``) always wins.
        """
        super().__init__(
            search_space=search_space,
            seed=seed,
            n_min_rewarded=n_min_rewarded,
            max_nb_pending_per_unrewarded_arm=max_nb_pending_per_unrewarded_arm,
            n_startup_trials=n_startup_trials,
            switch_strategy=switch_strategy,
            beta=beta,
            # `use_tpe` must stay True so IMABO.suggest()'s explore branch calls
            # our overridden `suggest_new` (the TabPFN oracle) rather than
            # falling back to a uniform random draw.
            use_tpe=True,
            memory=memory,
        )
        self.min_arms_for_fit = min_arms_for_fit
        self.n_candidates = n_candidates
        self.kappa = kappa
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        self.tabpfn_kwargs = tabpfn_kwargs or {}
        self.fit_granularity = fit_granularity
        # Default the TabPFN fit_mode per granularity: KV cache pays off only
        # for the big per-pull table; leave arm-mode on TabPFN's own default.
        self.fit_mode = fit_mode if fit_mode is not None else (
            "fit_with_cache" if fit_granularity == "pull" else None
        )
        # Per-arm list of individual pull rewards, populated by observe() only
        # when fit_granularity == "pull". Keyed exactly like the memory
        # (config_to_key) so _fit_surrogate can join it to rewarded_arms.
        self._pull_rewards: dict[ArmKey, list[float]] = {}

        cfg = tabpfn_model if tabpfn_model is not None else {
            "model_path": model_path,
            "device": device,
        }
        self._model_path = cfg.get("model_path", model_path)
        self._device = cfg.get("device", device)
        # Kept as a named attribute so the experiment's `_shadow_copy`
        # deepcopy-skip can find it; it is only a small settings dict, so there
        # is nothing heavy to skip -- the name is what matters.
        self._tabpfn_model = cfg

        self._pending_candidates: list[tuple[ArmConfig, float, float]] = []
        self.suggest_method = suggest_method
        self.on_suggestion = on_suggestion
        self.on_candidates_scored = on_candidates_scored

    def observe(self, reward: float) -> None:
        """Record the reward for the last suggested config.

        In ``fit_granularity="pull"`` mode we additionally append this single
        pull's raw reward to the per-arm log (keyed like the memory) before
        delegating to :meth:`IMABO.observe`, which updates the running mean and
        clears ``last_suggested``. This log is what lets ``_fit_surrogate``
        expand each arm into one row per pull. In ``"arm"`` mode this is a
        no-op wrapper, so behaviour is byte-for-byte the base optimizer's.
        """
        if self.fit_granularity == "pull" and self.last_suggested is not None:
            key = config_to_key(self.last_suggested, self.param_names)
            self._pull_rewards.setdefault(key, []).append(float(reward))
        super().observe(reward)

    def _configs_to_frame(self, configs: list[ArmConfig]) -> Any:
        """Build a DataFrame from configs, tagging categorical columns.

        Columns follow ``param_names`` order and categorical params are cast to
        pandas ``category`` dtype so TabPFN treats them as categorical features.
        """
        import pandas as pd

        df = pd.DataFrame(configs, columns=self.param_names)
        for name in self.param_names:
            if self.param_types[name] == "categorical":
                df[name] = df[name].astype("category")
        return df

    def _categorical_indices(self) -> list[int]:
        """0-based indices of categorical columns, in ``param_names`` order."""
        return [
            i
            for i, name in enumerate(self.param_names)
            if self.param_types[name] == "categorical"
        ]

    def _build_training_table(
        self, rewarded_arms: list[tuple[ArmKey, ArmStats]]
    ) -> tuple[list[ArmConfig], np.ndarray]:
        """Assemble the (configs, rewards) TabPFN is fit on for this step.

        ``fit_granularity="arm"`` (default) -> one row per rewarded arm at its
        running mean reward. ``"pull"`` -> one row per individual pull, expanding
        each arm via its ``_pull_rewards`` log so no averaging is done; an arm
        without a logged pull (e.g. rewarded before pull-logging was active)
        falls back to a single mean-reward row.
        """
        if self.fit_granularity == "pull":
            configs: list[ArmConfig] = []
            rewards_list: list[float] = []
            for key, stats in rewarded_arms:
                config = key_to_config(key, self.param_names)
                pulls = self._pull_rewards.get(key)
                if pulls:
                    configs.extend([config] * len(pulls))
                    rewards_list.extend(pulls)
                else:
                    configs.append(config)
                    rewards_list.append(float(stats.mean_reward))
            return configs, np.asarray(rewards_list, dtype=float)

        configs = [key_to_config(k, self.param_names) for k, _ in rewarded_arms]
        rewards = np.array([stats.mean_reward for _, stats in rewarded_arms])
        return configs, rewards

    def _fit_surrogate(self, rewarded_arms: list[tuple[ArmKey, ArmStats]]) -> Any:
        """Fit a fresh TabPFN-3 regressor on observed (config, reward) pairs.

        The training rows come from :meth:`_build_training_table` (one per arm,
        or one per pull when ``fit_granularity="pull"``). If more than
        ``max_num_rows`` rows exist, a seeded random subset is used as in-context
        data, bounding cost as the run grows. In arm mode the explore phase only
        opens O(t**beta) distinct arms, so with beta=0.5 the cap is effectively
        never hit; in pull mode the row count is the total number of pulls, so
        the cap (raise it for this mode) plus the KV-cache ``fit_mode`` are what
        keep the larger fits tractable.
        """
        from tabpfn import TabPFNRegressor

        configs, rewards = self._build_training_table(rewarded_arms)

        if self.max_num_rows is not None and len(configs) > self.max_num_rows:
            idx = self.rng.sample(range(len(configs)), self.max_num_rows)
            configs = [configs[i] for i in idx]
            rewards = rewards[idx]

        # Let an explicit fit_mode (ctor arg -> self.fit_mode, or one already in
        # tabpfn_kwargs) win; otherwise fall back to TabPFN's own default.
        extra_kwargs = dict(self.tabpfn_kwargs)
        if self.fit_mode is not None and "fit_mode" not in extra_kwargs:
            extra_kwargs["fit_mode"] = self.fit_mode

        X = self._configs_to_frame(configs)
        reg = TabPFNRegressor(
            n_estimators=self.n_estimators,
            categorical_features_indices=self._categorical_indices() or None,
            random_state=self.rng.randint(0, 2**32 - 1),
            model_path=self._model_path,
            device=self._device,
            # In arm mode our tables are tiny (~sqrt(T) rewarded arms); in pull
            # mode they can exceed TabPFN's pretraining size -- either way this
            # silences the >1000-samples-on-CPU guard so large per-pull fits go
            # through.
            ignore_pretraining_limits=True,
            **extra_kwargs,
        )
        reg.fit(X, rewards)
        return reg

    def _predict_members(self, surrogate: Any, X: Any) -> np.ndarray | None:
        """Per-ensemble-member mean predictions, shape ``(n_estimators,
        n_candidates)`` in reward units.

        TabPFN's public ``predict`` averages its ``n_estimators`` forward passes
        into one distribution before you see anything, so it exposes no
        per-member point predictions. But internally ``predict`` iterates the
        members (``_iter_forward_executor``) and *sums* their (border-aligned)
        distributions before dividing by ``n_estimators``. We run that same
        iteration ourselves and, for each member, take that member's own
        predictive-distribution mean via the model's bar-distribution criterion.
        Averaging these per-member means reproduces ``predict(output_type=
        "mean")`` exactly (the distribution mean is linear over the member
        mixture), which is the correctness check for this reconstruction.

        With these per-member point predictions, ``suggest_new`` reduces over
        members with ``mean``/``std``/``max`` -- so ``std`` is the true spread
        across ensemble members and ``suggest_method="max"`` is a real max over
        them, no quantile involved.

        Uses TabPFN internal (underscore) APIs. Returns ``None`` (rather than
        raising) if any of them is missing/changed; :meth:`suggest_new` turns
        that into a clear error, since there is no alternative readout.
        """
        try:
            import torch
            from tabpfn.base import ensure_compatible_predict_input_sklearn
            from tabpfn.preprocessing.clean import (
                fix_dtypes,
                process_text_na_dataframe,
            )
            from tabpfn.preprocessing.datamodel import FeatureModality
            from tabpfn.regressor import _logits_to_output
            from tabpfn.utils import translate_probs_across_borders

            # Mirror predict()'s X preprocessing before feeding the executor:
            # (1) reconcile columns/dtypes to the fitted schema, (2) fix dtypes,
            # (3) handle text/NA. Skipping (1) breaks on named-column frames.
            X = ensure_compatible_predict_input_sklearn(X, surrogate)
            cat_idx = surrogate.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            )
            Xp = fix_dtypes(X, cat_indices=cat_idx)
            Xp = process_text_na_dataframe(
                Xp,
                ord_encoder=getattr(surrogate, "ordinal_encoder_", None),
                passthrough_inf=surrogate.get_inference_config().PASSTHROUGH_INF,
            )

            znorm = surrogate.znorm_space_bardist_
            raw = surrogate.raw_space_bardist_
            member_means: list[np.ndarray] = []
            for borders_t, output in surrogate._iter_forward_executor(
                Xp, use_inference_mode=True
            ):
                # Align this member's distribution to the common border grid
                # (exactly what predict() does before accumulating), then read
                # off this single member's mean in raw reward units.
                transformed = translate_probs_across_borders(
                    output,
                    frm=torch.as_tensor(borders_t, device=output.device),
                    to=znorm.borders.to(output.device),
                )
                m = _logits_to_output(
                    output_type="mean",
                    logits=transformed.log(),
                    criterion=raw,
                    quantiles=[],
                )
                member_means.append(np.asarray(m, dtype=float))

            if not member_means:
                return None
            return np.stack(member_means, axis=0)
        except Exception:
            return None

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new configuration using the TabPFN-3 oracle (explore).

        One TabPFN fit+predict ranks ``n_candidates`` random configs and caches
        the top ``refit_every`` of them, so the expensive foundation-model call
        is amortized across several explore steps.
        """
        if len(rewarded_arms) < self.min_arms_for_fit:
            return self.generate_random_config()

        if self._pending_candidates:
            config, mean_pred, max_pred = self._pending_candidates.pop(0)
            if self.on_suggestion is not None:
                self.on_suggestion(config, mean_pred, max_pred)
            return config

        surrogate = self._fit_surrogate(rewarded_arms)

        candidates = [self.generate_random_config() for _ in range(self.n_candidates)]
        X_candidates = self._configs_to_frame(candidates)

        # Reduce over ensemble members with mean/std/max: std is the true spread
        # across members and suggest_method="max" is a real max over them (not a
        # quantile). Requires TabPFN's internal per-member outputs.
        preds = self._predict_members(surrogate, X_candidates)
        if preds is None:
            raise RuntimeError(
                "IMABOTabPFN could not read TabPFN's per-ensemble-member "
                "predictions -- its internal API (_iter_forward_executor et al.) "
                "may have changed in this TabPFN version. The acquisition needs "
                "per-member outputs to compute the ensemble mean/std/max."
            )
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        optimistic = preds.max(axis=0)
        if self.suggest_method == "ucb":
            scores = mean + self.kappa * std
        elif self.suggest_method == "max":
            scores = optimistic
        elif self.suggest_method == "mean":
            scores = mean
        else:
            raise ValueError(f"Invalid suggest_method: {self.suggest_method}")

        # Everything is already in reward units (TabPFN's predictive
        # distribution lives in raw target space), so no inverse transform is
        # applied before logging/ranking.
        if self.on_candidates_scored is not None:
            self.on_candidates_scored(candidates, mean)

        ranked_idx = np.argsort(scores)[::-1]
        ranked = [
            (candidates[i], float(mean[i]), float(optimistic[i]))
            for i in ranked_idx[: self.refit_every]
        ]

        (chosen, chosen_mean, chosen_max), self._pending_candidates = (
            ranked[0],
            ranked[1:],
        )
        if self.on_suggestion is not None:
            self.on_suggestion(chosen, chosen_mean, chosen_max)
        return chosen
