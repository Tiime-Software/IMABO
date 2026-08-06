"""IMABO with a TabPFN-3 tabular foundation model as the exploration oracle.

:class:`IMABOTabPFN` keeps IMABO's bandit exploit phase unchanged and drives the
exploration proposal with a TabPFN-3 regressor: at each explore step it fits
TabPFN on the observed ``(config, reward)`` table, scores a pool of random
candidate configs, and proposes the one maximizing an upper confidence bound
read from TabPFN's predictive distribution (``acquisition`` -- moment-based
``"ucb"`` or ``"quantile"``). A single fit+predict is amortized over
``refit_every`` candidates so the foundation-model call is shared across
several explore steps.

Acquisition (two UCB readouts of the calibrated predictive posterior)
    TabPFN ensembles ``n_estimators`` estimators, each conditioned on the same
    table under a different feature ordering/preprocessing, and averages them
    into a single calibrated predictive distribution per candidate
    (``predict(output_type="full")``). Two ways to score a candidate from it:

    * ``acquisition="quantile"`` (default): the ``quantile`` level of the
      distribution, ``F^-1(quantile)`` -- the deterministic quantile form of
      UCB (GIT-BO, arXiv:2505.20685), ranking on the 0.99 quantile by
      default. Off-Gaussian, the quantile readout is insensitive to skew and
      heavy tails of the predicted density. See
      :meth:`IMABOTabPFN._predict_mean_quantile`.
    * ``acquisition="ucb"``: ``mean + kappa * std``, where ``mean`` is the
      distribution mean and ``std`` its exact standard deviation (the mixture
      std, which already combines each estimator's own spread and the
      disagreement between estimators). See
      :meth:`IMABOTabPFN._predict_mean_std`.

    ``quantile`` is the single exploration knob for both rules: the UCB
    weight is derived from it assuming a Gaussian posterior,
    ``kappa = Phi^-1(quantile)`` (see :func:`quantile_to_kappa`), for which
    the two rules coincide exactly -- e.g. ``q=0.841 -> kappa~=1``,
    ``q=0.975 -> kappa~=1.96``, ``q=0.99 -> kappa~=2.33``.

Training-table granularity (``fit_granularity``)
    * ``"arm"`` (default): one row per rewarded arm, at its running mean reward.
    * ``"pull"``: one row per individual observation (no per-arm averaging),
      built from a per-arm log of raw rewards recorded in :meth:`observe`. The
      table then grows with the number of pulls, so TabPFN's KV-cache
      (``fit_mode="fit_with_cache"``) is enabled by default and ``max_num_rows``
      caps how many rows are passed in context.
"""

from __future__ import annotations

import contextlib
import threading
from statistics import NormalDist
from typing import Any, Callable, Literal

import numpy as np

from imabo.memory import (
    ArmStats,
    CurrentState,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.mutation import ValueSampler, best_config, local_value_sampler, mutate_value
from imabo.optimizer import IMABO
from imabo.types import ArmConfig, ArmKey

# Concurrent TabPFN fit/predict from multiple threads hard-crashes the whole
# process (no Python traceback) on Apple-silicon Metal/MPS, so on that device
# the oracle's fit+score block is serialized across threads. On CUDA/CPU the
# calls are left fully parallel.
_TABPFN_MPS_LOCK = threading.Lock()


def _tabpfn_serialization_lock() -> Any:
    try:
        import torch

        if torch.backends.mps.is_available():
            return _TABPFN_MPS_LOCK
    except Exception:
        pass
    return contextlib.nullcontext()


def kappa_to_quantile(kappa: float) -> float:
    """Quantile level equivalent to ``mean + kappa * std`` under normality.

    ``Phi(kappa)`` (standard normal CDF), clamped away from 0/1 so the
    bar-distribution ``icdf`` always gets a valid interior probability.
    """
    q = NormalDist().cdf(kappa)
    return min(max(q, 1e-6), 1.0 - 1e-6)


def quantile_to_kappa(quantile: float) -> float:
    """UCB weight equivalent to ranking on ``quantile`` under normality.

    ``Phi^-1(quantile)`` (standard normal inverse CDF) -- e.g. 0.841 -> ~1,
    0.975 -> ~1.96, 0.99 -> ~2.33.
    """
    return NormalDist().inv_cdf(min(max(quantile, 1e-6), 1.0 - 1e-6))


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
        beta: float = 0.5,
        memory: Memory | None = None,
        min_arms_for_fit: int = 10,
        n_candidates: int = 100,
        acquisition: Literal["ucb", "quantile"] = "quantile",
        quantile: float = 0.975,
        n_estimators: int = 4,
        max_num_rows: int | None = 200,
        refit_every: int = 1,
        candidate_source: Literal["uniform", "mutation"] = "mutation",
        candidate_uniform_frac: float = 0.1,
        mutation_scale: float | None = 0.1,
        filter_open_candidates: bool = True,
        fit_granularity: Literal["arm", "pull"] = "arm",
        fit_mode: str | None = None,
        model_type: str = "regression",
        tabpfn_model: dict[str, Any] | None = None,
        tabpfn_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        model_path: str = "auto",
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
            acquisition: How candidates are scored from TabPFN's predictive
                distribution (see the module docstring): ``"quantile"``
                (default) ranks on the ``quantile`` level of the distribution;
                ``"ucb"`` ranks on ``mean + kappa * std`` with
                ``kappa = Phi^-1(quantile)``.
            quantile: The single exploration knob, in (0, 1), shared by both
                acquisitions: the level ranked on when
                ``acquisition="quantile"``, and (via the normality conversion
                ``kappa = Phi^-1(quantile)``, see :func:`quantile_to_kappa`)
                the UCB weight for ``acquisition="ucb"``. The default 0.99 is
                the best and most seed-stable setting in our RF/HPO comparison
                (GIT-BO's ablations favor the nearby 0.95-0.975).
            n_estimators: TabPFN ensemble size (the number of estimators
                averaged into the calibrated predictive posterior).
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
        if acquisition not in ("ucb", "quantile"):
            raise ValueError(f"Invalid acquisition: {acquisition!r}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.min_arms_for_fit = min_arms_for_fit
        self.n_candidates = n_candidates
        self.acquisition = acquisition
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        if candidate_source not in ("uniform", "mutation"):
            raise ValueError(f"Invalid candidate_source: {candidate_source!r}")
        if not 0.0 <= candidate_uniform_frac <= 1.0:
            raise ValueError(
                f"candidate_uniform_frac must be in [0, 1], got {candidate_uniform_frac}"
            )
        self.candidate_source = candidate_source
        self.candidate_uniform_frac = candidate_uniform_frac
        self.mutation_scale = mutation_scale
        self.filter_open_candidates = filter_open_candidates
        self.tabpfn_kwargs = tabpfn_kwargs or {}
        self.fit_granularity = fit_granularity
        # Default the TabPFN fit_mode per granularity: KV cache pays off only
        # for the big per-pull table; leave arm-mode on TabPFN's own default.
        self.fit_mode = (
            fit_mode
            if fit_mode is not None
            else ("fit_with_cache" if fit_granularity == "pull" else None)
        )
        # Per-arm list of individual pull rewards, populated by observe() only
        # when fit_granularity == "pull". Keyed exactly like the memory
        # (config_to_key) so _fit_surrogate can join it to rewarded_arms.
        self._pull_rewards: dict[ArmKey, list[float]] = {}

        cfg = (
            tabpfn_model
            if tabpfn_model is not None
            else {
                "model_path": model_path,
                "device": device,
            }
        )
        self._model_path = cfg.get("model_path", model_path)
        self._device = cfg.get("device", device)
        # Kept as a named attribute so the experiment's `_shadow_copy`
        # deepcopy-skip can find it; it is only a small settings dict, so there
        # is nothing heavy to skip -- the name is what matters.
        self._tabpfn_model = cfg

        self._pending_candidates: list[tuple[ArmConfig, float, float]] = []
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

    def _predict_mean_std(
        self, surrogate: Any, X: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of TabPFN's calibrated predictive posterior per candidate.

        Uses the public ``predict(output_type="full")``, which returns the
        aggregated predictive distribution (the ``n_estimators`` estimators
        already averaged into one). ``mean`` is that distribution's mean and
        ``std`` its exact standard deviation, both in reward units, shape
        ``(n_candidates,)``.
        """
        full = surrogate.predict(X, output_type="full")
        logits, criterion = full["logits"], full["criterion"]
        mean = criterion.mean(logits).cpu().detach().numpy().astype(float)
        var = criterion.variance(logits).cpu().detach().numpy().astype(float)
        return mean, np.sqrt(np.clip(var, 0.0, None))

    def _predict_mean_quantile(
        self, surrogate: Any, X: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and ``self.quantile`` level of TabPFN's predictive posterior.

        Same aggregated distribution as :meth:`_predict_mean_std`
        (``predict(output_type="full")``); the quantile is exact for TabPFN's
        histogram posterior (piecewise-linear CDF), computed by
        ``criterion.icdf``. Both arrays are in reward units, shape
        ``(n_candidates,)``.
        """
        q = self.quantile
        full = surrogate.predict(X, output_type="full")
        logits, criterion = full["logits"], full["criterion"]
        mean = criterion.mean(logits).cpu().detach().numpy().astype(float)
        upper = criterion.icdf(logits, q).cpu().detach().numpy().astype(float)
        return mean, upper

    def _sample_candidates(
        self,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
    ) -> list[ArmConfig]:
        """Build the ``n_candidates`` pool the surrogate will rank.

        ``"uniform"`` draws the whole pool uniformly -- the surrogate then ranks a
        global random pool, which in a large space is the binding constraint:
        none of its members is near a good arm to begin with. ``"mutation"``
        keeps a ``candidate_uniform_frac`` share uniform (so the pool never loses
        access to unexplored regions) and makes the rest single-coordinate
        mutants of the incumbent. Measured on the RF grid, the mutation pool is
        worth -98.3 +- 24.4 against the uniform one.

        Falls back to the uniform pool while no arm has a reward yet.
        """
        if self.candidate_source == "uniform" or not rewarded_arms:
            return [self.generate_random_config() for _ in range(self.n_candidates)]

        n_uniform = min(
            self.n_candidates,
            int(round(self.candidate_uniform_frac * self.n_candidates)),
        )
        candidates = [self.generate_random_config() for _ in range(n_uniform)]
        parent = best_config(rewarded_arms, self.param_names)
        value_sampler = (
            local_value_sampler(self.distributions, self.rng, self.mutation_scale)
            if self.mutation_scale is not None
            else None
        )
        for _ in range(self.n_candidates - n_uniform):
            candidates.append(self._mutate_config(parent, value_sampler))
        return candidates

    def _mutate_config(
        self, config: ArmConfig, value_sampler: ValueSampler | None = None
    ) -> ArmConfig:
        """Resample one uniformly-chosen parameter of ``config``.

        Every other coordinate is inherited from the parent arm; the new value
        comes from :func:`imabo.mutation.mutate_value` -- a uniform draw over that
        parameter's domain excluding the parent's own value, or ``value_sampler``'s
        local step when ``mutation_scale`` is set.
        """
        name = self.rng.choice(self.param_names)
        return {
            **config,
            name: mutate_value(
                name, config[name], self.distributions, self.rng, value_sampler
            ),
        }

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new configuration using the TabPFN-3 oracle (explore).

        One TabPFN fit+predict ranks ``n_candidates`` random configs by the
        acquisition (``mean + kappa * std``, or the ``quantile`` level of the
        predictive distribution) and caches the top ``refit_every`` of them, so
        the expensive foundation-model call is amortized across several explore
        steps.
        """
        if len(rewarded_arms) < self.min_arms_for_fit:
            return self.generate_random_config()

        if self._pending_candidates:
            config, mean_pred, ucb_pred = self._pending_candidates.pop(0)
            if self.on_suggestion is not None:
                self.on_suggestion(config, mean_pred, ucb_pred)
            return config

        candidates = self._sample_candidates(rewarded_arms)

        # Deduplicate before scoring: a mutation pool draws each candidate
        # independently, so on a finite space collisions are common (90 draws
        # from a 25-configuration neighbourhood collapse to ~23 distinct).
        # Scoring duplicates wastes TabPFN rows and fills the refit_every cache
        # with repeats of one config instead of distinct runners-up.
        deduped: dict[ArmKey, ArmConfig] = {}
        for candidate in candidates:
            deduped.setdefault(config_to_key(candidate, self.param_names), candidate)

        # Then drop candidates that are already open arms. The pool exists so the
        # surrogate can pick the next arm to OPEN; a candidate already in memory
        # cannot do that. This is not rescuing an exhausted pool -- the pool almost
        # always holds novel candidates -- it OVERRIDES the acquisition, which on a
        # finite grid reliably ranks an already-open neighbour of the incumbent
        # above any unopened candidate, stalling |arms| below the t**beta target.
        novel = (
            [c for k, c in deduped.items() if k not in state.arms]
            if self.filter_open_candidates
            else list(deduped.values())
        )
        if not novel:
            novel = [self.generate_random_config() for _ in range(self.n_candidates)]
            fresh: dict[ArmKey, ArmConfig] = {}
            for candidate in novel:
                fresh.setdefault(config_to_key(candidate, self.param_names), candidate)
            novel = [c for k, c in fresh.items() if k not in state.arms] or list(
                deduped.values()
            )
        candidates = novel
        X_candidates = self._configs_to_frame(candidates)

        with _tabpfn_serialization_lock():
            surrogate = self._fit_surrogate(rewarded_arms)
            if self.acquisition == "quantile":
                # Quantile form of UCB: rank on F^-1(quantile) of the posterior.
                mean, scores = self._predict_mean_quantile(surrogate, X_candidates)
            else:
                # Moment UCB at the normality-equivalent exploration weight.
                kappa = quantile_to_kappa(self.quantile)
                mean, std = self._predict_mean_std(surrogate, X_candidates)
                scores = mean + kappa * std

        if self.on_candidates_scored is not None:
            self.on_candidates_scored(candidates, mean)

        ranked_idx = np.argsort(scores)[::-1]
        ranked = [
            (candidates[i], float(mean[i]), float(scores[i]))
            for i in ranked_idx[: self.refit_every]
        ]

        (chosen, chosen_mean, chosen_ucb), self._pending_candidates = (
            ranked[0],
            ranked[1:],
        )
        if self.on_suggestion is not None:
            self.on_suggestion(chosen, chosen_mean, chosen_ucb)
        return chosen
