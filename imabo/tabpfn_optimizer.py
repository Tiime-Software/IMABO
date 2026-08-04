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

Candidate pool (``candidate_source``)
    What the oracle actually gets to rank. TabPFN only ever picks the argmax of
    the pool it is handed, so in a large space a uniform pool is the binding
    constraint: none of its members is near a good arm to begin with.

    * ``"uniform"`` (default): ``n_candidates`` configs drawn uniformly from the
      search space, i.e. the surrogate ranks a global random pool.
    * ``"mutation"``: an evolutionary pool built around the current population.
      A ``candidate_uniform_frac`` share stays uniform (so the pool never loses
      access to unexplored regions), and each remaining candidate is a *parent*
      config with one uniformly-chosen parameter resampled uniformly (see
      :meth:`IMABOTabPFN._mutate_config`). ``parent_rule`` picks that parent --
      ``"softmax"`` draws one per candidate from the rewarded population, so the
      pool spans several arms, while ``"best"``, ``"last_proposal"`` and
      ``"moss"`` make the whole pool one point's neighbourhood (see
      :func:`imabo.mutation.parent_config`). The
      temperature is dispersion-adapted,
      ``T = candidate_temperature * std(mean_rewards)``, so the softmax runs on
      z-scores: selection pressure is set by how many standard deviations apart
      the arms are, not by the reward scale, and stays stable as the
      population's spread changes over a run (see
      :meth:`IMABOTabPFN._parent_probabilities`).
    * ``"mutation_tpe"``: the same pool, except the mutated coordinate's new
      value is drawn by a *univariate TPE* over that one parameter instead of
      uniformly -- the good/bad split is IMABO's own
      (:meth:`IMABO.tpe_split`), and the value is the EI-argmax of
      ``n_ei_candidates`` draws from the 1-D ``l`` density
      (:func:`imabo.tpe.univariate_tpe_values`). So the parent says *where* in
      the space to look, TPE says *which value* that coordinate should take, and
      TabPFN still ranks the resulting pool. See
      :meth:`IMABOTabPFN._tpe_value_sampler`.
    * ``"tpe"``: no parent and no per-coordinate mutation -- the whole pool is
      drawn from TPE's own proposal density, i.e. ``n_candidates`` joint samples
      from the good-arm Parzen mixture ``l`` (:func:`imabo.tpe.tpe_sample_configs`,
      the sampling half of the IMOSS-TPE oracle). TabPFN's acquisition then
      replaces TPE's ``l/g`` expected improvement as the rule that picks the
      winner: TPE proposes, TabPFN selects. See
      :meth:`IMABOTabPFN._tpe_pool`.

    * ``"mix"``: half the pool mutants of the population (as ``"mutation"``),
      half joint TPE draws (as ``"tpe"``), deduplicated and then topped up with
      uniform draws to a full ``n_candidates`` of DISTINCT configs. The two
      halves fail in opposite ways -- the mutation half saturates once the
      parent's neighbourhood is open, the TPE half concentrates wherever ``l``
      is and can miss a good arm one coordinate away -- so the mix asks whether
      the surrogate does better when handed both kinds of candidate at once.
      Unlike the sources above it does NOT reserve ``candidate_uniform_frac``
      up front; the uniform share is whatever the top-up needs, which is what
      keeps the pool full however much the two halves collide.

    Every non-uniform source except ``"mix"`` keeps the same
    ``candidate_uniform_frac`` uniform share, so the variants differ in exactly
    one thing: how the remaining candidates are generated.

Already-open candidates (``filter_open_candidates``)
    The pool exists so the surrogate can pick the next arm to OPEN, so by
    default a candidate already in memory is dropped before scoring. This is not
    a rescue for an exhausted pool -- the pool virtually always contains novel
    configs -- it OVERRIDES the acquisition, which on a finite grid reliably
    ranks an already-open neighbour of the incumbent above any unopened
    candidate. Measured on the RF grid with ``parent_rule="best"``: without the
    filter, runs reached 43-71 of the 71 arms the ``|arms| < t**beta`` schedule
    asks for and cumulative regret was 464.3; with it, every run reached 71 and
    regret rose to 536.9 (paired +72.5 +- 23.5). Set ``False`` to let the
    surrogate re-propose an open arm and spend the round re-pulling it.
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
from imabo.mutation import (
    ParentRule,
    ValueSampler,
    local_value_sampler,
    mutate_value,
    parent_config,
    parent_probabilities,
    tpe_value_sampler,
)
from imabo.optimizer import IMABO
from imabo.tpe import tpe_sample_configs
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
        beta: float = 0.8,
        memory: Memory | None = None,
        min_arms_for_fit: int = 10,
        n_candidates: int = 100,
        acquisition: Literal["ucb", "quantile"] = "quantile",
        quantile: float = 0.99,
        n_estimators: int = 4,
        max_num_rows: int | None = 200,
        refit_every: int = 10,
        fit_granularity: Literal["arm", "pull"] = "arm",
        candidate_source: Literal[
            "uniform", "mutation", "mutation_tpe", "tpe", "mix"
        ] = "uniform",
        candidate_uniform_frac: float = 0.1,
        candidate_tpe_frac: float = 0.0,
        candidate_mix_mutation_frac: float = 0.5,
        mutation_size: int | Literal["geometric"] = 1,
        candidate_topup: Literal["uniform", "runner_up"] = "uniform",
        mutation_scale: float | None = None,
        filter_open_candidates: bool = True,
        candidate_temperature: float = 1.0,
        parent_rule: ParentRule = "softmax",
        tpe_value_pick: Literal["ei_argmax", "sample"] = "ei_argmax",
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
            candidate_source: How the candidate pool handed to the surrogate is
                built (see the module docstring). ``"uniform"`` (default) draws
                all ``n_candidates`` uniformly from the search space;
                ``"mutation"`` draws a ``candidate_uniform_frac`` share
                uniformly and builds the rest by mutating one parameter of a
                parent arm sampled from the rewarded population by
                softmax(mean reward); ``"mutation_tpe"`` is the same but draws
                the mutated coordinate's value from a univariate TPE over that
                parameter rather than uniformly; ``"tpe"`` drops parents
                entirely and draws the whole pool from TPE's own proposal
                density, leaving TabPFN to pick the winner in place of TPE's
                expected improvement.
            filter_open_candidates: Drop candidates already in memory before
                scoring, so the proposal always opens a new arm (default). See
                the module docstring -- this overrides the acquisition rather
                than filling an empty pool, and it is not free.
            mutation_scale: Take a LOCAL Gaussian step of this fraction of an
                axis' width instead of resampling the coordinate uniformly (see
                :func:`imabo.mutation.local_value_sampler`). ``None`` (default)
                keeps the uniform redraw, which is a neighbour on a finite axis
                but a global move on a continuous one.
            candidate_topup: What fills the mix pool back to ``n_candidates``
                after deduplication -- ``"uniform"`` draws, or ``"runner_up"``
                mutants of the population's next-best arms. See :meth:`_top_up`.
            mutation_size: How many coordinates one mutant changes: an int, or
                ``"geometric"`` for a 1 + Geometric(0.5) draw capped at the
                dimension. Larger sizes enlarge the reachable neighbourhood, which
                is what stops a mutation pool saturating -- see
                :meth:`_mutate_config`.
            candidate_mix_mutation_frac: With ``candidate_source="mix"``, the
                share of the non-uniform slots that are mutants rather than
                joint-TPE draws (0.5 = the even split). Set
                ``candidate_uniform_frac=0`` alongside it to make the uniform
                share purely the post-dedup top-up.
            candidate_tpe_frac: With a mutation ``candidate_source``, the share
                of the pool drawn from the joint multivariate TPE instead of
                being a mutant, carved out of the mutation slots (the uniform
                share is untouched). 0.0 (default) is the pure mutation pool.
            candidate_uniform_frac: With a mutation ``candidate_source``, the
                fraction of the pool still drawn uniformly at random (the pool's
                exploration floor: it keeps regions far from every known arm
                reachable). ``0.0`` makes the pool purely local, ``1.0``
                reproduces ``candidate_source="uniform"``.
            parent_rule: With a mutation ``candidate_source``, which config the
                mutants are built from -- ``"softmax"`` (default) draws one per
                candidate so the pool spans several arms, the others make the pool
                a single config's neighbourhood. See
                :func:`imabo.mutation.parent_config`. Ignored by
                ``candidate_source`` ``"uniform"`` and ``"tpe"``, which use no
                parent.
            tpe_value_pick: With ``candidate_source="mutation_tpe"``, which of the
                ``n_ei_candidates`` TPE draws each mutant takes for its coordinate:
                the EI-argmax (default, and deterministic -- so every mutant on a
                coordinate gets the SAME value, collapsing the pool to ``d``
                distinct configs) or a uniform ``"sample"`` among the draws, which
                keeps the pool diverse at some cost in per-candidate quality. See
                :func:`imabo.mutation.tpe_value_sampler`.
            candidate_temperature: With a mutation ``candidate_source``, the
                parent-selection temperature *in units of the population's own
                score dispersion*: ``T = candidate_temperature *
                std(mean_rewards)`` (see :meth:`_parent_probabilities`). Must be
                > 0. Small values (-> 0) select the best arm greedily, 1.0 gives
                one e-fold of selection weight per standard deviation of reward,
                and large values flatten to uniform over the population.
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
        if candidate_source not in ("uniform", "mutation", "mutation_tpe", "tpe", "mix"):
            raise ValueError(f"Invalid candidate_source: {candidate_source!r}")
        if candidate_topup not in ("uniform", "runner_up"):
            raise ValueError(f"Invalid candidate_topup: {candidate_topup!r}")
        if mutation_size != "geometric" and int(mutation_size) < 1:
            raise ValueError(
                f"mutation_size must be >= 1 or 'geometric', got {mutation_size}"
            )
        if not 0.0 <= candidate_mix_mutation_frac <= 1.0:
            raise ValueError(
                "candidate_mix_mutation_frac must be in [0, 1], got "
                f"{candidate_mix_mutation_frac}"
            )
        if not 0.0 <= candidate_tpe_frac <= 1.0:
            raise ValueError(
                f"candidate_tpe_frac must be in [0, 1], got {candidate_tpe_frac}"
            )
        if candidate_uniform_frac + candidate_tpe_frac > 1.0:
            raise ValueError(
                "candidate_uniform_frac + candidate_tpe_frac must be <= 1, got "
                f"{candidate_uniform_frac} + {candidate_tpe_frac}"
            )
        if not 0.0 <= candidate_uniform_frac <= 1.0:
            raise ValueError(
                f"candidate_uniform_frac must be in [0, 1], got {candidate_uniform_frac}"
            )
        if parent_rule not in ("best", "softmax", "last_proposal", "moss"):
            raise ValueError(f"Invalid parent_rule: {parent_rule!r}")
        if candidate_temperature <= 0.0:
            raise ValueError(
                f"candidate_temperature must be > 0, got {candidate_temperature}"
            )
        self.min_arms_for_fit = min_arms_for_fit
        self.n_candidates = n_candidates
        self.acquisition = acquisition
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.refit_every = refit_every
        self.tabpfn_kwargs = tabpfn_kwargs or {}
        self.fit_granularity = fit_granularity
        self.candidate_source = candidate_source
        self.candidate_uniform_frac = candidate_uniform_frac
        self.candidate_tpe_frac = candidate_tpe_frac
        self.candidate_mix_mutation_frac = candidate_mix_mutation_frac
        self.mutation_size = mutation_size
        self.candidate_topup = candidate_topup
        self.mutation_scale = mutation_scale
        self.filter_open_candidates = filter_open_candidates
        self.candidate_temperature = candidate_temperature
        self.parent_rule = parent_rule
        self.tpe_value_pick = tpe_value_pick
        # This oracle's own previous proposal, for parent_rule="last_proposal";
        # IMABO.last_suggested is cleared by observe() and so is always None here.
        self._last_proposal: ArmConfig | None = None
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

    def _parent_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Parent-selection distribution: softmax at ``candidate_temperature``
        times the population's own score dispersion -- see
        :func:`imabo.mutation.parent_probabilities`."""
        return parent_probabilities(scores, self.candidate_temperature)

    def _tpe_value_sampler(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> ValueSampler:
        """Build the ``"mutation_tpe"`` mutation operator: a univariate-TPE draw
        for whichever coordinate is being mutated
        (:func:`imabo.mutation.tpe_value_sampler`).

        The good/bad split is IMABO's own :meth:`IMABO.tpe_split` (arms ranked by
        the MOSS-anytime index, top ``gamma_func`` fraction "good"), i.e. exactly
        the split the IMOSS-TPE oracle proposes from -- only here the Parzen pair
        is fit on the single parameter being mutated.
        """
        good, bad = self.tpe_split(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )
        return tpe_value_sampler(
            good_configs=[key_to_config(k, self.param_names) for k, _ in good],
            bad_configs=[key_to_config(k, self.param_names) for k, _ in bad],
            distributions=self.distributions,
            n_candidates=self.n_ei_candidates,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            weights_func=self.weights_func,
            pick=self.tpe_value_pick,
        )

    # Continuation probability of mutation_size="geometric": each coordinate
    # beyond the first is added with this probability, so E[k] = 1.9 on a
    # 4-dimensional space and half the pool stays a single-coordinate move.
    _GEOMETRIC_P = 0.5

    def _mutation_size(self) -> int:
        """How many coordinates this mutant changes, per ``mutation_size``."""
        d = len(self.param_names)
        if self.mutation_size != "geometric":
            return min(int(self.mutation_size), d)
        k = 1
        while k < d and self.rng.random() < self._GEOMETRIC_P:
            k += 1
        return k

    def _mutate_config(
        self, config: ArmConfig, value_sampler: ValueSampler | None = None
    ) -> ArmConfig:
        """Resample ``mutation_size`` uniformly-chosen parameters of ``config``.

        Every other coordinate is inherited from the parent arm, and each new
        value comes from :func:`imabo.mutation.mutate_value` -- a uniform draw over
        that parameter's domain excluding the parent's own value, or
        ``value_sampler``'s draw when one is given (``"mutation_tpe"``, see
        :meth:`_tpe_value_sampler`). Which coordinates get mutated is uniform
        either way.

        ``mutation_size=1`` (default) is a single-coordinate move, which confines
        the whole pool to a neighbourhood of ``sum_i(|A_i| - 1)`` configurations --
        25 on the RF grid, so 90 independent draws collapse to ~23 distinct and
        saturate once that many arms are open. Mutating two coordinates reaches
        ``sum_{i<j}(|A_i| - 1)(|A_j| - 1)`` instead, 224 there, which is what makes
        a larger size worth trying for a pool the surrogate RANKS (as opposed to a
        bandit that must credit each coordinate separately, where the extra
        coordinates cost attribution -- see :class:`imabo.coord_ucb.IMABOCoordUCB`).
        """
        mutant = dict(config)
        k = self._mutation_size()
        for name in self.rng.sample(self.param_names, k):
            mutant[name] = mutate_value(
                name, config[name], self.distributions, self.rng, value_sampler
            )
        return mutant

    def _tpe_pool(
        self,
        n_samples: int,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> list[ArmConfig]:
        """``n_samples`` configs drawn from TPE's proposal density ``l``.

        The good/bad split is IMABO's own (:meth:`IMABO.tpe_split`) and the draws
        come from the good-arm Parzen mixture at this optimizer's
        ``multivariate`` setting -- i.e. the IMOSS-TPE oracle's *sampling* step,
        run ``n_samples`` times instead of once. Its ``l/g`` expected-improvement
        ranking is deliberately not applied: that is the job TabPFN's acquisition
        takes over in ``candidate_source="tpe"``.
        """
        good, _ = self.tpe_split(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )
        return tpe_sample_configs(
            good_configs=[key_to_config(k, self.param_names) for k, _ in good],
            param_names=self.param_names,
            distributions=self.distributions,
            n_samples=n_samples,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            multivariate=self.multivariate,
            weights_func=self.weights_func,
        )

    def _sample_candidates(
        self,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        state: CurrentState | None = None,
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> list[ArmConfig]:
        """Build the ``n_candidates`` pool the surrogate will rank.

        Dispatches on ``candidate_source`` (see the module docstring):
        ``"uniform"`` (default) draws the whole pool uniformly; the others keep a
        ``round(candidate_uniform_frac * n_candidates)`` uniform share and fill
        the rest with mutants of population arms (``"mutation"``,
        ``"mutation_tpe"``) or with draws from TPE's proposal density
        (``"tpe"``).

        Falls back to the uniform pool when the population cannot support the
        requested source: no rewarded arm at all, or fewer than two of them for
        the TPE-based sources (a good/bad split needs both sides non-empty). The
        latter cannot happen through :meth:`suggest_new`, which only reaches here
        once ``min_arms_for_fit`` arms are rewarded.
        """
        tpe_based = (
            self.candidate_source in ("mutation_tpe", "tpe", "mix")
            or self.candidate_tpe_frac > 0.0
        )
        if (
            self.candidate_source == "uniform"
            or not rewarded_arms
            or (tpe_based and (state is None or len(rewarded_arms) < 2))
        ):
            return [self.generate_random_config() for _ in range(self.n_candidates)]

        n_uniform = min(
            self.n_candidates,
            int(round(self.candidate_uniform_frac * self.n_candidates)),
        )
        candidates = [self.generate_random_config() for _ in range(n_uniform)]
        n_rest = self.n_candidates - n_uniform

        # An optional joint-TPE share carved out of the mutation slots, so the
        # pool is uniform / joint-TPE / mutants rather than uniform / mutants.
        # A mutation pool can only ever reach one coordinate from its parent --
        # sum_i(|A_i| - 1) configs, 25 on the RF grid -- so a handful of joint
        # draws is the cheapest way to put something MORE than one coordinate
        # away in front of the surrogate without giving up the neighbourhood.
        n_tpe = 0
        if self.candidate_tpe_frac > 0.0 and self.candidate_source in (
            "mutation",
            "mutation_tpe",
        ):
            n_tpe = min(
                n_rest, int(round(self.candidate_tpe_frac * self.n_candidates))
            )
            candidates.extend(
                self._tpe_pool(
                    n_tpe, state, rewarded_arms, nb_pending_total, nb_rewarded_total
                )
            )
            n_rest -= n_tpe

        if self.candidate_source == "mix":
            # Half local, half global, then topped up to n_candidates with uniform
            # draws AFTER deduplication rather than by a fixed share up front. The
            # two halves have very different collision rates -- the mutation half
            # lives in a neighbourhood of sum_i(|A_i| - 1) configs (25 on the RF
            # grid) and collides constantly, while joint TPE draws rarely repeat --
            # so a fixed uniform share would leave the pool short by an amount that
            # depends on how saturated the neighbourhood is. Topping up last keeps
            # the surrogate scoring a full n_candidates of DISTINCT configs however
            # much the two sources overlap.
            n_mut = int(round(self.candidate_mix_mutation_frac * n_rest))
            candidates.extend(
                self._tpe_pool(
                    n_rest - n_mut, state, rewarded_arms,
                    nb_pending_total, nb_rewarded_total,
                )
            )
            for parent in self._select_parents(
                rewarded_arms, n_mut, state, nb_pending_total, nb_rewarded_total
            ):
                candidates.append(self._mutate_config(parent, None))
            return self._top_up(candidates, rewarded_arms)

        if self.candidate_source == "tpe":
            candidates.extend(
                self._tpe_pool(
                    n_rest, state, rewarded_arms, nb_pending_total, nb_rewarded_total
                )
            )
            return candidates

        if self.candidate_source == "mutation_tpe":
            value_sampler = self._tpe_value_sampler(
                state, rewarded_arms, nb_pending_total, nb_rewarded_total
            )
        elif self.mutation_scale is not None:
            value_sampler = local_value_sampler(
                self.distributions, self.rng, self.mutation_scale
            )
        else:
            value_sampler = None
        for parent in self._select_parents(
            rewarded_arms, n_rest, state, nb_pending_total, nb_rewarded_total
        ):
            candidates.append(self._mutate_config(parent, value_sampler))
        return candidates

    # Uniform draws attempted per missing slot when topping the mix pool up. The
    # space can be smaller than n_candidates, so the loop must be bounded.
    _TOP_UP_ATTEMPTS = 20

    def _top_up(
        self,
        candidates: list[ArmConfig],
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
    ) -> list[ArmConfig]:
        """Dedup ``candidates`` and refill to ``n_candidates``, per ``candidate_topup``.

        ``"uniform"`` fills with uniform draws. Those sit far from the training
        table, so TabPFN's predictive variance on them is large and a
        variance-rewarding acquisition (the 0.99 quantile) scores them high for
        being UNFAMILIAR rather than good: measured on the mix pool at
        ``refit_every=1``, the top-up supplied 33% of the pool and won 13% of the
        argmaxes. That is unbudgeted random exploration.

        ``"runner_up"`` fills with mutants of the population's next-best arms
        instead -- cycling through the rewarded arms in descending mean reward,
        skipping the incumbent whose own neighbourhood the pool already holds. The
        pool still gets fresh configurations when the incumbent's neighbourhood is
        exhausted, but they are one coordinate from an arm known to be good rather
        than uniform strangers. Falls back to uniform draws if the runners-up
        cannot supply enough distinct configs (a tiny or saturated space).

        Returns distinct configs, at most ``n_candidates``, fewer only if the
        search space itself cannot supply more.
        """
        pool: dict[ArmKey, ArmConfig] = {}
        for candidate in candidates:
            pool.setdefault(config_to_key(candidate, self.param_names), candidate)

        budget = self._TOP_UP_ATTEMPTS * self.n_candidates
        if self.candidate_topup == "runner_up" and len(rewarded_arms) > 1:
            # Descending mean reward, rank 2 onwards: rank 1 is the mutation
            # pool's own parent under parent_rule="best".
            runners = [
                key_to_config(k, self.param_names)
                for k, _ in sorted(
                    rewarded_arms, key=lambda kv: kv[1].mean_reward, reverse=True
                )[1:]
            ]
            attempts = 0
            while len(pool) < self.n_candidates and attempts < budget:
                parent = runners[attempts % len(runners)]
                candidate = self._mutate_config(parent)
                pool.setdefault(config_to_key(candidate, self.param_names), candidate)
                attempts += 1

        attempts = 0
        while len(pool) < self.n_candidates and attempts < budget:
            candidate = self.generate_random_config()
            pool.setdefault(config_to_key(candidate, self.param_names), candidate)
            attempts += 1
        return list(pool.values())[: self.n_candidates]

    def _select_parents(
        self,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        k: int,
        state: CurrentState,
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> list[ArmConfig]:
        """The ``k`` configs the pool's mutants are built from, per ``parent_rule``.

        ``"softmax"`` draws each one independently, so a single pool can span
        several arms; every other rule identifies one config and returns it ``k``
        times, making the pool that config's neighbourhood (``k`` different
        single-coordinate perturbations of it). See
        :func:`imabo.mutation.parent_config`.
        """
        args = (rewarded_arms, state, nb_pending_total, nb_rewarded_total)
        if self.parent_rule == "softmax":
            return [
                parent_config(
                    self, "softmax", *args, temperature=self.candidate_temperature
                )
                for _ in range(k)
            ]
        return [
            parent_config(
                self, self.parent_rule, *args, last_proposal=self._last_proposal
            )
        ] * k

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

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new configuration using the TabPFN-3 oracle (explore).

        One TabPFN fit+predict ranks the ``n_candidates`` pool built by
        :meth:`_sample_candidates` (uniform, or mutations of the population when
        ``candidate_source="mutation"``) by the acquisition (``mean + kappa *
        std``, or the ``quantile`` level of the predictive distribution) and
        caches the top ``refit_every`` of them, so the expensive
        foundation-model call is amortized across several explore steps.
        """
        if len(rewarded_arms) < self.min_arms_for_fit:
            return self.generate_random_config()

        if self._pending_candidates:
            config, mean_pred, ucb_pred = self._pending_candidates.pop(0)
            if self.on_suggestion is not None:
                self.on_suggestion(config, mean_pred, ucb_pred)
            self._last_proposal = config
            return config

        candidates = self._sample_candidates(
            rewarded_arms, state, nb_pending_total, nb_rewarded_total
        )
        # Deduplicate before scoring. A pool can contain the same configuration
        # many times over -- a mutation pool draws each candidate independently, so
        # on a finite space collisions are common, and a rule whose value choice is
        # deterministic per coordinate (mutation_tpe with the EI-argmax pick) makes
        # every mutant on a coordinate identical. Scoring duplicates costs TabPFN
        # rows and, worse, fills the `refit_every` cache with repeats of one config
        # instead of the several distinct runners-up it is meant to hold.
        deduped: dict[ArmKey, ArmConfig] = {}
        for candidate in candidates:
            deduped.setdefault(config_to_key(candidate, self.param_names), candidate)

        # Then drop candidates that are already open arms. The pool exists so the
        # surrogate can pick the next arm to OPEN; a candidate already in memory
        # cannot do that, and preferring one stalls the search outright. With
        # parent_rule="best" the mutation neighbourhood holds only sum_i(|A_i| - 1)
        # configurations -- 25 on the RF grid -- so once they are all open the
        # acquisition keeps re-selecting them, |arms| stops growing, the
        # `|arms| < t**beta` switch never clears, and every remaining round is an
        # explore round paying for a TabPFN fit. Measured before this filter:
        # parent=best pools reached 43-71 of the 71 arms the schedule asks for,
        # while the softmax and uniform pools always reached exactly 71.
        # ``filter_open_candidates=False`` restores the behaviour the pre-filter
        # results were produced with: the surrogate ranks open and unopened
        # candidates alike and is free to prefer a known-good arm, which on this
        # grid it almost always does (the pool's 10 uniform draws from 2250 configs
        # are essentially always novel and essentially never win the acquisition).
        novel = (
            [c for k, c in deduped.items() if k not in state.arms]
            if self.filter_open_candidates
            else list(deduped.values())
        )
        if not novel:
            # Everything reachable is already open (a small finite space, or a
            # saturated neighbourhood): draw uniformly instead of stalling.
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
        self._last_proposal = chosen
        return chosen
