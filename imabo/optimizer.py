from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from imabo.memory import (
    ArmStats,
    CurrentState,
    InMemoryStorage,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.moss import kl_ucb, moss_anytime, ucb, ucb_siri
from imabo.tpe import (
    create_search_space,
    default_gamma,
    default_weights,
    tpe_suggest,
)
from imabo.types import ArmConfig, ArmKey


class IMABO:
    """Infinite Multi-Armed Bandits with Oracles (IMABO).

    IMABO combines a TPE (Bayesian) oracle for discovering new configurations
    with a MOSS (bandit) oracle for exploiting known ones. A switching rule
    controlled by beta determines when to explore vs. exploit:
        - If |M_t| < t^beta: invoke TPE to propose a new arm.
        - Otherwise: invoke MOSS to select the best existing arm.

    Example:
        >>> optimizer = IMABO(
        ...     search_space={"lr": {"lower": 1e-5, "upper": 1.0, "log": True}},
        ...     n_startup_trials=5,
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
        prior_weight: float = 1.0,
        n_ei_candidates: int = 24,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        switch_strategy: Literal["beta", "delayed"] = "beta",
        beta: float = 0.8,
        multivariate: bool = True,
        use_tpe: bool = True,
        memory: Memory | None = None,
        tpe_split_bound: Literal["moss", "lcb"] = "moss",
    ):
        """Initialize the IMABO optimizer.

        Args:
            search_space: Dictionary defining the search space. Each key maps to
                a dict with either {"lower", "upper"} (optionally "log", "int")
                or {"choices": [...]}.
            seed: Random seed for reproducibility.
            n_min_rewarded: Minimum observations before an arm is considered.
            max_nb_pending_per_unrewarded_arm: Max pending pulls for unrewarded arms.
            n_startup_trials: Number of random initial configurations.
            prior_weight: Prior weight for the Parzen estimator.
            n_ei_candidates: Number of EI candidates sampled from l(x).
            gamma_func: Function mapping n_rewarded_arms -> n_good (quantile split).
            weights_func: Weight function for the Parzen estimator.
            switch_strategy: "beta" (synchronous) or "delayed" (asynchronous).
            beta: Switching exponent controlling exploration rate.
            multivariate: Whether to use multivariate Parzen estimation.
            use_tpe: Whether to use TPE for exploration.
            memory: Custom memory backend (defaults to InMemoryStorage).
            tpe_split_bound: Index used to rank arms in :meth:`tpe_split`,
                "moss" (default, MOSS-anytime UCB) or "lcb" (classic UCB1-style
                lower confidence bound, an alternative pessimistic ranking).
        """
        self.search_space_specs = search_space
        self.param_names = list(sorted(search_space.keys()))
        self.rng = random.Random(seed)

        self.distributions, self.param_types = create_search_space(search_space)

        self.memory: Memory = (
            memory
            if memory is not None
            else InMemoryStorage(param_names=self.param_names)
        )

        self.switch_strategy = switch_strategy
        self.n_min_rewarded = n_min_rewarded
        self.max_nb_pending = max_nb_pending_per_unrewarded_arm
        self.n_startup_trials = n_startup_trials
        self.beta = beta

        self.prior_weight = prior_weight
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func or default_gamma
        self.weights_func = weights_func or default_weights
        self.multivariate = multivariate
        self.use_tpe = use_tpe
        self.tpe_split_bound = tpe_split_bound

        self.last_suggested: ArmConfig | None = None

        self.initialize_startup_points()

    def suggest(self) -> ArmConfig:
        """Suggest the next configuration to evaluate.

        Returns:
            A dictionary mapping parameter names to values.
        """
        state = self.memory.get_current_state()
        rewarded_arms = self.get_rewarded_arms(state)

        undersampled = self.get_undersampled_point(state)
        if undersampled is not None:
            self.last_suggested = undersampled
            return undersampled

        nb_pending_total = sum(s.nb_pending for s in state.arms.values())
        nb_rewarded_total = sum(s.nb_rewarded for s in state.arms.values())

        if self.switch_strategy == "delayed":
            effective_t = (
                nb_rewarded_total
                + self.memory.get_reward_frequency() * nb_pending_total
            )
            explore = len(state.arms) < effective_t**self.beta
        else:
            explore = len(state.arms) < state.nb_steps**self.beta

        if explore:
            if not self.use_tpe:
                x = self.generate_random_config()
            else:
                x = self.suggest_new(
                    state, rewarded_arms, nb_pending_total, nb_rewarded_total
                )
        else:
            x = self.suggest_existing(
                state, rewarded_arms, nb_pending_total, nb_rewarded_total
            )

        key = config_to_key(x, self.param_names)
        self.memory.pull_arm(key)
        self.last_suggested = x
        return x

    def observe(self, reward: float) -> None:
        """Observe the reward for the last suggested configuration.

        Args:
            reward: The observed reward value.
        """
        if self.last_suggested is None:
            raise RuntimeError("observe() called before suggest()")
        self.memory.observe(self.last_suggested, reward)
        self.last_suggested = None

    @property
    def best_config(self) -> ArmConfig | None:
        """Return the arm with the highest empirical mean reward found so far."""
        state = self.memory.get_current_state()
        fully_sampled = [
            (key, stats.mean_reward)
            for key, stats in state.arms.items()
            if stats.nb_rewarded >= self.n_min_rewarded
        ]

        if fully_sampled:
            best_key, _ = max(fully_sampled, key=lambda x: x[1])
            return key_to_config(best_key, self.param_names)

        all_evaluated = [
            (key, stats.mean_reward)
            for key, stats in state.arms.items()
            if stats.nb_rewarded > 0
        ]
        if not all_evaluated:
            return None

        best_key, _ = max(all_evaluated, key=lambda x: x[1])
        return key_to_config(best_key, self.param_names)

    @property
    def best_x(self) -> ArmConfig | None:
        """Alias for :attr:`best_config` (kept for backward compatibility)."""
        return self.best_config

    def initialize_startup_points(self) -> None:
        """Seed memory with ``n_startup_trials`` random (unrewarded) arms."""
        for _ in range(self.n_startup_trials):
            config = self.generate_random_config()
            key = config_to_key(config, self.param_names)
            self.memory.set(key, ArmStats())

    def generate_random_config(self) -> ArmConfig:
        """Sample a configuration uniformly at random from the search space."""
        config: ArmConfig = {}
        for name in self.param_names:
            dist = self.distributions[name]
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    log_low = math.log(dist.low)
                    log_high = math.log(dist.high)
                    config[name] = math.exp(self.rng.uniform(log_low, log_high))
                else:
                    config[name] = self.rng.uniform(dist.low, dist.high)
            elif isinstance(dist, IntDistribution):
                config[name] = self.rng.randint(dist.low, dist.high)
            elif isinstance(dist, CategoricalDistribution):
                config[name] = self.rng.choice(dist.choices)
        return config

    def get_rewarded_arms(self, state: CurrentState) -> list[tuple[ArmKey, ArmStats]]:
        """Return the (key, stats) pairs of arms with at least one reward."""
        return [
            (key, stats) for key, stats in state.arms.items() if stats.nb_rewarded > 0
        ]

    def get_undersampled_point(self, state: CurrentState) -> ArmConfig | None:
        """Return an unrewarded arm still below the pending budget, if any."""
        for key, stats in state.arms.items():
            if stats.nb_rewarded == 0 and stats.nb_pending < self.max_nb_pending:
                self.memory.pull_arm(key)
                return key_to_config(key, self.param_names)
        return None

    def compute_moss_score(
        self,
        state: CurrentState,
        stats: ArmStats,
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> float:
        """Compute the MOSS-anytime score of a single arm."""
        return moss_anytime(
            mean_reward=stats.mean_reward,
            n_arms=len(state.arms),
            step_counter=state.nb_steps,
            nb_rewarded_arm=stats.nb_rewarded,
            alpha=0.1,
            beta=self.beta,
            switch_strategy=self.switch_strategy,
            reward_frequency=self.memory.get_reward_frequency(),
            nb_rewarded_total=nb_rewarded_total,
            nb_pending_total=nb_pending_total,
            nb_pending_arm=stats.nb_pending,
        )

    def compute_ucb_lcb_score(
        self,
        state: CurrentState,
        stats: ArmStats,
    ) -> float:
        """Compute the classic UCB1-style lower confidence bound of an arm.

        Pessimistic alternative to :meth:`compute_moss_score`, used by
        :meth:`tpe_split` when ``tpe_split_bound == "lcb"``.
        """
        return ucb(
            mean=stats.mean_reward,
            nb_rewarded_arm=stats.nb_rewarded,
            total_pulls=state.nb_steps,
            bonus_type="lcb",
        )

    def suggest_existing(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> ArmConfig:
        """Select the existing arm with the highest MOSS score (exploit)."""
        if not rewarded_arms:
            return self.generate_random_config()

        best_key, _ = max(
            rewarded_arms,
            key=lambda t: self.compute_moss_score(
                state, t[1], nb_pending_total, nb_rewarded_total
            ),
        )
        return key_to_config(best_key, self.param_names)

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new configuration using the TPE oracle (explore)."""
        if not rewarded_arms:
            return self.generate_random_config()

        good, bad = self.tpe_split(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )

        good_configs = [key_to_config(k, self.param_names) for k, _ in good]
        bad_configs = [key_to_config(k, self.param_names) for k, _ in bad]

        rng_state = np.random.RandomState(self.rng.randint(0, 2**32 - 1))

        candidate = tpe_suggest(
            good_configs=good_configs,
            bad_configs=bad_configs,
            param_names=self.param_names,
            distributions=self.distributions,
            n_candidates=self.n_ei_candidates,
            rng=rng_state,
            prior_weight=self.prior_weight,
            multivariate=self.multivariate,
            weights_func=self.weights_func,
        )

        if candidate is None:
            return self.generate_random_config()
        return candidate

    def tpe_split(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> tuple[list[tuple[ArmKey, ArmStats]], list[tuple[ArmKey, ArmStats]]]:
        """Split rewarded arms into 'good' and 'bad' sets, ranked by score.

        Ranking index is controlled by ``tpe_split_bound``: the default
        MOSS-anytime UCB, or the classic UCB1-style LCB as an alternative.
        """

        def score(arm: tuple[ArmKey, ArmStats]) -> float:
            if self.tpe_split_bound == "lcb":
                return self.compute_ucb_lcb_score(state, arm[1])
            return self.compute_moss_score(
                state, arm[1], nb_pending_total, nb_rewarded_total
            )

        sorted_arms = sorted(rewarded_arms, key=score, reverse=True)
        n_good = self.gamma_func(len(sorted_arms))
        n_good = max(1, min(n_good, len(sorted_arms) - 1))
        return sorted_arms[:n_good], sorted_arms[n_good:]


class FiniteIMABO(IMABO):
    """Finite-budget variant of IMABO (alternative optimizer).

    This is an alternative to the default :class:`IMABO`. Instead of the
    MOSS-anytime exploitation oracle, it exploits known arms with a UCB / KL-UCB
    index tuned for a fixed evaluation budget, and uses a schedule-based
    switching rule:

        - When ``n_rewarded_arms < step ** ef``: explore via TPE (new config).
        - Otherwise: exploit by pulling an existing arm via :meth:`pull`.

    An optional ``max_pulls_per_config`` caps how often a single configuration
    can be pulled, forcing exploration of new arms once arms saturate.
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        total_budget: int = 200,
        seed: int | None = 42,
        nb_pulls_new_arm: int = 50,
        min_pulls_for_fit: int = 30,
        min_arms_for_fit: int = 10,
        max_pulls_per_config: int = 1000,
        n_startup_trials: int = 10,
        prior_weight: float = 1.0,
        n_ei_candidates: int = 24,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        ef: float = 0.5,
        schedule: Literal["power", "linear"] = "power",
        multivariate: bool = True,
        memory: Memory | None = None,
        pull_strategy: Literal["kl_ucb", "ucb"] = "ucb",
    ):
        """Initialize FiniteIMABO.

        Args:
            search_space: Dictionary defining the search space.
            total_budget: Total number of arm pulls (time horizon).
            seed: Random seed for reproducibility.
            nb_pulls_new_arm: Initial pulls for new arms.
            min_pulls_for_fit: Minimum pulls before an arm is considered for best.
            min_arms_for_fit: Minimum arms before fitting the population model.
            max_pulls_per_config: Maximum pulls per configuration.
            n_startup_trials: Number of random initial configurations.
            prior_weight: Prior weight for the Parzen estimator.
            n_ei_candidates: Number of EI candidates to sample.
            gamma_func: Function to determine the number of "good" trials.
            weights_func: Function to compute trial weights.
            ef: Schedule exponent (explore when n_rewarded_arms < step ** ef).
            schedule: Schedule type ("power" or "linear").
            multivariate: Whether to use multivariate TPE.
            memory: Optional pre-initialized storage.
            pull_strategy: Arm-selection index, "ucb" or "kl_ucb".
        """
        super().__init__(
            search_space=search_space,
            seed=seed,
            n_startup_trials=n_startup_trials,
            prior_weight=prior_weight,
            n_ei_candidates=n_ei_candidates,
            gamma_func=gamma_func,
            weights_func=weights_func,
            switch_strategy="beta",
            beta=ef,
            multivariate=multivariate,
            memory=memory,
        )
        self.total_budget = total_budget
        self.nb_pulls_new_arm = nb_pulls_new_arm
        self.min_pulls_for_fit = min_pulls_for_fit
        self.min_arms_for_fit = min_arms_for_fit
        self.max_pulls_per_config = max_pulls_per_config

        self.schedule = schedule
        self.pull_strategy = pull_strategy
        self.ef = ef

    @property
    def best_config(self) -> ArmConfig | None:
        """Return the most-pulled arm (ties broken by mean reward)."""
        state = self.memory.get_current_state()
        all_evaluated = [
            (key, stats.mean_reward, stats.nb_rewarded)
            for key, stats in state.arms.items()
            if stats.nb_rewarded > 0
        ]
        if not all_evaluated:
            return None

        best_key, _, _ = max(all_evaluated, key=lambda x: (x[2], x[1]))
        return key_to_config(best_key, self.param_names)

    def get_undersampled_point(self, state: CurrentState) -> ArmConfig | None:
        """Return any unrewarded arm (no pending budget cap, unlike IMABO)."""
        for key, stats in state.arms.items():
            if stats.nb_rewarded == 0:
                self.memory.pull_arm(key)
                return key_to_config(key, self.param_names)
        return None

    def pull(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
    ) -> ArmConfig:
        """Pull the best existing arm according to the UCB pull strategy."""
        assert len(rewarded_arms) > 0
        if self.pull_strategy == "kl_ucb":
            scores = [
                kl_ucb(stats.mean_reward, stats.nb_rewarded, state.nb_steps)
                for _, stats in rewarded_arms
            ]
        elif self.pull_strategy == "ucb":
            scores = [
                ucb_siri(stats.mean_reward, stats.nb_rewarded, self.total_budget)
                for _, stats in rewarded_arms
            ]
        else:
            raise ValueError(f"Invalid pull strategy: {self.pull_strategy}")

        # Penalize saturated arms so they are no longer selected.
        invalid_arms = np.array(
            [
                stats.nb_rewarded >= self.max_pulls_per_config
                for _, stats in rewarded_arms
            ],
        )
        scores = np.array(scores) - invalid_arms * 1e9

        best_idx = int(np.argmax(scores))
        best_key, _ = rewarded_arms[best_idx]
        return key_to_config(best_key, self.param_names)

    def suggest(self) -> ArmConfig:
        """Suggest the next configuration to evaluate."""
        state = self.memory.get_current_state()
        rewarded_arms = self.get_rewarded_arms(state)

        undersampled = self.get_undersampled_point(state)
        if undersampled is not None:
            self.last_suggested = undersampled
            return undersampled

        threshold = state.nb_steps**self.ef

        all_saturated = len(rewarded_arms) > 0 and all(
            stats.nb_rewarded >= self.max_pulls_per_config for _, stats in rewarded_arms
        )

        if len(rewarded_arms) < threshold or all_saturated:
            x = self.suggest_new(state, rewarded_arms)
        else:
            x = self.pull(state, rewarded_arms)

        key = config_to_key(x, self.param_names)
        self.memory.pull_arm(key)
        self.last_suggested = x
        return x


def load_tabfm(model_type: str = "regression") -> Any:
    """Load TabFM weights via safetensors.

    Works around a bug in tabfm==1.0.0's own ``load()``: it hardcodes
    looking for ``pytorch_model.bin``, but the HF repo
    ``google/tabfm-1.0.0-pytorch`` currently only ships ``model.safetensors``,
    so ``load()`` raises ``FileNotFoundError``.

    Imports tabfm/huggingface_hub/safetensors lazily (only when this function
    is called) so that importing this module never requires those optional
    dependencies -- only constructing or using :class:`IMABOTabFM` does.
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
    base_path = snapshot_download(repo_id=HF_REPO_ID)
    checkpoint_file = Path(base_path) / model_type / "model.safetensors"
    model.load_state_dict(load_file(checkpoint_file), strict=True)
    model.eval()
    return model


class IMABOTabFM(IMABO):
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
        >>> optimizer = IMABOTabFM(
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
        """Initialize IMABOTabFM.

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

    def _configs_to_frame(self, configs: list[ArmConfig]) -> Any:
        """Build a DataFrame from configs, tagging categorical columns."""
        import pandas as pd

        df = pd.DataFrame(configs, columns=self.param_names)
        for name in self.param_names:
            if self.param_types[name] == "categorical":
                df[name] = df[name].astype("category")
        return df

    def _fit_surrogate(self, rewarded_arms: list[tuple[ArmKey, ArmStats]]) -> Any:
        """Fit a fresh TabFM regressor on all observed (config, reward) pairs."""
        from tabfm import TabFMRegressor

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
