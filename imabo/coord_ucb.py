"""IMABO whose explore oracle is a coordinate bandit over population mutations.

:class:`IMABOCoordUCB` keeps IMABO's MOSS exploit phase unchanged and proposes
new arms with no surrogate model at all -- three cheap decisions per explore
step:

1. **Which arm to improve** (``parent_rule``, see
   :func:`imabo.mutation.parent_config` for all four): ``"best"`` (the incumbent
   hill-climb -- the highest empirical mean, which is NOT the arm MOSS pulls),
   ``"softmax"`` (the whole population, weighted), ``"last_proposal"`` (whatever
   this oracle proposed last -- a walk, not a climb), or ``"moss"`` (the arm the
   exploit phase would pull right now).
2. **Which coordinate to improve** (``coord_rule``). ``"ucb"``: a UCB1 bandit over
   the ``d`` parameters -- literally Hier-MAB / AutoRAG-HP's high-level bandit
   (:class:`imabo.moss.UCB1`), credited with the reward the resulting arm earns.
   ``"random"``: uniform over the coordinates, the ablation of that bandit.

   With ``global_tpe_arm=True`` that bandit gets one EXTRA arm, index ``d``, which
   proposes a whole configuration from the multivariate TPE
   (:func:`imabo.tpe.tpe_suggest`, IMABO's own oracle) instead of perturbing the
   incumbent. The bandit then arbitrates between local search and a global jump on
   the same footing, learning how much of each this landscape rewards -- rather
   than the mixing rate being a hyperparameter. A local mutation can only reach
   configurations one coordinate from the best arm, which is exactly what fails on
   a landscape rewarding coordinated moves.
3. **Which value to give it** (``value_rule``). ``"tpe"``: a univariate TPE over
   that one coordinate -- the EI-argmax of ``n_ei_candidates`` draws from the 1-D
   ``l`` density fit on the good arms of IMABO's own :meth:`IMABO.tpe_split`
   (:func:`imabo.tpe.univariate_tpe_values`). ``"ucb"``: one UCB1 bandit per
   axis over that axis' finite value set (:func:`imabo.mutation.axis_values`) --
   Hier-MAB's low level, same bandit and same arm sets. ``"random"``: a uniform
   draw over the parameter's own domain, excluding the parent's current value
   (:func:`imabo.mutation.mutate_value`) -- no model and no bandit at the low
   level, which makes it the ablation that asks whether choosing the value
   deliberately buys anything over choosing the coordinate deliberately.

So ``parent_rule="best", value_rule="ucb"`` is Hier-MAB's entire proposal rule
running as an IMOSS explore oracle, and the other combinations vary one decision
at a time from there.

Relation to Hier-MAB. Hier-MAB *is* this rule with no MOSS exploit phase, no
switching rule, and an incumbent that it updates greedily per axis. Two
differences matter beyond that: with ``parent_rule="softmax"`` the whole rewarded
population stays alive (several basins at once, rather than one incumbent), and
with ``value_rule="tpe"`` the value comes from a density over good arms instead
of a grid, so continuous parameters are proposed at arbitrary precision.

Credit assignment (``credit_rule``). Each proposed arm credits its decision
exactly once -- later exploit pulls of the same arm are not new decisions -- but
*what* value that one vote carries has three settings:

* ``"first_pull"`` (default): the arm's first observed reward, which is what
  Hier-MAB itself credits. On a Bernoulli benchmark that is a single 0/1 draw.
* ``"arm_mean"``: that arm's empirical mean over every reward observed for it
  (:meth:`imabo.moss.UCB1.revise` keeps the vote count at one while its value is
  corrected). Between two oracle calls the exploit phase pulls the proposed arm
  many more times -- measured on the RF grid at beta=0.5: median 50 rewards per
  proposed arm, 96% of them above 10 -- so this is the same target measured with
  roughly 7x less standard error.

* ``"improvement"``: the coordinate bandit is credited ``mean(child) -
  mean(parent)``, i.e. *did perturbing this coordinate pay off*, which is the
  question the high level is meant to answer; the value bandit still gets
  ``mean(child)``, since "which value is good" is an absolute question. Both means
  are read at credit time and the vote is re-cast from whatever both arms' current
  estimates are -- which is why a decision stores its parent's key rather than a
  snapshot of the parent's mean. Freezing the parent would sharpen one side of the
  comparison over ~50 pulls while leaving the other at its initial estimate.
  Note the credit is centred near zero and can be negative, which shrinks the
  signal against an unchanged UCB1 bonus -- lower ``coord_alpha`` to compensate.

The two revising rules re-score their votes lazily, in
:meth:`IMABOCoordUCB._refresh_credits` at the top of an oracle call. Nothing reads
a bandit index between two oracle calls, so revising on every reward would be work
no one observes; the selections come out identical either way (verified against a
per-reward implementation on a stored run).

Novelty (``require_new_arm``). The proposal is normally served as-is, even when it
reproduces a configuration already opened -- Hier-MAB re-serves its incumbent all
the time, and IMOSS simply spends the round re-pulling that arm. Measured on the
RF grid that is most of the explore budget: ``value_rule="tpe"`` re-proposed a
known arm in 98% of its explore rounds. With ``require_new_arm=True`` the oracle
instead walks its own preference order -- coordinates by UCB1 index, and within a
coordinate the values by their own index (``"ucb"``) or EI rank (``"tpe"``) -- and
returns the first combination that opens a NEW arm, falling back to a uniform draw
if none does. The first choice is unchanged whenever it is already new, so this
only ever replaces a wasted repeat.

Non-stationarity (``bandit_rule``, ``bandit_discount``). Both bandits are asked a
question whose answer changes during the run. The credit of mutating coordinate
``i`` is what that mutation is worth *against the current incumbent*, and once the
incumbent already holds a good value on axis ``i`` there is nothing left to win
there -- the arm rots as it is pulled. UCB1 and KL-UCB both assume the opposite
(a fixed distribution, estimated ever better), so they keep re-picking an axis
that paid off early. Four answers, none of which change anything else:

* ``bandit_discount`` < 1 keeps the UCB index but decays old votes, so the
  estimate follows the recent past and the effective sample size saturates at
  ``1 / (1 - bandit_discount)`` -- discounted-UCB (Garivier & Moulines 2008).
* ``bandit_rule="exp3"`` drops the stationarity assumption entirely
  (:class:`imabo.moss.EXP3`).
* ``exp3_mixing`` > 0 makes that EXP3.S, which competes with the best *sequence*
  of coordinates rather than the best fixed one -- the variant actually built for
  a moving optimum.
* ``coord_bandit_scope="parent"`` denies the premise: the credit is not drifting,
  it is *conditional* on the arm being mutated, so give each parent its own
  bandit rather than forgetting a shared one. That makes the reward stationary
  within a bandit by construction, and an incumbent regained after being displaced
  resumes with its history rather than from scratch (which is what a reset-on-change
  rule would throw away). The cost is that the votes are split across parents and
  each new parent starts with a round of round-robin.

The UCB1 bonus is not variance-adapted (see :class:`imabo.moss.UCB1`): on rewards
whose spread is far below 1 (validation accuracies, typically), the bonus dominates
the mean gaps and the choice stays close to round-robin, exactly as in Hier-MAB.
``coord_alpha`` / ``value_alpha`` scale those bonuses, so ``-> 0`` recovers a
greedy "always pick what has paid best so far". This bites hardest inside IMOSS,
where the ``|arms| < t**beta`` switching rule only calls the oracle O(t**beta)
times: at beta=0.5 on a 5000-round run that is ~112 decisions spread over the
coordinates, so the bandits see far less data than Hier-MAB's own levels do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from imabo.memory import (
    ArmStats,
    CurrentState,
    Memory,
    config_to_key,
    key_to_config,
)
from imabo.moss import EXP3, UCB1
from imabo.mutation import (
    ParentRule,
    axis_values,
    coordinate_importance,
    mutate_value,
    parent_config,
)
from imabo.optimizer import IMABO
from imabo.tpe import tpe_suggest, univariate_tpe_values
from imabo.types import ArmConfig, ArmKey


@dataclass
class _Decision:
    """One explore-step proposal: which coordinate/value produced which arm, from
    which parent.

    Keys, not means: under ``credit_rule="improvement"`` the vote is
    ``mean(child) - mean(parent)`` evaluated at credit time, so both arms' current
    estimates are looked up on every revision.

    ``credited_*`` remember what this decision has already contributed to each
    bandit, so a revision adjusts by the difference instead of casting a second
    vote (see :meth:`IMABOCoordUCB._cast`). ``token_*`` are what the bandit handed
    back when the vote was cast -- a discount epoch for :class:`imabo.moss.UCB1`,
    an importance weight for :class:`imabo.moss.EXP3` -- so the revision lands at
    the same weight as the vote it corrects.
    """

    coord: int
    value_idx: int | None
    arm_key: ArmKey
    parent_key: ArmKey
    n_mutated: int = 1
    credited_high: float | None = None
    credited_low: float | None = None
    credited_effect: float | None = None
    token_high: float | None = None
    token_low: float | None = None


class IMABOCoordUCB(IMABO):
    """
    Example:
        >>> optimizer = IMABOCoordUCB(
        ...     search_space={"x0": {"choices": [0, 1, 2, 3, 4, 5]}},
        ... )
        >>> for _ in range(100):
        ...     config = optimizer.suggest()
        ...     reward = evaluate(config)
        ...     optimizer.observe(reward)
        >>> print(optimizer.best_config)
    """

    # Value draws attempted per coordinate under value_rule="random" before the
    # walk moves on: a continuous axis cannot be enumerated.
    _RANDOM_VALUE_ATTEMPTS = 20
    # Continuation probability of mutation_size="geometric": each extra coordinate
    # is added with this probability, so E[k] = 1.9 on a 4-dimensional space.
    _GEOMETRIC_P = 0.5

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
        n_ei_candidates: int = 24,
        prior_weight: float = 1.0,
        multivariate: bool = True,
        gamma_func: Callable[[int], int] | None = None,
        weights_func: Callable[[int], np.ndarray] | None = None,
        min_arms_for_mutation: int = 10,
        temperature: float = 1.0,
        coord_alpha: float = 1.0,
        coord_rule: Literal["ucb", "random"] = "ucb",
        parent_rule: ParentRule = "softmax",
        credit_rule: Literal["first_pull", "arm_mean", "improvement"] = "first_pull",
        global_tpe_arm: bool = False,
        value_rule: Literal["tpe", "ucb", "random"] = "tpe",
        tpe_value_pick: Literal["ei_argmax", "sample"] = "ei_argmax",
        value_alpha: float = 1.0,
        bandit_bonus: Literal["hoeffding", "kl", "kl_scaled"] = "hoeffding",
        bandit_rule: Literal["ucb", "exp3"] = "ucb",
        bandit_discount: float = 1.0,
        exp3_mixing: float = 0.0,
        exp3_eta_scale: float = 1.0,
        exp3_feedback: Literal["reward", "loss"] = "loss",
        coord_bandit_scope: Literal["global", "parent"] = "global",
        n_points: int = 10,
        require_new_arm: bool = False,
        mutation_size: int | Literal["geometric"] = 1,
    ):
        """Initialize IMABOCoordUCB.

        Args:
            parent_rule: Which configuration gets mutated: ``"best"``,
                ``"softmax"``, ``"last_proposal"`` or ``"moss"`` -- see
                :func:`imabo.mutation.parent_config`.
            global_tpe_arm: Give the coordinate bandit an extra arm that proposes
                a full multivariate-TPE configuration rather than a mutation (see
                the module docstring). Credited and ranked exactly like the
                coordinate arms, so its usage rate is learned.
            credit_rule: What value each decision's single vote carries (see the
                module docstring): the proposed arm's first reward
                (``"first_pull"``, Hier-MAB's own rule), its running mean over
                every reward since the decision (``"arm_mean"``), or that mean
                minus the parent's (``"improvement"``, for the coordinate bandit
                only).
            value_rule: How the mutated coordinate's value is chosen.
                ``"tpe"`` = univariate TPE over that parameter; ``"ucb"`` = a
                UCB1 bandit per axis over that axis' finite value set
                (Hier-MAB's low level, same bandit and same value sets);
                ``"random"`` = uniform over the parameter's domain, excluding the
                parent's current value.
            mutation_size: How many coordinates one mutation changes. ``1``
                (default) is the single-coordinate perturbation every other rule
                here uses, and the assumption Hier-MAB's credit assignment rests
                on. An int ``k`` changes exactly ``k`` (clamped to the
                dimension); ``"geometric"`` draws ``k`` as 1 plus a geometric
                tail, so most mutations stay local while some jump further. The
                extra coordinates beyond the first are chosen uniformly and take
                the first value their ``value_rule`` offers.
            require_new_arm: Keep looking, down the oracle's own preference order,
                until the proposal is a configuration not yet opened (see the
                module docstring). ``False`` (default) serves the first choice
                even when it repeats a known arm, as Hier-MAB does.
            bandit_bonus: Confidence width for BOTH bandit levels --
                ``"hoeffding"`` (UCB1, scaled by the alphas), ``"kl"`` (KL-UCB,
                variance-adapted, ignoring the alphas, and requiring credits in
                [0, 1] -- so pair it with ``credit_rule`` ``"first_pull"`` or
                ``"arm_mean"``), or ``"kl_scaled"`` (KL-UCB on arm means rescaled
                by their own spread, which is what makes it usable with the
                ``"improvement"`` credit). See :class:`imabo.moss.UCB1`.
                Ignored under ``bandit_rule="exp3"``, which has no bonus term.
            bandit_rule: Which bandit runs BOTH levels. ``"ucb"`` (default) is
                :class:`imabo.moss.UCB1` with the width ``bandit_bonus`` names --
                it assumes each choice has a fixed reward distribution. ``"exp3"``
                is :class:`imabo.moss.EXP3`, which assumes nothing: worth trying
                because the credit of mutating a coordinate is measured against
                the current incumbent, so it decays once the incumbent already
                holds a good value on that axis, and a stationary estimator keeps
                re-picking an axis that paid off long ago.
            bandit_discount: With ``bandit_rule="ucb"``, the per-vote decay of
                discounted-UCB (1.0 = no discounting, the stationary estimator).
                The other, cheaper answer to the same non-stationarity: keep the
                UCB index but let the effective sample size saturate at
                ``1 / (1 - bandit_discount)`` so old credit fades. Given ~100
                votes over 5 coordinates per run, 0.95-0.99 is the useful range.
            exp3_mixing: With ``bandit_rule="exp3"``, the EXP3.S mixing rate
                (0 = plain EXP3, which tracks a fixed best choice only).
            exp3_eta_scale: Multiplies EXP3's anytime learning rate. The worst-case
                schedule is far too conservative at this horizon -- see
                :class:`imabo.moss.EXP3`.
            exp3_feedback: Whether EXP3 importance-weights the reward or the loss
                ``1 - r``. Defaults to the loss, which is much the quieter
                estimator on rewards that sit near 1 (see :class:`imabo.moss.EXP3`).
            coord_bandit_scope: ``"global"`` (default) runs one coordinate bandit
                for the whole run. ``"parent"`` runs one PER PARENT
                CONFIGURATION -- the contextual answer to the same
                non-stationarity ``bandit_discount`` attacks by forgetting: the
                credit does not drift, it is conditional on which arm is being
                mutated, so condition on it instead. Costs statistical power (the
                votes are split across parents) and forces a round of round-robin
                each time a new parent appears; buys an estimate that is never
                stale and, unlike resetting on an incumbent change, survives the
                incumbent oscillating back. Most meaningful with
                ``parent_rule="best"``, where the parents are few; under
                ``"softmax"`` every arm in the population gets its own bandit.
            tpe_value_pick: With ``value_rule="tpe"``, whether the coordinate's
                value is the EI-argmax of the draws (default) or one of them
                sampled uniformly, i.e. a draw from ``l`` itself. The argmax is
                deterministic given the fitted densities, so while the good/bad
                split moves slowly it keeps proposing the same value -- measured on
                the RF grid: 302 oracle calls yielding 61 new arms, one config
                proposed 67 times. Sampling trades per-proposal quality for that
                variety.
            value_alpha: Exploration multiplier of the per-axis value bandits
                under ``value_rule="ucb"`` (Hier-MAB's ``alpha_l``, default 1.0).
            n_points: Values per continuous/integer axis when discretising for
                ``value_rule="ucb"`` (:func:`imabo.mutation.axis_values`);
                categorical axes are used as given, so this is a no-op on an
                all-categorical space. Unused by ``value_rule="tpe"``, which
                proposes on the parameter's own domain.
            min_arms_for_mutation: Rewarded arms required before the oracle
                mutates at all; below it, new configs are drawn uniformly at
                random (the same warm-up rule as IMABOTabPFN's
                ``min_arms_for_fit``, so the variants are comparable at equal
                budget).
            temperature: Parent-selection temperature *in units of the
                population's own reward dispersion*: ``T = temperature *
                std(mean_rewards)`` (see
                :func:`imabo.mutation.parent_probabilities`). Must be > 0;
                ``-> 0`` always mutates the best arm, large values pick a parent
                uniformly. Only used by ``parent_rule="softmax"``.
            coord_rule: How the mutated coordinate is chosen: ``"ucb"`` (the
                UCB1 bandit) or ``"random"`` (uniform). Under ``"random"`` the
                bandit is still credited, so its statistics stay readable as a
                diagnostic, but nothing consults it.
            coord_alpha: Exploration multiplier of the coordinate bandit
                (``alpha`` of :class:`imabo.moss.UCB1`, 1.0 = Hier-MAB's own
                ``alpha_h``). Scales the ``sqrt(2 log t / n)`` bonus, so smaller
                values make the coordinate choice greedier on measured payoff.
            beta, switch_strategy, memory, n_min_rewarded,
            max_nb_pending_per_unrewarded_arm, n_startup_trials: As in
                :class:`IMABO` -- the exploit phase and switching rule are
                untouched.
            n_ei_candidates, prior_weight, multivariate, gamma_func,
            weights_func: TPE settings, used here for the good/bad split
                (``gamma_func``) and the univariate value density
                (``n_ei_candidates``, ``prior_weight``, ``weights_func``).
                ``multivariate`` is inherited from :class:`IMABO` but unused:
                this oracle only ever fits 1-D densities.
        """
        super().__init__(
            search_space=search_space,
            seed=seed,
            n_min_rewarded=n_min_rewarded,
            max_nb_pending_per_unrewarded_arm=max_nb_pending_per_unrewarded_arm,
            n_startup_trials=n_startup_trials,
            prior_weight=prior_weight,
            n_ei_candidates=n_ei_candidates,
            gamma_func=gamma_func,
            weights_func=weights_func,
            switch_strategy=switch_strategy,
            beta=beta,
            multivariate=multivariate,
            # Must stay True so IMABO.suggest()'s explore branch calls our
            # suggest_new below instead of falling back to a uniform draw.
            use_tpe=True,
            memory=memory,
        )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if parent_rule not in ("best", "softmax", "last_proposal", "moss"):
            raise ValueError(f"Invalid parent_rule: {parent_rule!r}")
        if credit_rule not in ("first_pull", "arm_mean", "improvement"):
            raise ValueError(f"Invalid credit_rule: {credit_rule!r}")
        if coord_rule not in ("ucb", "random"):
            raise ValueError(f"Invalid coord_rule: {coord_rule!r}")
        if value_rule not in ("tpe", "ucb", "random"):
            raise ValueError(f"Invalid value_rule: {value_rule!r}")
        self.min_arms_for_mutation = min_arms_for_mutation
        self.temperature = temperature
        self.coord_alpha = coord_alpha
        self.coord_rule = coord_rule
        self.parent_rule = parent_rule
        self.value_rule = value_rule
        self.credit_rule = credit_rule
        self.tpe_value_pick = tpe_value_pick
        # Kept for the require_new_arm fallback in _value_candidates, which
        # enumerates an axis when the TPE draws are all already-known arms.
        self.n_points = n_points
        self.require_new_arm = require_new_arm
        if mutation_size != "geometric" and int(mutation_size) < 1:
            raise ValueError(f"mutation_size must be >= 1 or 'geometric', got {mutation_size}")
        self.mutation_size = mutation_size
        # This oracle's own previous proposal, for parent_rule="last_proposal".
        # IMABO.last_suggested cannot serve: observe() clears it, so in a
        # synchronous suggest/observe loop it is always None by the next call.
        self._last_proposal: ArmConfig | None = None
        # Per-coordinate [decisions, total |child mean - parent mean|]: the
        # magnitude of the effect mutating that coordinate had, accumulated over
        # this oracle's own parent/child pairs. Read by
        # :meth:`coordinate_importance`; nothing in the search consults it, so it
        # is free to measure a different quantity than the bandit's credit.
        self._coord_effect: list[list[float]] = [
            [0.0, 0.0] for _ in self.param_names
        ]
        # The oracle's own previous proposal, for parent_rule="last_proposal"
        # (IMABO.last_suggested cannot serve: observe() clears it, so in a
        # synchronous suggest/observe loop it is always None by the next call).
        self._last_proposal: ArmConfig | None = None

        if bandit_rule not in ("ucb", "exp3"):
            raise ValueError(f"Invalid bandit_rule: {bandit_rule!r}")
        if bandit_rule == "ucb" and bandit_bonus == "kl" and credit_rule == "improvement":
            raise ValueError(
                "bandit_bonus='kl' needs credits in [0, 1]; credit_rule="
                "'improvement' can be negative -- use 'kl_scaled', which rescales "
                "the arm means onto [0, 1] by their own spread"
            )
        if bandit_rule == "exp3" and credit_rule == "improvement":
            raise ValueError(
                "bandit_rule='exp3' needs credits in [0, 1]: its importance-weighted "
                "estimate r / p is unbounded below on a negative reward, and "
                "credit_rule='improvement' is centred near zero"
            )
        self.bandit_bonus = bandit_bonus
        self.bandit_rule = bandit_rule
        self.bandit_discount = bandit_discount
        self.global_tpe_arm = global_tpe_arm
        # Arm d (past the last coordinate) is the global-TPE proposal, when enabled.
        self._global_arm = len(self.param_names) if global_tpe_arm else None

        # EXP3 selects stochastically, so each bandit needs a stream. They are
        # drawn from a dedicated generator seeded ONCE from this optimizer's rng
        # rather than from the rng itself: under coord_bandit_scope="parent" a
        # bandit is created at an unpredictable point mid-run, and drawing from
        # the shared rng there would make every later proposal depend on how many
        # distinct parents had appeared. Only spun up for EXP3, so the UCB1 path
        # consumes exactly the draws it always did.
        bandit_seeds = (
            np.random.default_rng(self.rng.randint(0, 2**32 - 1))
            if bandit_rule == "exp3"
            else None
        )

        def make_bandit(n_choices: int, alpha: float) -> UCB1 | EXP3:
            if bandit_rule == "exp3":
                return EXP3(
                    n_choices,
                    mixing=exp3_mixing,
                    eta_scale=exp3_eta_scale,
                    feedback=exp3_feedback,
                    rng=np.random.default_rng(bandit_seeds.integers(2**32 - 1)),
                )
            return UCB1(
                n_choices, alpha=alpha, bonus=bandit_bonus, discount=bandit_discount
            )

        if coord_bandit_scope not in ("global", "parent"):
            raise ValueError(f"Invalid coord_bandit_scope: {coord_bandit_scope!r}")
        self.coord_bandit_scope = coord_bandit_scope
        self._n_coord_choices = len(self.param_names) + (1 if global_tpe_arm else 0)
        self._make_coord_bandit = lambda: make_bandit(
            self._n_coord_choices, coord_alpha
        )
        # Parent key -> that parent's own coordinate bandit, under
        # coord_bandit_scope="parent", created on first use. Under "global" the
        # dict holds one bandit at key None, built here so that the construction
        # order (and hence the seed order above) does not depend on the run.
        self.coord_bandits: dict[ArmKey | None, UCB1 | EXP3] = {}
        if coord_bandit_scope == "global":
            self.coord_bandits[None] = self._make_coord_bandit()
        # Which bandit the next proposal will consult, set from the parent in
        # suggest_new. Credit does not read it -- a decision routes by its own
        # stored parent_key, so a vote revised much later still lands on the
        # bandit that cast it.
        self._coord_key: ArmKey | None = None
        # Hier-MAB's low level, when asked for: one value set + one bandit per
        # axis, from the shared axis_values() the baseline itself uses.
        self.values: dict[str, list[Any]] = {}
        self.value_bandits: dict[str, UCB1 | EXP3] = {}
        if value_rule == "ucb":
            self.values = {
                name: axis_values(self.distributions[name], n_points)
                for name in self.param_names
            }
            self.value_bandits = {
                name: make_bandit(len(self.values[name]), value_alpha)
                for name in self.param_names
            }

        # Arm key -> the decisions that proposed it, one per explore step that
        # landed on this config (the oracle does re-propose known arms). Each is
        # one vote. Under "first_pull" the list is dropped once that arm's first
        # reward has been credited, so later exploit pulls cannot re-credit; the
        # other rules keep it and revise instead, alongside the arm's running
        # score below.
        self._pending_choice: dict[ArmKey, list[_Decision]] = {}
        # Arm key -> [reward count, reward total] over every reward this optimizer
        # has observed for that arm, i.e. its empirical mean. Maintained for all
        # arms, not just proposed ones, because a decision's parent needs a
        # current mean too (see `credit_rule="improvement"`).
        self._arm_scores: dict[ArmKey, list[float]] = {}


    def observe(self, reward: float) -> None:
        """Record the reward, crediting the choices that proposed this arm.

        Delegates to :meth:`IMABO.observe` for the arm statistics; additionally,
        if the observed config is one this oracle created by mutating coordinate
        ``i`` (to value index ``j`` under ``value_rule="ucb"``), the decision's
        vote is cast on the coordinate bandit's arm ``i`` and that axis' value
        bandit's arm ``j``, carrying whatever ``credit_rule`` says (see the module
        docstring). One vote per proposed arm either way.
        """
        key = (
            config_to_key(self.last_suggested, self.param_names)
            if self.last_suggested is not None
            else None
        )
        super().observe(reward)
        if key is None:
            return

        if self.credit_rule == "first_pull":
            decisions = self._pending_choice.get(key)
            if decisions:
                for decision in decisions:
                    self._cast(decision, float(reward), float(reward))
                del self._pending_choice[key]
            return

        # Track this arm's empirical mean (kept here rather than read back from
        # memory: O(1) per observation instead of an O(K) state snapshot). The
        # votes that depend on it are re-scored at the next oracle call, which is
        # the only place a bandit index is read -- see _refresh_credits.
        score = self._arm_scores.setdefault(key, [0.0, 0.0])
        score[0] += 1
        score[1] += float(reward)

    def _refresh_credits(self) -> None:
        """Re-score every registered decision from the current estimates.

        Called once per oracle call, immediately before the bandits are consulted.
        Cost is O(#decisions) = O(t**beta), against O(1) per reward for the
        bookkeeping in :meth:`observe`.
        """
        for decisions in self._pending_choice.values():
            for decision in decisions:
                self._recast(decision)

    def _recast(self, decision: _Decision) -> None:
        """Re-evaluate a decision's vote from the current estimates."""
        arm_mean = self._arm_mean(decision.arm_key)
        if arm_mean is None:
            return
        parent_mean = self._arm_mean(decision.parent_key)
        # Effect MAGNITUDE, tracked alongside the (signed) credit: one entry per
        # decision, replaced as the estimates sharpen. Decisions whose "mutation"
        # reproduced the parent (the value rule re-picked the value it already had
        # -- 15-18% of them here) are skipped: comparing an arm with itself
        # contributes a spurious zero, and how often it happens differs per axis,
        # so including them biases the ranking towards whichever axis repeats most.
        if (
            parent_mean is not None
            and decision.arm_key != decision.parent_key
            and decision.n_mutated == 1  # multi-coordinate: not attributable
        ):
            effect = self._coord_effect[decision.coord]
            if decision.credited_effect is not None:
                effect[1] -= decision.credited_effect
            else:
                effect[0] += 1
            decision.credited_effect = abs(arm_mean - parent_mean)
            effect[1] += decision.credited_effect
        if self.credit_rule == "improvement":
            if parent_mean is None:
                return
            self._cast(decision, arm_mean - parent_mean, arm_mean)
        else:
            self._cast(decision, arm_mean, arm_mean)

    def _arm_mean(self, key: ArmKey) -> float | None:
        """This arm's empirical mean, or None if it has no reward yet."""
        score = self._arm_scores.get(key)
        return score[1] / score[0] if score and score[0] else None

    def _bandit_for(self, parent_key: ArmKey | None) -> UCB1 | EXP3:
        """The coordinate bandit that speaks for this parent.

        Under ``coord_bandit_scope="global"`` there is one, at key ``None``. Under
        ``"parent"`` there is one per parent configuration, created on first use:
        "which coordinate is worth changing" is a question ABOUT a configuration,
        and its answer expires when the configuration does. Keeping them separate
        rather than resetting a single bandit means an incumbent that is displaced
        and later regained comes back with its own history intact -- and the best
        arm does oscillate while the estimates sharpen.
        """
        key = parent_key if self.coord_bandit_scope == "parent" else None
        bandit = self.coord_bandits.get(key)
        if bandit is None:
            bandit = self.coord_bandits[key] = self._make_coord_bandit()
        return bandit

    @property
    def coord_bandit(self) -> UCB1 | EXP3:
        """The bandit for the parent currently being mutated."""
        return self._bandit_for(self._coord_key)

    def _cast(self, decision: _Decision, high: float, low: float) -> None:
        """Cast (or re-cast) a decision's single vote on both bandits."""
        coord_bandit = self._bandit_for(decision.parent_key)
        if decision.credited_high is None:
            decision.token_high = coord_bandit.update(decision.coord, high)
        else:
            coord_bandit.revise(
                decision.coord, high - decision.credited_high, decision.token_high
            )
        decision.credited_high = high

        if decision.value_idx is None:
            return
        bandit = self.value_bandits[self.param_names[decision.coord]]
        if decision.credited_low is None:
            decision.token_low = bandit.update(decision.value_idx, low)
        else:
            bandit.revise(
                decision.value_idx, low - decision.credited_low, decision.token_low
            )
        decision.credited_low = low

    def suggest_new(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """Propose a new arm: parent (``parent_rule``), coordinate by UCB1, value
        (``value_rule``) -- see the module docstring.

        Falls back to a uniform random config until ``min_arms_for_mutation``
        arms are rewarded (a good/bad split needs at least two, and a
        1-D density fit on one arm is meaningless).
        """
        if len(rewarded_arms) < max(2, self.min_arms_for_mutation):
            return self.generate_random_config()

        if self.credit_rule != "first_pull":
            self._refresh_credits()

        parent = self._select_parent(
            rewarded_arms, state, nb_pending_total, nb_rewarded_total
        )
        # Point the coordinate bandit at this parent before anything consults it
        # (a no-op under coord_bandit_scope="global").
        self._coord_key = config_to_key(parent, self.param_names)

        order = self._coordinate_order()

        for coord in order:
            if coord == self._global_arm:
                mutant = self._global_tpe_config(
                    state, rewarded_arms, nb_pending_total, nb_rewarded_total
                )
                if mutant is None:
                    continue
                key = config_to_key(mutant, self.param_names)
                if self.require_new_arm and key in state.arms:
                    continue
                # Not a single-coordinate pair: n_mutated marks it so the
                # importance accumulator skips it (see _recast).
                return self._register(
                    mutant, key, parent, coord, None, len(self.param_names)
                )
            name = self.param_names[coord]
            for value, value_idx in self._value_candidates(
                name, parent, state, rewarded_arms, nb_pending_total, nb_rewarded_total
            ):
                mutant = {**parent, name: value}
                # Extra coordinates are re-drawn on every step of the walk, so a
                # collision retries a different combination rather than the same
                # one.
                mutant, n_mutated = self._add_extra_mutations(
                    mutant, name, state, rewarded_arms,
                    nb_pending_total, nb_rewarded_total,
                )
                key = config_to_key(mutant, self.param_names)
                if self.require_new_arm and key in state.arms:
                    continue
                return self._register(
                    mutant, key, parent, coord, value_idx, n_mutated
                )
            if not self.require_new_arm:
                break

        # Every preferred combination reproduces a known arm: take a uniform draw
        # rather than spend the explore round on a repeat.
        fallback = self.generate_random_config()
        self._last_proposal = fallback
        return fallback

    def coordinate_importance(
        self,
        source: Literal["decisions", "population"] = "decisions",
        min_rewards: int = 2,
    ) -> dict[str, float]:
        """Which parameters matter, estimated from this run's own data.

        Both sources estimate the same quantity -- the mean ``|change in reward|``
        from moving one coordinate and nothing else -- and both differ from
        anything the coordinate bandit holds. The bandit's credit is *signed*, so
        it ranks the axis that matters most LAST (mutating an already-good arm on
        its most important axis is what hurts most); and ``|mean(signed)|`` is not
        a fix, since with all-negative credits it is just that ranking reversed.
        ``mean(|delta|)`` is the statistic that actually measures effect size.

        * ``"decisions"`` (default): over this oracle's own parent/child pairs,
          which differ in exactly one coordinate BY CONSTRUCTION and are both
          pulled repeatedly by the exploit phase -- the cheapest well-measured
          pairs available. Requires ``credit_rule`` to be revising
          (``"arm_mean"``/``"improvement"``), since ``"first_pull"`` never revisits
          a decision.
        * ``"population"``: over every pair of arms in memory that happens to
          differ in one coordinate (:func:`imabo.mutation.coordinate_importance`).
          More pairs when the population is dense, but on a large grid most opened
          arms differ in several coordinates and contribute nothing.

        Diagnostic only; nothing in the search consults either one.
        """
        if source == "population":
            state = self.memory.get_current_state()
            return coordinate_importance(
                self.get_rewarded_arms(state), self.param_names, min_rewards
            )
        self._refresh_credits()
        return {
            name: (self._coord_effect[i][1] / self._coord_effect[i][0])
            if self._coord_effect[i][0]
            else 0.0
            for i, name in enumerate(self.param_names)
        }

    def _mutation_count(self) -> int:
        """How many coordinates this mutation changes (see ``mutation_size``)."""
        d = len(self.param_names)
        if self.mutation_size != "geometric":
            return max(1, min(int(self.mutation_size), d))
        k = 1
        while k < d and self.rng.random() < self._GEOMETRIC_P:
            k += 1
        return k

    def _add_extra_mutations(
        self,
        mutant: ArmConfig,
        primary: str,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> tuple[ArmConfig, int]:
        """Mutate ``mutation_size - 1`` further coordinates, chosen uniformly.

        Each takes the first value its ``value_rule`` offers. Returns the mutant
        and how many coordinates ended up changed -- the caller records that, since
        a multi-coordinate mutant is not a single-coordinate pair and must not feed
        the importance estimator (see :meth:`_recast`).
        """
        k = self._mutation_count()
        if k <= 1:
            return mutant, 1
        others = [n for n in self.param_names if n != primary]
        changed = 1
        for name in self.rng.sample(others, min(k - 1, len(others))):
            for value, _ in self._value_candidates(
                name, mutant, state, rewarded_arms,
                nb_pending_total, nb_rewarded_total,
            ):
                if value != mutant[name]:
                    mutant = {**mutant, name: value}
                    changed += 1
                break
        return mutant, changed

    def _global_tpe_config(
        self,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ) -> ArmConfig | None:
        """A whole configuration from the multivariate TPE (IMABO's own oracle).

        The good/bad split is :meth:`IMABO.tpe_split`, and the proposal is the
        EI-argmax of ``n_ei_candidates`` draws from the joint ``l`` density --
        i.e. exactly what ``IMOSS-TPE`` would suggest at this state. Returns None
        if the estimator cannot sample, letting the caller fall through to the
        next arm.
        """
        good, bad = self.tpe_split(
            state, rewarded_arms, nb_pending_total, nb_rewarded_total
        )
        return tpe_suggest(
            good_configs=[key_to_config(k, self.param_names) for k, _ in good],
            bad_configs=[key_to_config(k, self.param_names) for k, _ in bad],
            param_names=self.param_names,
            distributions=self.distributions,
            n_candidates=self.n_ei_candidates,
            rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
            prior_weight=self.prior_weight,
            multivariate=self.multivariate,
            weights_func=self.weights_func,
        )

    def _coordinate_order(self) -> list[int]:
        """Coordinates to try, best first.

        Under ``coord_rule="ucb"``: ``select()`` first -- so the bandit's own
        ``t``/``n`` bookkeeping is untouched -- then the rest by index, which is
        what the ``require_new_arm`` walk falls back to. Under ``"random"``: a
        uniform permutation, so the walk is equally unbiased at every position.
        """
        d = self._n_coord_choices
        if self.coord_rule == "random":
            return self.rng.sample(range(d), d)
        bandit = self.coord_bandit
        first = bandit.select()
        return [first] + [
            int(c) for c in np.argsort(-bandit.indices()) if int(c) != first
        ]

    def _register(
        self,
        mutant: ArmConfig,
        key: ArmKey,
        parent: ArmConfig,
        coord: int,
        value_idx: int | None,
        n_mutated: int = 1,
    ) -> ArmConfig:
        """Record this decision's vote, to be cast when the arm is rewarded.

        Re-proposing a config appends a second decision rather than replacing the
        first: two explore steps really did choose, possibly different,
        (coordinate, value) pairs, and each is entitled to its own vote.
        """
        decision = _Decision(
            coord=coord,
            value_idx=value_idx,
            arm_key=key,
            parent_key=config_to_key(parent, self.param_names),
            n_mutated=n_mutated,
        )
        self._pending_choice.setdefault(key, []).append(decision)
        self._last_proposal = mutant
        return mutant

    def _value_candidates(
        self,
        name: str,
        parent: ArmConfig,
        state: CurrentState,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        nb_pending_total: int,
        nb_rewarded_total: int,
    ):
        """Values to try on axis ``name``, best first, per ``value_rule``.

        Yields ``(value, value_index)`` -- the index is the value bandit's arm to
        credit under ``"ucb"``, and ``None`` for the rules with no value bandit.
        The first item is what the rule would have proposed on its own; the rest
        exist for the ``require_new_arm`` walk.
        """
        current = parent[name]
        if self.value_rule == "ucb":
            bandit = self.value_bandits[name]
            bandit.select()  # keep the bandit's own t advancing once per step
            for idx in np.argsort(-bandit.indices()):
                yield self.values[name][int(idx)], int(idx)
        elif self.value_rule == "random":
            # Every draw already excludes the parent's value; a bounded number of
            # attempts, since a continuous axis has no list to enumerate.
            for _ in range(self._RANDOM_VALUE_ATTEMPTS):
                yield mutate_value(name, current, self.distributions, self.rng), None
        else:
            good, bad = self.tpe_split(
                state, rewarded_arms, nb_pending_total, nb_rewarded_total
            )
            ranked = univariate_tpe_values(
                good_configs=[key_to_config(k, self.param_names) for k, _ in good],
                bad_configs=[key_to_config(k, self.param_names) for k, _ in bad],
                name=name,
                distribution=self.distributions[name],
                n_candidates=self.n_ei_candidates,
                rng=np.random.RandomState(self.rng.randint(0, 2**32 - 1)),
                prior_weight=self.prior_weight,
                weights_func=self.weights_func,
            )
            if self.tpe_value_pick == "sample":
                ranked = [ranked[i] for i in self.rng.sample(range(len(ranked)), len(ranked))]
            for value in ranked:
                if value != current:  # a mutation must change something
                    yield value, None
            if self.require_new_arm:
                # The draws above are concentrated where l is, so on a finite axis
                # they can all land on arms already opened. Fall back to the axis'
                # remaining values (in EI order where known, else as declared), so
                # novelty is found by MASKING what is already there rather than by
                # resampling and hoping.
                seen = set(ranked)
                for value in axis_values(self.distributions[name], self.n_points):
                    if value != current and value not in seen:
                        yield value, None

    def _select_parent(
        self,
        rewarded_arms: list[tuple[ArmKey, ArmStats]],
        state: CurrentState,
        nb_pending_total: int = 0,
        nb_rewarded_total: int = 0,
    ) -> ArmConfig:
        """The configuration to mutate -- see :func:`imabo.mutation.parent_config`."""
        return parent_config(
            self,
            self.parent_rule,
            rewarded_arms,
            state,
            nb_pending_total,
            nb_rewarded_total,
            temperature=self.temperature,
            last_proposal=self._last_proposal,
        )
