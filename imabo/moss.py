import math
from typing import Literal

import numpy as np


class UCB1:
    """UCB1 over a fixed finite set of choices.

    ``bonus="hoeffding"`` (default) is textbook UCB1:
    ``mean_i + sqrt(alpha * 2 * log(t) / n_i)``, every choice pulled once first.
    That width is not variance-adapted, so on a reward range much narrower than 1
    it dominates the mean differences and the index degenerates towards
    round-robin (lower ``alpha`` to counteract that).

    ``bonus="kl_scaled"`` is KL-UCB made usable on credits that are neither in
    [0, 1] nor Bernoulli-like -- notably an *improvement* credit, which is centred
    near zero and often negative. The arm means are mapped affinely onto [0, 1] by
    their own observed spread across arms, the KL index is taken there, and arms
    are ranked in that space. The map is shared by all arms, so it cannot reorder
    them; what it changes is the width, which becomes scale-free (it responds to
    how far apart the arms are, not to the units of the reward). A plain affine map
    onto a fixed [-1, 1] would instead park a tightly clustered signal at mean 0.5,
    where the Bernoulli variance KL-UCB assumes is maximal, and so widen the
    interval exactly where the signal is smallest.

    ``bonus="kl"`` uses the KL-UCB index (:func:`kl_ucb`) instead, which adapts to
    the Bernoulli variance ``p(1-p)`` and so narrows on its own where Hoeffding
    cannot: at ``t=112, n=25, mean=0.9`` -- the coordinate bandit's actual regime
    in :class:`imabo.coord_ucb.IMABOCoordUCB` -- the widths are 0.61 (Hoeffding)
    versus 0.09 (KL), against a signal of 0.09-0.16 to resolve. ``alpha`` is
    ignored in this mode, and rewards must lie in [0, 1] (so it does not fit a
    credit rule that can go negative, such as ``"improvement"``).

    ``discount`` < 1 turns this into discounted-UCB (Garivier & Moulines 2008) in
    whichever of the three modes is selected: every :meth:`update` first multiplies
    all counts and sums by ``discount``, so an observation cast ``k`` votes ago
    carries weight ``discount ** k`` and the effective sample size saturates at
    ``1 / (1 - discount)``. The log term switches from the number of selections to
    that discounted total, which is what keeps the bonus from vanishing once the
    counts stop growing. Use it when the reward attached to a choice *changes* over
    the run rather than merely being estimated better -- as it does for
    :class:`imabo.coord_ucb.IMABOCoordUCB`'s coordinate bandit, where the credit of
    mutating coordinate ``i`` is the gain over the current incumbent, and drops off
    once the incumbent already holds a good value on that axis.

    Shared by the Hier-MAB / AutoRAG-HP baseline (both of its levels) and by
    :class:`imabo.coord_ucb.IMABOCoordUCB`'s coordinate-selection bandit, so
    "the same bandit as Hier-MAB" is literally the same code.
    """

    # Below this spread of arm means, "kl_scaled" stops rescaling (the division
    # would amplify pure noise into a full-range signal).
    _MIN_SPAN = 1e-9

    def __init__(
        self,
        n_choices: int,
        alpha: float = 1.0,
        bonus: Literal["hoeffding", "kl", "kl_scaled"] = "hoeffding",
        discount: float = 1.0,
    ):
        if bonus not in ("hoeffding", "kl", "kl_scaled"):
            raise ValueError(f"Invalid bonus: {bonus!r}")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {discount}")
        self.n = np.zeros(n_choices, dtype=np.float64)
        self.sum = np.zeros(n_choices, dtype=np.float64)
        self.alpha = alpha
        self.bonus = bonus
        self.discount = discount
        self.t = 0
        # Votes cast so far. A vote cast at epoch e has since been discounted
        # (self.epoch - e) times, which is what revise() has to reproduce.
        self.epoch = 0

    def select(self) -> int:
        self.t += 1
        unpulled = np.flatnonzero(self.n == 0)
        if unpulled.size:
            return int(unpulled[0])
        return int(np.argmax(self.indices()))

    def indices(self) -> np.ndarray:
        """The index of every choice: ``mean + bonus``, ``+inf`` if never pulled.

        :meth:`select` is the argmax of this. Exposed so a caller can fall back to
        the *next* best choice when its first one is unusable -- e.g. a proposal
        rule that must not return a configuration it has already opened (see
        :class:`imabo.coord_ucb.IMABOCoordUCB`'s ``require_new_arm``). Call it
        after ``select`` so both see the same ``t``.
        """
        pulled = self.n > 0
        index = np.full(self.n.size, np.inf)
        if not pulled.any():
            # Nothing credited yet: every choice is maximally uncertain. Returning
            # early also keeps the rescaling below off a zero-size array (votes can
            # lag well behind select() calls -- an improvement credit is only cast
            # once both the child and its parent have an estimate).
            return index
        # Guard the division for never-pulled choices only -- they are masked out
        # below anyway. Clamping every count up to 1 would be wrong under
        # discounting, where a stale choice's weight legitimately decays past 1:
        # its mean would be pulled towards 0 instead of merely being forgotten.
        n = np.where(pulled, self.n, 1.0)
        mean = self.sum / n
        # Discounted: the number of selections keeps growing while the counts
        # saturate, so log(t) would inflate every bonus without bound. The
        # discounted total (<= 1/(1-discount)) is the matching horizon.
        horizon = max(2.0, float(self.n.sum()) if self.discount < 1.0 else self.t)
        if self.bonus in ("kl", "kl_scaled"):
            t = horizon
            scaled = mean
            if self.bonus == "kl_scaled":
                lo, hi = mean[pulled].min(), mean[pulled].max()
                span = hi - lo
                # A degenerate spread carries no ranking information anyway, so
                # park every arm mid-interval and let n alone order the widths.
                scaled = (
                    (mean - lo) / span if span > self._MIN_SPAN
                    else np.full_like(mean, 0.5)
                )
            for i in np.flatnonzero(pulled):
                index[i] = kl_ucb(
                    min(max(scaled[i], 0.0), 1.0), float(self.n[i]), t
                )
            return index
        bonus = np.sqrt(self.alpha * 2.0 * math.log(horizon) / n)
        index[pulled] = mean[pulled] + bonus[pulled]
        return index

    def update(self, idx: int, reward: float) -> int:
        """Cast one vote of weight 1 on ``idx``; returns the epoch it was cast at.

        Hand that epoch back to :meth:`revise` so a correction is applied at the
        same (by then discounted) weight as the vote it corrects.
        """
        if self.discount < 1.0:
            self.n *= self.discount
            self.sum *= self.discount
        self.epoch += 1
        self.n[idx] += 1.0
        self.sum[idx] += reward
        return self.epoch

    def revise(self, idx: int, delta: float, epoch: int | None = None) -> None:
        """Adjust an arm's accumulated reward WITHOUT counting a new pull.

        For a caller that credits one decision per pull but keeps sharpening its
        estimate of that decision's outcome (see
        :class:`imabo.coord_ucb.IMABOCoordUCB`'s ``credit_rule``): the arm keeps
        one "vote" per decision while the value of that vote is corrected as more
        observations of the resulting configuration arrive.

        ``epoch`` is the value :meth:`update` returned for the vote being revised.
        Under discounting the correction is scaled down by the decay that vote has
        taken on since, so the revised total is exactly what would have been
        accumulated had the final value been known at vote time.
        """
        if self.discount < 1.0 and epoch is not None:
            delta *= self.discount ** (self.epoch - epoch)
        self.sum[idx] += delta

    @property
    def means(self) -> np.ndarray:
        return self.sum / np.where(self.n > 0, self.n, 1.0)


class EXP3:
    """EXP3 / EXP3.S over a fixed finite set of choices -- drop-in for :class:`UCB1`.

    UCB1 (and KL-UCB) assume each choice has a *fixed* reward distribution that is
    merely being estimated better over time. That assumption fails for
    :class:`imabo.coord_ucb.IMABOCoordUCB`'s coordinate bandit: the credit of
    mutating coordinate ``i`` is measured against the current incumbent, so it
    decays as soon as the incumbent holds a good value on that axis. EXP3 makes no
    stationarity assumption at all -- it competes with the best fixed choice under
    an adversary -- and ``mixing`` > 0 turns it into EXP3.S (Auer et al. 2002),
    which competes with the best *sequence* of choices and is the variant that
    actually tracks a moving optimum.

    Selection is stochastic: ``p_i = (1 - gamma) * softmax(eta * S_i)_i + gamma / K``
    where ``S_i`` is the importance-weighted cumulative reward estimate. Rewards
    must lie in [0, 1] (the estimator ``r / p_i`` is unbounded below otherwise).

    ``eta`` and ``gamma`` default to the anytime schedules
    ``eta_t = eta_scale * sqrt(log K / (K t))`` and ``gamma_t = min(1/2, K eta_t)``
    over the number of votes ``t`` cast so far; ``eta_scale`` pushes the rate past
    what the worst-case bound licenses, which the short horizons here may want.

    ``feedback="loss"`` (the default) accumulates ``-(1 - r) / p_i`` instead of
    ``+r / p_i``. Both are unbiased and rank the choices the same way in
    expectation, but the variance of the estimator scales with the magnitude of
    what is divided by ``p_i``, and the rewards here are validation accuracies
    sitting near 0.9 -- so the loss version divides a number near 0.1 rather than
    one near 0.9 and is roughly an order of magnitude quieter. ``"reward"`` is the
    textbook form.
    """

    def __init__(
        self,
        n_choices: int,
        eta: float | None = None,
        gamma: float | None = None,
        mixing: float = 0.0,
        eta_scale: float = 1.0,
        feedback: Literal["reward", "loss"] = "loss",
        rng: np.random.Generator | None = None,
    ):
        if n_choices < 1:
            raise ValueError(f"n_choices must be >= 1, got {n_choices}")
        if not 0.0 <= mixing <= 1.0:
            raise ValueError(f"mixing must be in [0, 1], got {mixing}")
        if feedback not in ("reward", "loss"):
            raise ValueError(f"Invalid feedback: {feedback!r}")
        self.k = n_choices
        self.eta = eta
        self.gamma = gamma
        self.mixing = mixing
        self.eta_scale = eta_scale
        self.feedback = feedback
        self.rng = rng if rng is not None else np.random.default_rng()
        # Importance-weighted cumulative reward estimate per choice.
        self.s = np.zeros(n_choices, dtype=np.float64)
        # Selection probability each choice last had when it was picked, used as
        # the importance weight for the vote that selection eventually produces.
        # (A vote can arrive several steps after the selection that caused it, and
        # under require_new_arm the walk may land on a different choice than the
        # one select() named -- so the weight is stored per choice, not globally.)
        self.p_at_select = np.full(n_choices, 1.0 / n_choices, dtype=np.float64)
        self.n = np.zeros(n_choices, dtype=np.float64)
        self.sum = np.zeros(n_choices, dtype=np.float64)
        self.t = 0
        self.epoch = 0

    def _rates(self) -> tuple[float, float]:
        """(eta, gamma) at the current number of votes."""
        u = max(1, self.epoch)
        eta = (
            self.eta
            if self.eta is not None
            else self.eta_scale * math.sqrt(math.log(max(self.k, 2)) / (self.k * u))
        )
        gamma = self.gamma if self.gamma is not None else min(0.5, self.k * eta)
        return eta, gamma

    def probabilities(self) -> np.ndarray:
        eta, gamma = self._rates()
        z = eta * self.s
        w = np.exp(z - z.max())
        p = (1.0 - gamma) * (w / w.sum()) + gamma / self.k
        return p / p.sum()

    def select(self) -> int:
        self.t += 1
        p = self.probabilities()
        idx = int(self.rng.choice(self.k, p=p))
        self.p_at_select[idx] = p[idx]
        return idx

    def indices(self) -> np.ndarray:
        """Selection probabilities -- the ranking a caller walks when its first
        choice is unusable (see :meth:`UCB1.indices`)."""
        return self.probabilities()

    def _signed(self, reward: float) -> float:
        """The quantity accumulated per unit importance weight."""
        return reward if self.feedback == "reward" else reward - 1.0

    def update(self, idx: int, reward: float) -> float:
        """Cast a vote; returns the importance weight used, for :meth:`revise`."""
        self.epoch += 1
        weight = 1.0 / max(self.p_at_select[idx], 1e-12)
        if self.mixing > 0.0:
            # EXP3.S: bleed each estimate towards the mean so no choice can be
            # ruled out permanently -- what lets the weights follow a moving best.
            self.s += self.mixing * (self.s.mean() - self.s)
        self.s[idx] += self._signed(reward) * weight
        self.n[idx] += 1.0
        self.sum[idx] += reward
        return weight

    def revise(self, idx: int, delta: float, epoch: float | None = None) -> None:
        """Correct a vote's value without casting a new one (see
        :meth:`UCB1.revise`). ``epoch`` is the weight :meth:`update` returned."""
        weight = epoch if epoch is not None else 1.0 / max(self.p_at_select[idx], 1e-12)
        # A delta on the reward is the same delta on the loss (they differ by a
        # constant), so no _signed() here.
        self.s[idx] += delta * weight
        self.sum[idx] += delta

    @property
    def means(self) -> np.ndarray:
        return self.sum / np.where(self.n > 0, self.n, 1.0)


def moss_anytime(
    *,
    mean_reward: float,
    n_arms: int,
    step_counter: int,
    nb_rewarded_arm: int,
    alpha: float = 0.1,
    beta: float = 0.8,
    switch_strategy: Literal["beta", "delayed"] = "beta",
    reward_frequency: float | None = None,
    nb_rewarded_total: int | None = None,
    nb_pending_total: int | None = None,
    nb_pending_arm: int | None = None,
) -> float:
    """Compute the MOSS-anytime upper confidence bound for an arm.

    In the "beta" (synchronous) mode, this implements the standard formula:
        mu_hat + sqrt((1+alpha)/2 * max(0, log(t / (K * n_x))) / n_x)

    In the "delayed" (asynchronous) mode, it replaces t with an estimated
    effective count that accounts for pending (unobserved) pulls.

    Args:
        mean_reward: Empirical mean reward of the arm.
        n_arms: Current number of arms K_t = |M_t|.
        step_counter: Total number of steps t.
        nb_rewarded_arm: Number of observed rewards for this arm (n_x).
        alpha: Tuning parameter (default 0.1).
        beta: Exponent for the switching schedule.
        switch_strategy: "beta" for synchronous, "delayed" for asynchronous.
        reward_frequency: Estimated fraction of trials receiving a reward.
        nb_rewarded_total: Total observed rewards across all arms.
        nb_pending_total: Total pending observations across all arms.
        nb_pending_arm: Pending observations for this specific arm.

    Returns:
        MOSS upper confidence bound for the arm.
    """
    alpha_term = (1 + alpha) / 2

    if switch_strategy == "delayed":
        assert reward_frequency is not None
        assert nb_pending_total is not None
        assert nb_pending_arm is not None
        assert nb_rewarded_total is not None

        total_estimated = nb_rewarded_total + reward_frequency * nb_pending_total
        arm_estimated = nb_rewarded_arm + reward_frequency * nb_pending_arm + 1

        bound = math.sqrt(
            alpha_term
            * max(
                0,
                math.log(total_estimated / (total_estimated**beta * arm_estimated)),
            )
            / arm_estimated
        )
    else:
        nb_rewarded_arm = max(nb_rewarded_arm, 1)
        bound = math.sqrt(
            alpha_term
            * max(0, math.log(step_counter / (n_arms * nb_rewarded_arm)))
            / nb_rewarded_arm
        )

    return mean_reward + bound


def ucb(
    *,
    mean: float,
    nb_rewarded_arm: int,
    total_pulls: int,
    ucb_c: float = 1.5,
    bonus_type: Literal["ucb", "lcb"] = "ucb",
) -> float:
    """Classic UCB1-style index (alternative to MOSS).

    Args:
        mean: Empirical mean reward of the arm.
        nb_rewarded_arm: Number of observed rewards for this arm.
        total_pulls: Total number of pulls across all arms.
        ucb_c: Exploration constant.
        bonus_type: "ucb" adds the bonus, "lcb" subtracts it.

    Returns:
        The (optimistic or pessimistic) confidence bound.
    """
    bonus = ucb_c * math.sqrt(
        2.0 * math.log(max(2.0, total_pulls)) / max(1, nb_rewarded_arm)
    )
    return mean + bonus if bonus_type == "ucb" else mean - bonus


def kl_divergence(p: float, q: float) -> float:
    """KL divergence between two Bernoulli distributions, KL(p || q)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb(mean: float, pulls: int, t: int, c: float = 0.0) -> float:
    """KL-UCB index for Bernoulli bandits (alternative to MOSS).

    Returns the largest q in [mean, 1] such that::

        pulls * KL(mean, q) <= log(t) + c * log(log(t))

    Args:
        mean: Empirical mean of the arm.
        pulls: Number of pulls of the arm.
        t: Total number of steps (time horizon).
        c: Exploration constant (0 for asymptotic optimality).

    Returns:
        KL-UCB upper confidence bound.
    """
    pulls = max(pulls, 1)

    rhs = (math.log(t) + c * math.log(max(math.log(t), 1))) / pulls
    lo, hi = mean, 1.0 - 1e-12

    if mean > 1 - 1e-6:
        return 1.0

    for _ in range(40):
        mid = (lo + hi) / 2
        if kl_divergence(mean, mid) <= rhs:
            lo = mid
        else:
            hi = mid
    return lo


def ucb_siri(
    mean: float,
    nb_arm_pulls: int,
    t: int,
    c: float = 1.5,
    delta: float = 0.05,
    bonus_type: Literal["ucb", "lcb"] = "ucb",
) -> float:
    """Anytime UCB index with an explicit confidence level delta.

    Alternative exploitation index (used by FiniteIMABO's default pull strategy).

    Args:
        mean: Empirical mean reward of the arm.
        nb_arm_pulls: Number of pulls of the arm.
        t: Total budget / time horizon.
        c: Exploration constant.
        delta: Confidence level.
        bonus_type: "ucb" adds the bonus, "lcb" subtracts it.

    Returns:
        The (optimistic or pessimistic) confidence bound.
    """
    confidence_bound = math.sqrt(
        math.log(t / delta / nb_arm_pulls) / max(1, nb_arm_pulls)
    )

    return (
        mean + c * confidence_bound
        if bonus_type == "ucb"
        else mean - c * confidence_bound
    )
