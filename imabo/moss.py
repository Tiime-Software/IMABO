import math
from typing import Literal


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
                math.log(
                    total_estimated
                    / (total_estimated**beta * arm_estimated)
                ),
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
