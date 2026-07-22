"""Per-iteration distribution of the arm each oracle would suggest"""

import copy
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from imabo import IMABO
from imabo.memory import config_to_key
from imabo.optimizer import IMABOTabFM, load_tabfm

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite_arm_distribution"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5
N_SHADOW = 10
N_JOBS = 8


class Algorithm(Enum):
    IMOSS_TPE = "IMOSS-TPE"
    IMOSS = "IMOSS-Random"
    RANDOM = "Random Search"
    IMOSS_TABFM = "IMOSS-TabFM"
    UCB_AIR = "UCB-AIR"


def algo_slug(algorithm: Algorithm) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames."""
    return algorithm.value.lower().replace(" ", "_").replace("-", "_")


def benchmark_tag(bm_id: int, noise: bool) -> str:
    """Filename prefix -- keeps different benchmarks (bm_id) and the noiseless
    ablation's files from ever colliding with (or overwriting) each other."""
    return f"rf{bm_id}" if noise else f"rf{bm_id}noiseless"


def build_optimizer(
    algorithm: Algorithm,
    search_space: dict[str, Any],
    seed: int,
    tabfm_model: Any = None,
):
    if algorithm == Algorithm.IMOSS_TPE:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            beta=BETA,
        )
    elif algorithm == Algorithm.IMOSS:
        return IMABO(
            search_space=search_space,
            seed=seed,
            multivariate=True,
            use_tpe=False,
            beta=BETA,
        )
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.IMOSS_TABFM:
        model = tabfm_model if tabfm_model is not None else load_tabfm()
        return IMABOTabFM(
            search_space=search_space,
            seed=seed,
            tabfm_model=model,
            beta=BETA,
        )
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(
            search_space=search_space,
            seed=seed,
            beta=BETA,
        )


# The oracle-proposal shadow probe (see _oracle_propose) always calls the real
# oracle (bypassing the cheap MOSS exploit branch), which is expensive for
# IMOSS-TabFM -- one TabFM fit+predict per probed iteration, no cross-
# iteration caching possible since N_SHADOW already equals TabFM's own
# refit_every. Probing every iteration costs ~10h/run; we only need this
# signal at a resolution that supports the cumulative-regret figure, not a
# per-iteration trace, so it's sampled every ORACLE_PROBE_EVERY iterations.
ORACLE_PROBE_EVERY = 100


def _oracle_propose(shadow: Any) -> Any:
    """The oracle's own raw proposal, decoupled from the optimizer's current
    explore/exploit phase.

    `IMABO.suggest()` (imabo/optimizer.py) only consults the oracle --
    uniform random for plain IMOSS, the TPE Parzen estimator for IMOSS-TPE, the
    TabFM surrogate for IMOSS-TabFM -- while in its "explore" phase; once
    `len(arms) >= t**beta` it switches to `suggest_existing`, a MOSS/UCB index
    lookup over already-pulled arms (imabo/moss.py). That mixture is what the
    *algorithm* suggests, not what the oracle itself would propose. Calling
    the oracle path directly here, every iteration regardless of phase, is
    what isolates "the distribution of the oracle every time it suggests an
    arm" from the exploit-driven convergence measured previously.

    Bypassing suggest() also means memory.pull_arm() -- and therefore
    step_counter/nb_pending -- is never touched (imabo/optimizer.py:163-164),
    so this is even safer against state leakage than the previous
    shadow.suggest() probe.
    """
    if not getattr(shadow, "use_tpe", False):
        return shadow.generate_random_config()
    state = shadow.memory.get_current_state()
    rewarded_arms = shadow.get_rewarded_arms(state)
    nb_pending_total = sum(s.nb_pending for s in state.arms.values())
    nb_rewarded_total = sum(s.nb_rewarded for s in state.arms.values())
    return shadow.suggest_new(state, rewarded_arms, nb_pending_total, nb_rewarded_total)


def _shadow_copy(opt: Any) -> Any:
    """Deep-copy an optimizer's mutable state for a disposable "what would it
    suggest right now" probe, without deep-copying (or corrupting) heavy
    read-only attributes shared across instances.

    IMABOTabFM's `_tabfm_model` is a single frozen pretrained model reused
    across every run/budget (see run_experiment below); deep-copying its
    weights on every one of thousands of iterations would be
    both unnecessary (it's never mutated) and far too slow. Pre-seeding the
    deepcopy memo with its id makes copy.deepcopy skip it and reuse the same
    reference in the copy instead.
    """
    memo = {}
    model = getattr(opt, "_tabfm_model", None)
    if model is not None:
        memo[id(model)] = model
    return copy.deepcopy(opt, memo)


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    n_shadow: int = N_SHADOW,
    oracle_probe_every: int = ORACLE_PROBE_EVERY,
) -> dict:
    """Run one seed, recording every `oracle_probe_every` iterations the
    mean/std of the true reward of `n_shadow` independent draws from the
    oracle's raw proposal at that state (see _shadow_copy, _oracle_propose),
    alongside the real-trajectory fields (regrets, simple regret, suggestion
    counts).
    """
    if algorithm == Algorithm.IMOSS_TABFM:
        opt = IMABOTabFM(
            search_space=bench.get_search_space(),
            seed=seed,
            tabfm_model=tabfm_model,
            beta=BETA,
            suggest_method="max",
            n_estimators=4,
        )
    else:
        opt = build_optimizer(algorithm, bench.get_search_space(), seed, tabfm_model)
    param_names = sorted(bench.get_search_space().keys())

    # UCB-AIR (experiments/baselines/ucb_air.py) has no oracle/exploit split --
    # it's kept in this experiment for the shared cumulative-regret comparison
    # only, so it's exempt from the oracle-proposal shadow probe below.
    has_oracle = hasattr(opt, "generate_random_config")

    regrets = []
    simple_regret_trace = []
    shadow_probe_iterations = []
    shadow_reward_mean = []
    shadow_reward_std = []
    tabfm_suggestion_probe_iterations = []
    tabfm_suggestion_predicted_rewards = []
    tabfm_suggestion_predicted_max_rewards = []
    tabfm_suggestion_true_rewards = []
    tabfm_train_rewards = []
    tabfm_candidate_probe_iterations = []
    tabfm_candidate_predicted_rewards = []
    tabfm_candidate_true_rewards = []
    suggestion_counts: Counter = Counter()
    for i in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        if has_oracle and i % oracle_probe_every == 0:
            # _oracle_propose() never calls memory.pull_arm()
            shadow = _shadow_copy(opt)

            # Piggyback on the same N_SHADOW draws to also get TabFM's own
            # predicted value for each one (see IMABOTabFM.on_suggestion) --
            # collecting onto `shadow` only, never `opt`, so the real
            # trajectory stays uninstrumented. `on_suggestion` doesn't fire
            # when suggest_new falls back to a random config (not enough
            # rewarded arms yet), which is all-or-nothing across these 10
            # draws since none of them mutate `shadow`'s state.
            probe_preds: list[tuple[Any, float, float]] = []
            if hasattr(shadow, "on_suggestion"):
                shadow.on_suggestion = (
                    lambda config, mean_pred, max_pred: probe_preds.append(
                        (config, mean_pred, max_pred)
                    )
                )

            # The full candidate pool (all n_candidates), captured on the one
            # real TabFM fit among these draws -- for a pool-wide MSE (TabFM's
            # accuracy across the candidate space), not just at its picks.
            probe_pool: list[tuple[Any, float]] = []
            if hasattr(shadow, "on_candidates_scored"):
                shadow.on_candidates_scored = lambda cands, preds: probe_pool.extend(
                    zip(cands, preds)
                )

            shadow_configs = [_oracle_propose(shadow) for _ in range(n_shadow)]
            shadow_rewards = [bench.mean_reward(c) for c in shadow_configs]
            shadow_probe_iterations.append(i)
            shadow_reward_mean.append(float(np.mean(shadow_rewards)))
            shadow_reward_std.append(float(np.std(shadow_rewards)))

            if probe_preds:
                # Raw predicted/true reward pairs, not a pre-computed metric:
                # lets any per-draw metric (squared error, signed bias,
                # anything dreamed up later) be derived downstream from
                # these two lists without rerunning the experiment.
                tabfm_suggestion_probe_iterations.append(i)
                tabfm_suggestion_predicted_rewards.append(
                    [mean_pred for _, mean_pred, _ in probe_preds]
                )
                tabfm_suggestion_predicted_max_rewards.append(
                    [max_pred for _, _, max_pred in probe_preds]
                )
                tabfm_suggestion_true_rewards.append(
                    [bench.mean_reward(config) for config, _, _ in probe_preds]
                )

                # The reward labels TabFM actually fit on at this probe (one
                # per distinct rewarded arm, the exact set _fit_surrogate
                # uses -- see _oracle_propose). Logged raw and aligned 1:1
                # with the suggestion probes above, to test whether the
                # predicted-reward collapse tracks a downward drift in this
                # training-label distribution as random exploration keeps
                # adding mostly-mediocre arms.
                shadow_state = shadow.memory.get_current_state()
                shadow_rewarded_arms = shadow.get_rewarded_arms(shadow_state)
                tabfm_train_rewards.append(
                    [float(stats.mean_reward) for _, stats in shadow_rewarded_arms]
                )

            if probe_pool:
                # Predicted (reward units) vs true reward for every candidate
                # in the scored pool -- logged raw so a pool-wide MSE (or any
                # other metric) is derivable downstream without rerunning.
                tabfm_candidate_probe_iterations.append(i)
                tabfm_candidate_predicted_rewards.append(
                    [float(pred) for _, pred in probe_pool]
                )
                tabfm_candidate_true_rewards.append(
                    [bench.mean_reward(config) for config, _ in probe_pool]
                )

        x = opt.suggest()
        y = bench(x, noise=True)
        opt.observe(y)
        regrets.append(bench.regret(x))
        suggestion_counts[config_to_key(x, param_names)] += 1
        incumbent = opt.best_config
        simple_regret_trace.append(
            bench.regret(incumbent) if incumbent is not None else bench.max_value
        )

    best = opt.best_config
    simple_regret = bench.regret(best) if best is not None else bench.max_value
    best_reward = bench.mean_reward(best) if best is not None else None

    best_key = config_to_key(best, param_names) if best is not None else None
    most_suggested_key, most_suggested_count = (
        suggestion_counts.most_common(1)[0] if suggestion_counts else (None, 0)
    )
    return {
        "regrets": regrets,
        "simple_regret_trace": simple_regret_trace,
        "simple_regrets": simple_regret,
        "shadow_probe_iterations": shadow_probe_iterations if has_oracle else None,
        "shadow_reward_mean": shadow_reward_mean if has_oracle else None,
        "shadow_reward_std": shadow_reward_std if has_oracle else None,
        "tabfm_suggestion_probe_iterations": (
            tabfm_suggestion_probe_iterations
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_suggestion_predicted_rewards": (
            tabfm_suggestion_predicted_rewards
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_suggestion_predicted_max_rewards": (
            tabfm_suggestion_predicted_max_rewards
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_suggestion_true_rewards": (
            tabfm_suggestion_true_rewards
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_train_rewards": (
            tabfm_train_rewards if algorithm == Algorithm.IMOSS_TABFM else None
        ),
        "tabfm_candidate_probe_iterations": (
            tabfm_candidate_probe_iterations
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_candidate_predicted_rewards": (
            tabfm_candidate_predicted_rewards
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "tabfm_candidate_true_rewards": (
            tabfm_candidate_true_rewards
            if algorithm == Algorithm.IMOSS_TABFM
            else None
        ),
        "best_config": best,
        "best_reward": best_reward,
        "best_config_suggestions": (
            suggestion_counts[best_key] if best_key is not None else 0
        ),
        "most_suggested_count": most_suggested_count,
        "is_best_most_suggested": best_key is not None
        and best_key == most_suggested_key,
    }


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    tabfm_model: Any = None,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run."""
    stem = (
        f"{benchmark_tag(bench.bm_id, True)}_{algo_slug(algorithm)}_{n_iterations}iters"
    )

    all_results: list[dict | None] = [None] * n_runs
    pending = []
    for i in range(n_runs):
        run_path = RESULT_DIR / f"{stem}_run{i}.json"
        if run_path.exists():
            with open(run_path) as f:
                all_results[i] = json.load(f)
            tqdm.write(f"--- {stem}_run{i} already complete, skipping ---")
        else:
            pending.append(i)

    def _one_run(i: int) -> dict:
        seed = base_seed + i
        local_bench = copy.copy(bench)
        local_bench.reset_noise(seed)
        result = run_single_experiment(
            local_bench,
            n_iterations,
            algorithm,
            seed=seed,
            tabfm_model=tabfm_model,
        )
        with open(RESULT_DIR / f"{stem}_run{i}.json", "w") as f:
            json.dump(result, f)
        return result

    if pending:
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=5)(
            delayed(_one_run)(i) for i in pending
        )
        for i, result in zip(pending, results):
            all_results[i] = result

    return all_results


def run_experiment(
    bench,
    n_runs,
    base_seed,
    n_iter,
    algorithm: Algorithm,
    n_jobs: int = N_JOBS,
) -> None:
    tabfm_model = load_tabfm() if algorithm == Algorithm.IMOSS_TABFM else None
    if tabfm_model is not None:
        print("Loaded TabFM model (once, reused across all runs/budgets).")

    print(f"\n{algorithm.value}: T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        tabfm_model=tabfm_model,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    n_runs = 10
    base_seed = 42
    n_iter = 5000
    # Three benchmarks spanning reward-noise regimes: segment (clean),
    # credit-g (noisy), numerai28.6 (hard).
    bm_ids = [146822, 31, 167120]
    # The IMOSS proposal-oracle family for the oracle-distribution shadow
    # probe, plus UCB-AIR kept only for the shared cumulative-regret grid
    # (run_single_experiment skips the shadow probe for it -- see has_oracle).
    algorithms = [
        Algorithm.IMOSS,
        Algorithm.IMOSS_TPE,
        Algorithm.IMOSS_TABFM,
        Algorithm.UCB_AIR,
    ]

    for bm_id in bm_ids:
        bench = RFTabularFiniteBenchmark(bm_id=bm_id)
        print(
            f"RF tabular finite benchmark (OpenML task {bm_id}): "
            f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}"
        )
        for algorithm in algorithms:
            run_experiment(bench, n_runs, base_seed, n_iter, algorithm)
