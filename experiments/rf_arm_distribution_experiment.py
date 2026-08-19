"""Per-iteration distribution of the arm each oracle would suggest.

One file runs the RF-tabular-bandit experiment for every method, including both
foundation-model oracles: Google TabFM (``IMOSS-TabFM``) and Prior-Labs
TabPFN-3 (``IMOSS-TabPFN``). All runs are checkpointed per seed and resumable,
so reruns skip finished seeds.

Reproduce (each command resumable; the surrogate-free baselines are shared by
both foundation-model figures):

    # run everything (all algorithms x all benchmarks), then plot both figures:
    python -m experiments.rf_arm_distribution_experiment

    # run a single method:
    python -m experiments.rf_arm_distribution_experiment --algorithm IMOSS-TabPFN

    # TabPFN per-pull variant (one row per pull, no per-arm averaging, KV-cache);
    # written to distinct '..._pull' files/plots so it never clobbers per-arm:
    python -m experiments.rf_arm_distribution_experiment \
        --algorithm IMOSS-TabPFN --fit-granularity pull

    # only (re)plot a foundation model's figure from existing result JSONs:
    python -m experiments.rf_arm_distribution_experiment --plot-only --foundation tabpfn

The TabPFN arm needs the experiment extra (``pip install -e ".[experiments]"``).
"""

import argparse
import copy
import json
import time
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm

from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.rf_tabular_bandit import RFTabularFiniteBenchmark
from imabo import (
    IMABO,
    IMOSSTPE,
    CandidatePool,
    IMOSSMutateKLTPE,
    IMOSSRandom,
    IMOSSTabFM,
    IMOSSTabPFN,
    config_to_key,
    load_tabfm,
    load_tabpfn,
)

# Pick up TABPFN_TOKEN (PriorLabs license/API key) and friends from .env, as
# the HotpotQA experiment already does for OPENROUTER_API_KEY.
load_dotenv()

RESULT_DIR = Path(__file__).parent.parent / "results" / "hpo_finite_arm_distribution"
RESULT_DIR.mkdir(exist_ok=True)

BETA = 0.5
N_SHADOW = 10
N_JOBS = 8


def _silence_known_warnings() -> None:
    """Mute the noisy, harmless ``FutureWarning`` TabPFN's internal
    ``ColumnTransformer`` raises on every fit (about ``force_int_remainder_cols``
    in scikit-learn >=1.6) -- one block per TabPFN fit, saying nothing about the
    experiment. Set as a process-global filter (safe under the joblib threading
    backend); re-applied after ``load_tabpfn`` since importing tabpfn/sklearn can
    reset the warnings registry. A no-op for the non-TabPFN algorithms.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"The format of the columns of the 'remainder' transformer",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"sklearn\.compose\._column_transformer",
    )


_silence_known_warnings()


class Algorithm(Enum):
    IMOSS_TPE = "IMOSS-TPE"
    # TPE with a univariate Parzen estimator (multivariate=False): one
    # independent density per hyperparameter, i.e. a factored proposal.
    IMOSS_TPE_UNI = "IMOSS-TPE-univ"
    IMOSS = "IMOSS-Random"
    RANDOM = "Random Search"
    IMOSS_TABFM = "IMOSS-TabFM"
    IMOSS_TABPFN = "IMOSS-TabPFN"
    # The two tuned explore oracles (see winning_configs.pdf). Both mutate the
    # incumbent rather than sampling globally; they differ in what picks the
    # mutation -- a KL-UCB coordinate bandit plus a univariate TPE, or TabPFN
    # ranking a 100-candidate pool.
    IMOSS_MUTATE_KLXTPE = "IMOSS-mutate-KLxTPE"
    IMOSS_TABPFN_TUNED = "IMOSS-TabPFN-tuned"
    UCB_AIR = "UCB-AIR"


def algo_slug(
    algorithm: Algorithm,
    fit_granularity: str = "arm",
    acquisition: str = "quantile",
    quantile: float = 0.99,
) -> str:
    """Filesystem-safe label, used for per-algorithm result filenames.

    IMOSS-TabPFN's default configuration -- quantile acquisition at the
    0.99 level -- keeps the plain ``imoss_tabpfn`` slug. Non-default variants
    get distinct suffixes -- ``_pull`` (per-pull tables), ``_q<q>`` (quantile
    at a non-default level), ``_ucb_q<q>`` (moment UCB at the
    normality-equivalent kappa) -- so their result
    JSONs (and plotted series) live alongside, and never clobber, the
    default's. The surrogate-free arms are unaffected by these options, so
    they keep their usual slugs and are reused across all variants and both
    foundation models.
    """
    slug = algorithm.value.lower().replace(" ", "_").replace("-", "_")
    if algorithm == Algorithm.IMOSS_TABPFN and fit_granularity == "pull":
        slug += "_pull"
    if algorithm == Algorithm.IMOSS_TABPFN:
        if acquisition == "ucb":
            slug += f"_ucb_q{quantile:g}"
        elif quantile != 0.99:
            slug += f"_q{quantile:g}"
    return slug


def benchmark_tag(bm_id: int, noise: bool) -> str:
    """Filename prefix -- keeps different benchmarks (bm_id) and the noiseless
    ablation's files from ever colliding with (or overwriting) each other."""
    return f"rf{bm_id}" if noise else f"rf{bm_id}noiseless"


def build_optimizer(
    algorithm: Algorithm,
    search_space: dict[str, Any],
    seed: int,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
):
    if algorithm == Algorithm.IMOSS_TPE:
        return IMOSSTPE(search_space, beta=BETA, seed=seed, multivariate=True)
    elif algorithm == Algorithm.IMOSS_TPE_UNI:
        return IMOSSTPE(search_space, beta=BETA, seed=seed, multivariate=False)
    elif algorithm == Algorithm.IMOSS:
        return IMOSSRandom(search_space, beta=BETA, seed=seed)
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=search_space, seed=seed)
    elif algorithm == Algorithm.IMOSS_TABFM:
        model = tabfm_model if tabfm_model is not None else load_tabfm()
        return IMOSSTabFM(search_space, beta=BETA, seed=seed, model=model)
    elif algorithm == Algorithm.IMOSS_TABPFN:
        model = tabpfn_model if tabpfn_model is not None else load_tabpfn()
        # The untuned baseline: a uniform candidate pool, the shipped shortlist
        # depth and the 0.99 quantile. Spelled out because the class defaults are
        # now the tuned configuration.
        return IMOSSTabPFN(
            search_space,
            beta=BETA,
            seed=seed,
            model=model,
            pool=CandidatePool(source="uniform", scale=None),
            refit_every=10,
            quantile=0.99,
        )
    elif algorithm == Algorithm.IMOSS_TABPFN_TUNED:
        model = tabpfn_model if tabpfn_model is not None else load_tabpfn()
        return IMOSSTabPFN(
            search_space,
            beta=BETA,
            seed=seed,
            model=model,
            # Everything else is a TabPFNOracle default: those defaults are the
            # tuned configuration.
        )
    elif algorithm == Algorithm.IMOSS_MUTATE_KLXTPE:
        return IMOSSMutateKLTPE(search_space, beta=BETA, seed=seed)
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(
            search_space=search_space,
            seed=seed,
            beta=BETA,
        )


# Arms backed by TabPFN. The model must be warmed ONCE up front: it is loaded
# lazily otherwise, and eight worker threads racing to load it hard-crashes the
# process on Apple-silicon MPS with no traceback (see imabo.oracles.tabpfn_oracle).
_TABPFN_ALGORITHMS = (Algorithm.IMOSS_TABPFN, Algorithm.IMOSS_TABPFN_TUNED)


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
    `len(arms) >= t**beta` it switches to `AllocationPolicy.select`, a MOSS/UCB
    index lookup over already-pulled arms (imabo/policies/). That mixture is what the
    *algorithm* suggests, not what the oracle itself would propose. Calling
    the oracle path directly here, every iteration regardless of phase, is
    what isolates "the distribution of the oracle every time it suggests an
    arm" from the exploit-driven convergence measured previously.

    Bypassing suggest() also means memory.pull() -- and therefore the round
    counter and the pending counts -- is never touched, so this is even safer
    against state leakage than the previous shadow.suggest() probe.
    """
    return shadow.propose()


def _shadow_copy(opt: Any) -> Any:
    """Deep-copy an optimizer's mutable state for a disposable "what would it
    suggest right now" probe, without deep-copying (or corrupting) heavy
    read-only attributes shared across instances.

    TabFMOracle's `_model` is a single frozen pretrained model reused across every
    run/budget (see run_experiment below); deep-copying its weights on every one of
    thousands of iterations would be both unnecessary (it is never mutated) and far
    too slow. Pre-seeding the deepcopy memo with its id makes copy.deepcopy skip it
    and reuse the same reference in the copy instead. TabPFNOracle's `_model` (a
    small shared settings dict) is skipped the same way, for parity.
    """
    memo = {}
    model = getattr(getattr(opt, "oracle", None), "_model", None)
    if model is not None:
        memo[id(model)] = model
    return copy.deepcopy(opt, memo)


def run_single_experiment(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    seed: int = 42,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_shadow: int = N_SHADOW,
    oracle_probe_every: int = ORACLE_PROBE_EVERY,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
) -> dict:
    """Run one seed, recording every `oracle_probe_every` iterations the
    mean/std of the true reward of `n_shadow` independent draws from the
    oracle's raw proposal at that state (see _shadow_copy, _oracle_propose),
    alongside the real-trajectory fields (regrets, simple regret, suggestion
    counts).

    The surrogate diagnostics (the ``tabfm_*`` fields) fire for either
    foundation-model oracle -- IMOSS-TabFM or IMOSS-TabPFN -- and keep the
    ``tabfm_*`` names verbatim so the plotting code reads both unchanged.
    ``fit_granularity`` / ``max_num_rows`` / ``acquisition`` / ``quantile``
    only affect IMOSS-TabPFN (see :class:`imabo.oracles.tabpfn_oracle.TabPFNOracle`):
    ``"arm"`` (default) fits one row per arm at its mean reward, ``"pull"``
    one row per individual pull (no averaging, KV-cache on); ``acquisition``
    is the ``quantile`` level of the predictive distribution (``"quantile"``,
    default) or the moment UCB at the normality-equivalent kappa (``"ucb"``)
    -- ``quantile`` is the single exploration knob for both.
    """
    if algorithm == Algorithm.IMOSS_TABFM:
        opt = IMOSSTabFM(
            bench.get_search_space(),
            beta=BETA,
            seed=seed,
            model=tabfm_model,
            suggest_method="max",
            n_estimators=4,
        )
    elif algorithm == Algorithm.IMOSS_TABPFN:
        # Per-pull tables grow to O(#pulls) rows, so lift the default in-context
        # row cap well above the per-arm default (200) unless told otherwise.
        if max_num_rows is None:
            effective_max_rows = 10000 if fit_granularity == "pull" else 200
        else:
            effective_max_rows = max_num_rows
        opt = IMOSSTabPFN(
            bench.get_search_space(),
            beta=BETA,
            seed=seed,
            model=tabpfn_model,
            n_estimators=4,
            fit_granularity=fit_granularity,
            max_num_rows=effective_max_rows,
            acquisition=acquisition,
            quantile=quantile,
        )
    else:
        opt = build_optimizer(
            algorithm, bench.get_search_space(), seed, tabfm_model, tabpfn_model
        )
    param_names = sorted(bench.get_search_space().keys())

    # Both foundation-model oracles log the surrogate diagnostics below (under
    # the shared ``tabfm_*`` field names).
    is_surrogate = algorithm in (Algorithm.IMOSS_TABFM, Algorithm.IMOSS_TABPFN)

    # UCB-AIR (experiments/baselines/ucb_air.py) has no oracle/exploit split --
    # it's kept in this experiment for the shared cumulative-regret comparison
    # only, so it's exempt from the oracle-proposal shadow probe below.
    has_oracle = isinstance(opt, IMABO)

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
    tabfm_candidate_configs = []
    tabfm_candidate_predicted_rewards = []
    tabfm_candidate_true_rewards = []
    suggestion_counts: Counter = Counter()
    for i in tqdm(range(n_iterations), desc=algorithm.value, leave=False):
        if has_oracle and n_shadow > 0 and i % oracle_probe_every == 0:
            # _oracle_propose() never calls memory.pull_arm()
            shadow = _shadow_copy(opt)

            # Piggyback on the same N_SHADOW draws to also get TabFM's own
            # predicted value for each one (see TabFMOracle.on_suggestion) --
            # collecting onto `shadow` only, never `opt`, so the real
            # trajectory stays uninstrumented. `on_suggestion` doesn't fire
            # when the oracle falls back to a random config (not enough
            # rewarded arms yet), which is all-or-nothing across these 10
            # draws since none of them mutate `shadow`'s state.
            probe_preds: list[tuple[Any, float, float]] = []
            if hasattr(shadow.oracle, "on_suggestion"):
                shadow.oracle.on_suggestion = (
                    lambda config, mean_pred, max_pred: probe_preds.append(
                        (config, mean_pred, max_pred)
                    )
                )

            # The full candidate pool (all n_candidates), captured on the one
            # real TabFM fit among these draws -- for a pool-wide MSE (TabFM's
            # accuracy across the candidate space), not just at its picks.
            probe_pool: list[tuple[Any, float]] = []
            if hasattr(shadow.oracle, "on_candidates_scored"):
                shadow.oracle.on_candidates_scored = lambda cands, preds: probe_pool.extend(
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
                # per distinct rewarded arm, the exact set the oracle's fit
                # uses -- see _oracle_propose). Logged raw and aligned 1:1
                # with the suggestion probes above, to test whether the
                # predicted-reward collapse tracks a downward drift in this
                # training-label distribution as random exploration keeps
                # adding mostly-mediocre arms.
                shadow_state = shadow.state
                tabfm_train_rewards.append(
                    [
                        float(stats.mean_reward)
                        for _, stats in IMABO.rewarded_arms(shadow_state)
                    ]
                )

            if probe_pool:
                # Config + predicted (reward units) + true reward for every
                # candidate in the scored pool -- logged raw so a pool-wide
                # MSE, or a per-(depth,features)-cell MSE map of where TabFM's
                # surrogate is inaccurate, is derivable downstream without
                # rerunning. Configs stored as key lists (param_names order).
                tabfm_candidate_probe_iterations.append(i)
                tabfm_candidate_configs.append(
                    [[config[p] for p in param_names] for config, _ in probe_pool]
                )
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
            tabfm_suggestion_probe_iterations if is_surrogate else None
        ),
        "tabfm_suggestion_predicted_rewards": (
            tabfm_suggestion_predicted_rewards if is_surrogate else None
        ),
        "tabfm_suggestion_predicted_max_rewards": (
            tabfm_suggestion_predicted_max_rewards if is_surrogate else None
        ),
        "tabfm_suggestion_true_rewards": (
            tabfm_suggestion_true_rewards if is_surrogate else None
        ),
        "tabfm_train_rewards": (tabfm_train_rewards if is_surrogate else None),
        "tabfm_candidate_probe_iterations": (
            tabfm_candidate_probe_iterations if is_surrogate else None
        ),
        "tabfm_candidate_configs": (tabfm_candidate_configs if is_surrogate else None),
        "tabfm_candidate_predicted_rewards": (
            tabfm_candidate_predicted_rewards if is_surrogate else None
        ),
        "tabfm_candidate_true_rewards": (
            tabfm_candidate_true_rewards if is_surrogate else None
        ),
        "best_config": best,
        "best_reward": best_reward,
        "best_config_suggestions": (
            suggestion_counts[best_key] if best_key is not None else 0
        ),
        "most_suggested_count": most_suggested_count,
        "is_best_most_suggested": best_key is not None
        and best_key == most_suggested_key,
        # Full real-trajectory pull distribution over the search space (every
        # opt.suggest(), explore + exploit), keyed by config. Tuples aren't
        # JSON keys, so stored as [config_key_list, count] pairs. The 2D
        # pulls-overlay plot reconstructs (param -> value) via param_names.
        "suggestion_counts": [
            [list(key), count] for key, count in suggestion_counts.items()
        ],
        "param_names": param_names,
    }


def run_multiple_experiments(
    bench: RFTabularFiniteBenchmark,
    n_iterations: int,
    algorithm: Algorithm,
    n_runs: int = 10,
    base_seed: int = 42,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_jobs: int = N_JOBS,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    n_shadow: int = N_SHADOW,
) -> list[dict]:
    """Run multiple independent runs of a single algorithm, checkpointed per
    run."""
    stem = (
        f"{benchmark_tag(bench.bm_id, True)}"
        f"_{algo_slug(algorithm, fit_granularity, acquisition, quantile)}"
        f"_{n_iterations}iters"
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
            tabpfn_model=tabpfn_model,
            fit_granularity=fit_granularity,
            max_num_rows=max_num_rows,
            acquisition=acquisition,
            quantile=quantile,
            n_shadow=n_shadow,
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
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    n_jobs: int = N_JOBS,
    fit_granularity: str = "arm",
    max_num_rows: int | None = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
    n_shadow: int = N_SHADOW,
) -> None:
    # Load/warm up the surrogate model once (both loaders are memoized, so
    # passing a preloaded model in from the caller avoids re-warming per bench).
    if algorithm == Algorithm.IMOSS_TABFM and tabfm_model is None:
        tabfm_model = load_tabfm()
        print("Loaded TabFM model (once, reused across all runs/budgets).")
    if algorithm in _TABPFN_ALGORITHMS and tabpfn_model is None:
        tabpfn_model = load_tabpfn()
        _silence_known_warnings()  # importing tabpfn/sklearn can reset filters
        print("TabPFN-3 ready (checkpoint cached; reused across all runs/budgets).")

    gran = ""
    if algorithm == Algorithm.IMOSS_TABPFN:
        acq = (
            f"quantile q={quantile:g}"
            if acquisition == "quantile"
            else f"ucb q={quantile:g}"
        )
        gran = f" [{fit_granularity}, {acq}]"
    print(f"\n{algorithm.value}{gran}: T={n_iter}, {n_runs} runs...")
    run_multiple_experiments(
        bench,
        n_iter,
        algorithm,
        n_runs=n_runs,
        base_seed=base_seed,
        tabfm_model=tabfm_model,
        tabpfn_model=tabpfn_model,
        n_jobs=n_jobs,
        fit_granularity=fit_granularity,
        max_num_rows=max_num_rows,
        acquisition=acquisition,
        quantile=quantile,
        n_shadow=n_shadow,
    )


# The IMOSS proposal-oracle family for the oracle-distribution shadow probe,
# plus UCB-AIR kept only for the shared cumulative-regret grid
# (run_single_experiment skips the shadow probe for it -- see has_oracle). Both
# foundation-model oracles are included so a single run reproduces both figures.
_DEFAULT_ALGORITHMS = [
    Algorithm.IMOSS,
    Algorithm.IMOSS_TPE,
    Algorithm.IMOSS_TABPFN,
    Algorithm.UCB_AIR,
]


def make_plots(
    benchmarks,
    n_iterations,
    foundation: str = "tabpfn",
    fit_granularity: str = "arm",
    save_fig: bool = True,
    acquisition: str = "quantile",
    quantile: float = 0.99,
) -> None:
    """Draw the paper's RF figures for one foundation-model oracle: the static
    reward-landscape structure grid (benchmark-only), the combined
    cumulative-regret + oracle-proposal-quality grid, and the surrogate
    suggested-config MSE grid.

    ``foundation`` selects the foundation-model series ("tabfm" or "tabpfn");
    the three surrogate-free baselines are shared by both figures. For "tabpfn"
    with ``fit_granularity="pull"``, ``acquisition="quantile"`` and/or a
    non-default ``quantile``, the variant series and suffixed filenames
    (``..._pull``/``..._q0.975``/``..._ucb_q0.99``) are used, so they sit next
    to (never overwrite) the default per-arm q=0.99 quantile ones.
    """
    # Head-less: make the plotting helpers' trailing ``plt.show()`` a no-op so
    # every PDF is written without a GUI (an interactive backend would block).
    import matplotlib

    matplotlib.use("Agg")
    from experiments.utils.plots.rf_arm_distribution_plot import (
        _load_suggestion_mse_traces,
        _plot_suggestion_metric_grid,
        plot_regret_and_oracle_grid,
    )
    from experiments.utils.plots.rf_landscape_plot import (
        plot_landscape_heatmap_grid,
    )

    is_tabpfn = foundation == "tabpfn"
    is_pull = is_tabpfn and fit_granularity == "pull"
    fm_label = (
        "IMOSS-TabPFN-pull"
        if is_pull
        else "IMOSS-TabPFN" if is_tabpfn else "IMOSS-TabFM"
    )
    suffix = "_pull" if is_pull else ""
    if is_tabpfn and acquisition == "ucb":
        variant = f"ucb_q{quantile:g}"
    elif is_tabpfn and acquisition == "quantile" and quantile != 0.99:
        variant = f"q{quantile:g}"
    else:
        variant = None
    if variant is not None:
        fm_label += f"-{variant}"
        suffix += f"_{variant}"
        # Register the variant's slug -> display label so every loader that
        # globs result files (they map slugs through _PRETTY_LABELS) picks the
        # variant series up under a readable name.
        from experiments.utils.plots.plot_configs import _PRETTY_LABELS

        _PRETTY_LABELS[
            algo_slug(Algorithm.IMOSS_TABPFN, fit_granularity, acquisition, quantile)
        ] = fm_label
    # Hier-MAB's per-run JSONs are produced by factored_baseline_experiment.py
    # (same directory, filename scheme, and seed pairing); it has no proposal
    # oracle, so like UCB-AIR it appears only in the regret row. IMOSS-TPE-univ
    # (multivariate=False) has stored runs but is not part of the paper figure.
    regret_algos = [
        "IMOSS-Random",
        "IMOSS-TPE",
        "IMOSS-mutate-KLxTPE",
        # fm_label,
        "UCB-AIR",
        "Hier-MAB",
    ]
    oracle_algos = [
        "IMOSS-Random",
        "IMOSS-TPE",
        "IMOSS-mutate-KLxTPE",
        # fm_label,
    ]

    print("Generating RF reward-landscape structure grid (benchmark-only)...")
    plot_landscape_heatmap_grid(
        bm_ids=tuple(int(tag.removeprefix("rf")) for tag in benchmarks),
        save_fig=save_fig,
    )

    print(f"Generating combined regret + oracle-proposal-quality grid ({fm_label})...")
    plot_regret_and_oracle_grid(
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        regret_algorithms=regret_algos,
        oracle_algorithms=oracle_algos,
        out_name=f"regret_and_oracle_grid_{foundation}{suffix}",
    )

    print(f"Generating {foundation} suggested-config MSE grid...")
    _plot_suggestion_metric_grid(
        _load_suggestion_mse_traces,
        f"{'TabPFN' if is_tabpfn else 'TabFM'} Suggested-Config MSE",
        f"{foundation}_suggestion_mse_grid{suffix}",
        benchmarks=benchmarks,
        n_iterations=n_iterations,
        save_fig=save_fig,
        log_scale=True,
        # Only this figure's foundation-model series: the loader also picks
        # up the acquisition-sweep variant runs, which must not be drawn.
        algorithms=[fm_label],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--algorithm",
        default="all",
        choices=["all"] + [a.value for a in Algorithm],
        help="method to run (default: 'all' -- every algorithm x benchmark)",
    )
    p.add_argument(
        "--n-runs", type=int, default=10, help="independent seeds per algorithm"
    )
    p.add_argument("--n-iter", type=int, default=5000, help="iterations (T) per run")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=N_JOBS, help="parallel run workers")
    p.add_argument(
        "--benchmarks",
        type=int,
        nargs="+",
        default=[146822, 31, 167120],
        help="OpenML task ids: segment=146822, credit-g=31, numerai28.6=167120",
    )
    p.add_argument(
        "--fit-granularity",
        choices=["arm", "pull"],
        default="arm",
        help=(
            "IMOSS-TabPFN training-table granularity: 'arm' (default, paper) "
            "fits one row per arm at its mean reward; 'pull' fits one row per "
            "individual pull (no averaging) with TabPFN's KV-cache. 'pull' "
            "results go to distinct '..._pull' files/plots."
        ),
    )
    p.add_argument(
        "--max-num-rows",
        type=int,
        default=None,
        help="TabPFN in-context row cap (default: 200 for arm, 10000 for pull).",
    )
    p.add_argument(
        "--acquisition",
        choices=["ucb", "quantile"],
        default="quantile",
        help=(
            "IMOSS-TabPFN acquisition: 'quantile' (default) ranks candidates "
            "on the --quantile level of TabPFN's predictive distribution "
            "(the GIT-BO quantile form of UCB); 'ucb' ranks on "
            "mean + kappa*std at kappa = Phi^-1(--quantile). 'ucb' results "
            "go to distinct '..._ucb_q<q>' files/plots."
        ),
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help=(
            "IMOSS-TabPFN exploration level in (0,1) (default 0.99, the "
            "best/most seed-stable setting in the acquisition comparison): "
            "the level ranked on by the quantile acquisition, or converted "
            "to the UCB weight via kappa = Phi^-1(quantile) for 'ucb'. "
            "Non-default values go to distinct '..._q<q>' files/plots."
        ),
    )
    p.add_argument(
        "--foundation",
        choices=["tabfm", "tabpfn"],
        default=None,
        help=(
            "which foundation-model figure to (re)plot; default: inferred from "
            "--algorithm, or both when --algorithm all."
        ),
    )
    p.add_argument(
        "--plot", action="store_true", help="after running, draw the paper figure(s)"
    )
    p.add_argument(
        "--plot-only", action="store_true", help="skip running, only (re)plot"
    )
    p.add_argument(
        "--n-shadow",
        type=int,
        default=N_SHADOW,
        help=(
            "oracle-quality probe draws per probe point; 0 disables the probe. "
            "It runs on a DEEP COPY of the optimizer, so `regrets` is identical "
            "either way -- but for a surrogate oracle it dominates the run: at "
            "5000 iterations it makes ~50 TabPFN fits against the search's ~10. "
            "Pass 0 when only the regret is needed (~7.5x faster)."
        ),
    )
    p.add_argument("--no-plot", action="store_true", help="run but skip plotting")
    p.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke test: T=60, 2 runs, credit-g only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.quick:
        n_runs, n_iter, bm_ids = 2, 60, [31]
    else:
        n_runs, n_iter, bm_ids = args.n_runs, args.n_iter, args.benchmarks

    bench_tags = tuple(f"rf{bm_id}" for bm_id in bm_ids)
    algorithms = (
        list(_DEFAULT_ALGORITHMS)
        if args.algorithm == "all"
        else [Algorithm(args.algorithm)]
    )

    if not args.plot_only:
        # Warm up each needed surrogate model once, reused across all benchmarks.
        tabfm_model = load_tabfm() if Algorithm.IMOSS_TABFM in algorithms else None
        if tabfm_model is not None:
            print("Loaded TabFM model (once, reused across all runs/budgets).")
        tabpfn_model = None
        if any(a in algorithms for a in _TABPFN_ALGORITHMS):
            tabpfn_model = load_tabpfn()
            _silence_known_warnings()
            print("TabPFN-3 ready (checkpoint cached; reused across all runs/budgets).")

        total_tasks = len(bm_ids) * len(algorithms)
        start = time.time()
        with tqdm(total=total_tasks, desc="benchmark x algorithm", unit="task") as bar:
            for bm_id in bm_ids:
                bench = RFTabularFiniteBenchmark(bm_id=bm_id)
                print(
                    f"\nRF tabular finite benchmark (OpenML task {bm_id}): "
                    f"{bench.n_arms} arms, best val_acc={bench.max_value:.4f}"
                )
                for algorithm in algorithms:
                    run_experiment(
                        bench,
                        n_runs,
                        args.base_seed,
                        n_iter,
                        algorithm,
                        tabfm_model=tabfm_model,
                        tabpfn_model=tabpfn_model,
                        n_jobs=args.n_jobs,
                        fit_granularity=args.fit_granularity,
                        max_num_rows=args.max_num_rows,
                        acquisition=args.acquisition,
                        quantile=args.quantile,
                        n_shadow=args.n_shadow,
                    )
                    bar.update(1)
                    done, total = bar.n, bar.total
                    elapsed = time.time() - start
                    eta = elapsed / done * (total - done) if done else 0.0
                    bar.set_postfix_str(f"elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m")

    # Plot when asked (--plot/--plot-only) or by default after an "all" run.
    want_plot = not args.no_plot and (
        args.plot or args.plot_only or args.algorithm == "all"
    )
    if want_plot:
        if args.foundation is not None:
            foundations = [args.foundation]
        elif args.algorithm == "all":
            foundations = ["tabfm", "tabpfn"]
        elif args.algorithm == Algorithm.IMOSS_TABFM.value:
            foundations = ["tabfm"]
        elif args.algorithm == Algorithm.IMOSS_TABPFN.value:
            foundations = ["tabpfn"]
        else:
            foundations = []  # a surrogate-free run defines no figure on its own
        for foundation in foundations:
            make_plots(
                bench_tags,
                n_iter,
                foundation=foundation,
                fit_granularity=args.fit_granularity,
                save_fig=True,
                acquisition=args.acquisition,
                quantile=args.quantile,
            )
