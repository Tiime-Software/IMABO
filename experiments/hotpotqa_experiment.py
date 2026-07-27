"""HotpotQA online-HPO experiment (the paper's HotpotQA figure).

One file runs every method, including both foundation-model oracles (Google
TabFM via ``IMOSS-TABFM`` and Prior-Labs TabPFN-3 via ``IMOSS-TABPFN``). All
runs are checkpointed and resumable, so reruns skip finished seeds.

Reproduce the figure (each command is resumable; the three baselines are
model-agnostic and shared by the TabFM and TabPFN figures):

    python -m experiments.hotpotqa_experiment --algorithm IMABO       # IMOSS-TPE
    python -m experiments.hotpotqa_experiment --algorithm UCB-AIR
    python -m experiments.hotpotqa_experiment --algorithm Random
    python -m experiments.hotpotqa_experiment --algorithm IMOSS-TABPFN  # or IMOSS-TABFM

    # then draw the figure (uses --algorithm as the foundation-model series):
    python -m experiments.hotpotqa_experiment --algorithm IMOSS-TABPFN --plot-only

Requires an ``OPENROUTER_API_KEY`` (in a ``.env`` file or the environment) for
the LLM calls, and the TabPFN extra for the ``IMOSS-TABPFN`` arm
(``pip install -e ".[experiments]"``).
"""

from pathlib import Path
import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv
from enum import Enum
from typing import Any
from tqdm import tqdm
from imabo import IMABO, IMABOTabFM, IMABOTabPFN
from imabo.optimizer import load_tabfm
from imabo.tabpfn_optimizer import load_tabpfn
from experiments.baselines.optuna_bandit import OptunaBandit
from experiments.baselines.random_search import RandomSearch
from experiments.baselines.ucb_air import UCBAIR
from experiments.benchmarks.hotpotqa.benchmark import HotpotQABenchmark
from experiments.benchmarks.hotpotqa.types import Result
from openrouter import OpenRouter
from openrouter.errors import (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    ResponseValidationError,
)
from joblib.memory import Memory
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

memory = Memory(location=Path(__file__).parents[1] / "data" / ".cache", verbose=0)

load_dotenv()

DATA_NAME = "hotpotqa"
DATA_FOLDER = Path(__file__).parent / "data" / DATA_NAME
RESULT_FOLDER = Path(__file__).parent.parent / "results" / DATA_NAME
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
BETA = 0.5


class Algorithm(Enum):
    IMABO = "IMABO"  # full method: TPE explore + MOSS exploit
    RANDOM = "Random"  # true uniform random search (RandomSearch)
    NO_TPE = "IMABO-noTPE"  # ablation: MOSS-only (IMABO with use_tpe=False)
    OPTUNA = "Optuna"  # sequential TPE with k-observation averaging
    UCB_AIR = "UCB-AIR"  # infinitely-many-armed bandit, arm-increasing rule + UCBV
    IMOSS_TABFM = "IMOSS-TABFM"  # IMOSS-TABFM
    IMOSS_TABPFN = "IMOSS-TABPFN"  # IMOSS with a TabPFN-3 explore oracle


def build_optimizer(
    algorithm: Algorithm,
    seed: int,
    optuna_k: int = 1,
    beta: float = 0.8,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
):
    """Construct the optimizer for ``algorithm``.

    All returned optimizers expose ``suggest()`` / ``observe(reward)``; use
    :func:`best_config_of` to read the incumbent uniformly across them.
    ``beta`` (the explore/exploit switching exponent) only applies to
    IMABO/IMABO-noTPE/IMOSS-TABFM/IMOSS-TABPFN. ``tabfm_model`` should be a
    single pre-loaded model shared across all seeds (see :func:`load_tabfm`) --
    IMOSS-TABFM only builds its own copy via ``load_tabfm()`` as a fallback
    for standalone/one-off calls. ``tabpfn_model`` is the analogous shared
    handle for IMOSS-TABPFN (see :func:`load_tabpfn`).
    """
    if algorithm == Algorithm.IMABO:
        return IMABO(search_space=SEARCH_SPACE, seed=seed, use_tpe=True, beta=beta)
    elif algorithm == Algorithm.NO_TPE:
        return IMABO(search_space=SEARCH_SPACE, seed=seed, use_tpe=False, beta=beta)
    elif algorithm == Algorithm.IMOSS_TABFM:
        model = tabfm_model if tabfm_model is not None else load_tabfm()
        return IMABOTabFM(
            search_space=SEARCH_SPACE,
            seed=seed,
            beta=beta,
            tabfm_model=model,
            suggest_method="max",
        )
    elif algorithm == Algorithm.IMOSS_TABPFN:
        model = tabpfn_model if tabpfn_model is not None else load_tabpfn()
        return IMABOTabPFN(
            search_space=SEARCH_SPACE,
            seed=seed,
            beta=beta,
            tabpfn_model=model,
            acquisition=acquisition,
            quantile=quantile,
        )
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=SEARCH_SPACE, seed=seed)
    elif algorithm == Algorithm.OPTUNA:
        return OptunaBandit(search_space=SEARCH_SPACE, k=optuna_k, seed=seed)
    elif algorithm == Algorithm.UCB_AIR:
        return UCBAIR(search_space=SEARCH_SPACE, beta=beta, seed=seed)


def best_config_of(optimizer) -> dict | None:
    """Read the incumbent config across optimizer interfaces.

    IMABO / RandomSearch expose a ``best_config`` property; OptunaBandit
    exposes ``suggest_best()`` instead (mirrors hpo_experiment's tree-vs-IMABO
    branch).
    """
    if hasattr(optimizer, "best_config"):
        return optimizer.best_config
    return optimizer.suggest_best()


def require_best_config_of(optimizer) -> dict:
    config = best_config_of(optimizer)
    if config is None:
        raise RuntimeError("No best config available from optimizer")
    return config


def algo_label(algorithm: Algorithm, optuna_k: int = 1, beta: float = 0.8) -> str:
    """File/stats label; distinguishes Optuna-k and non-default-beta runs."""
    label = algorithm.value
    if algorithm == Algorithm.OPTUNA and optuna_k != 1:
        label += f"-k{optuna_k}"
    if algorithm in (
        Algorithm.IMABO,
        Algorithm.NO_TPE,
        Algorithm.IMOSS_TABFM,
        Algorithm.IMOSS_TABPFN,
        Algorithm.UCB_AIR,
    ):
        label += f"-beta{beta}"
    return label


client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))


SEARCH_SPACE = {
    "top_k": {"lower": 1, "upper": 10, "int": True},
    "temperature": {"lower": 0.0, "upper": 1.0, "log": False},
    "prompt_template": {"choices": ["few_shot", "zero_shot", "naive"]},
    "model": {
        "choices": [
            "qwen/qwen3.5-flash-02-23",
            "inclusionai/ling-2.6-flash",
            "deepseek/deepseek-v4-flash",
            "ibm-granite/granite-4.1-8b",
            "tencent/hy3-preview",
            "meta-llama/llama-3.2-1b-instruct",
        ]
    },
}


FIXED_PARAMS = {"retrieval": "dense"}

PROMPT_TEMPLATES = {
    "few_shot": """
You are a precise question-answering assistant. Answer using only the provided context:

Rules:
- Answer in as few words as possible.
- Never explain or add context - just the answer.
- For yes/no questions answer with only "yes" or "no".
- If the context does not contain the answer, output "unknown".

Examples:
Q: What is the capital of France? | Context: The capital of France is Paris.
A: Paris

Q: What year was the Eiffel Tower built? | Context: The Eiffel Tower was built in 1889.
A: 1889

Q: Is Berlin the capital of Germany? | Context: The capital of Germany is Berlin.
A: yes
""",
    "zero_shot": """
You are a precise question-answering assistant. Answer using only the provided context:

Rules:
- Answer in as few words as possible.
- Never explain or add context - just the answer.
- For yes/no questions answer with only "yes" or "no".
- If the context does not contain the answer, output "unknown".
""",
    "naive": """
Answer the question using the provided context.
""",
}


@retry(
    retry=retry_if_exception_type(
        (
            TooManyRequestsResponseError,
            ProviderOverloadedResponseError,
            ServiceUnavailableResponseError,
            ResponseValidationError,
        )
    ),
    wait=wait_random_exponential(multiplier=2, max=300),
    stop=stop_after_attempt(30),
    reraise=True,
)
@memory.cache(ignore=["passages"])
def call_llm(question: str, config: dict, passages: list[str]) -> str:
    time.sleep(random.uniform(0.3, 0.8))

    usr_content = f"Context: {'\n'.join(passages)}\n\nQuestion: {question}"

    system_prompt = PROMPT_TEMPLATES[config["prompt_template"]]

    response = client.chat.send(
        model=config["model"],
        temperature=config["temperature"],
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": usr_content},
        ],
    )
    return response.choices[0].message.content


def _run_algorithm(
    benchmark: HotpotQABenchmark,
    train_qids: list[str],
    holdout_qids: list[str],
    algorithm: Algorithm,
    seed: int,
    checkpoint_path: Path,
    max_workers: int = 4,
    optuna_k: int = 1,
    beta: float = 0.8,
    position: int | None = None,
    tabfm_model: Any = None,
    tabpfn_model: Any = None,
    acquisition: str = "quantile",
    quantile: float = 0.99,
) -> dict:
    optimizer = build_optimizer(
        algorithm,
        seed,
        optuna_k=optuna_k,
        beta=beta,
        tabfm_model=tabfm_model,
        tabpfn_model=tabpfn_model,
        acquisition=acquisition,
        quantile=quantile,
    )
    done = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            lines = [ln for ln in f if ln.strip()]
        for i, ln in enumerate(lines):
            try:
                done.append(json.loads(ln))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                tqdm.write(
                    f"--- Dropped truncated final checkpoint line in {checkpoint_path.name} ---"
                )

    raw_results = []
    configs = []
    best_configs = []
    regrets = []
    ckpt = open(checkpoint_path, "a")
    try:
        for step, qid in enumerate(
            tqdm(
                train_qids,
                desc=f"{algorithm.value} seed{seed}",
                leave=False,
                position=position,
            )
        ):
            config = optimizer.suggest()
            if step < len(done):
                rec = done[step]
                if rec["qid"] != qid:
                    raise ValueError(
                        f"Checkpoint mismatch at step {step}: expected qid {qid}, "
                        f"found {rec['qid']}. Stale checkpoint for seed {seed} "
                        f"(different sampling/search space?)."
                    )
                optimizer.observe(rec["reward"])
                regrets.append(rec["regret"])
                configs.append(rec["config"])
                best_configs.append(rec["best_config"])
                raw_results.append(rec["raw_result"])
                continue
            llm_config = {
                **FIXED_PARAMS,
                **config,
                "temperature": round(config["temperature"], 2),
            }
            result: Result = benchmark.eval_question(qid, llm_config, call_llm)
            reward = result.reward.weighted_f1
            optimizer.observe(reward)
            rec = {
                "qid": qid,
                "config": config,
                "reward": reward,
                "regret": 1.0 - reward,
                "best_config": best_config_of(optimizer),
                "raw_result": asdict(result),
            }
            regrets.append(rec["regret"])
            configs.append(config)
            best_configs.append(rec["best_config"])
            raw_results.append(rec["raw_result"])
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
    finally:
        ckpt.close()
    best_config = {**FIXED_PARAMS, **require_best_config_of(optimizer)}
    best_config["temperature"] = round(best_config["temperature"], 2)

    # holdout is independent — evaluate in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(benchmark.eval_question, qid, best_config, call_llm): idx
            for idx, qid in enumerate(holdout_qids)
        }
        holdout_rewards_map = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Holdout seed{seed}",
            leave=False,
            position=position,
        ):
            holdout_rewards_map[futures[future]] = future.result().reward.weighted_f1
    holdout_rewards = [holdout_rewards_map[i] for i in range(len(holdout_qids))]

    simple_regret = 1.0 - (sum(holdout_rewards) / len(holdout_rewards))
    return {
        "regrets": regrets,
        "simple_regrets": simple_regret,
        "best_config": best_config,
        "configs": configs,
        "best_configs": best_configs,
        "holdout_rewards": holdout_rewards,
        "raw_results": raw_results,
    }


def save_hotpotqa_to_csv(
    all_results: list[dict],
    label: str,
    n_samples: int,
    n_runs: int,
) -> None:
    from experiments.utils.stats import calculate_statistics

    stats = calculate_statistics([{label: r} for r in all_results])
    algo_stats = stats[label]
    stem = f"{label}_hotpotqa_{n_samples}samples_{n_runs}runs"

    # iterations CSV: one row per iteration, mean/std across runs
    mean_regrets = algo_stats["regrets"]["mean"]
    std_regrets = algo_stats["regrets"]["std"]
    with open(RESULT_FOLDER / f"{stem}_iterations.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_iterations",
                "algorithm",
                "iteration",
                "regret_mean",
                "regret_std",
            ],
        )
        writer.writeheader()
        for i, (mean, std) in enumerate(zip(mean_regrets, std_regrets)):
            writer.writerow(
                {
                    "n_iterations": n_samples,
                    "algorithm": label,
                    "iteration": i + 1,
                    "regret_mean": mean,
                    "regret_std": std,
                }
            )

    # summary CSV: one row with aggregated simple/total regret stats
    simple_regrets_arr = algo_stats["simple_regrets"]["simple_regrets"]
    sum_regrets_arr = algo_stats["regrets"]["sum_regrets"]
    with open(RESULT_FOLDER / f"{stem}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_iterations",
                "algorithm",
                "simple_regret_mean",
                "simple_regret_std",
                "total_regret_mean",
                "total_regret_std",
                "simple_regrets",
                "sum_regrets",
                "cov_regrets",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n_iterations": n_samples,
                "algorithm": label,
                "simple_regret_mean": algo_stats["simple_regrets"]["mean"],
                "simple_regret_std": algo_stats["simple_regrets"]["std"],
                "total_regret_mean": float(sum_regrets_arr.mean()),
                "total_regret_std": float(sum_regrets_arr.std()),
                "simple_regrets": str(simple_regrets_arr),
                "sum_regrets": str(sum_regrets_arr),
                "cov_regrets": str(algo_stats["cov"]),
            }
        )


def run_multiple_experiments(
    n_samples: int = 100,
    n_runs: int = 5,
    base_seed: int = 42,
    n_holdout: int = 100,
    algorithm: Algorithm = Algorithm.IMABO,
    optuna_k: int = 1,
    beta: float = 0.8,
    max_parallel_runs: int = 5,
    acquisition: str = "quantile",
    quantile: float = 0.99,
) -> list[dict]:
    # Built once; the Dense index spans the full corpus, so each run only
    # re-samples its (train, holdout) split per seed (no embedding rebuild).
    benchmark = HotpotQABenchmark(
        data_folder=DATA_FOLDER,
        n_samples=n_samples,
        n_holdout=n_holdout,
        seed=base_seed,
    )

    label = algo_label(algorithm, optuna_k, beta)
    seeds = [base_seed + i * 1000 for i in range(n_runs)]

    tabfm_model = load_tabfm() if algorithm == Algorithm.IMOSS_TABFM else None
    tabpfn_model = load_tabpfn() if algorithm == Algorithm.IMOSS_TABPFN else None

    splits: dict[int, tuple[list[str], list[str]]] = {}
    for seed in seeds:
        benchmark.resample(seed)
        splits[seed] = (list(benchmark.train_qids), list(benchmark.holdout_qids))

    def run_one(i: int) -> dict:
        seed = seeds[i]
        stem = f"{label}_hotpotqa_{n_samples}samples"
        run_path = RESULT_FOLDER / f"{stem}_run{i}.json"
        # A finished run (training + holdout) is saved as run{i}.json — skip it.
        if run_path.exists():
            with open(run_path) as f:
                run_result = json.load(f)
            tqdm.write(f"--- Run {i} (seed {seed}) already complete, skipping ---")
            return run_result

        train_qids, holdout_qids = splits[seed]
        checkpoint_path = (
            RESULT_FOLDER / f"checkpoint_{label}_hotpotqa_seed{seed}.jsonl"
        )
        run_result = _run_algorithm(
            benchmark,
            train_qids,
            holdout_qids,
            algorithm,
            seed,
            checkpoint_path,
            optuna_k=optuna_k,
            beta=beta,
            position=i,
            tabfm_model=tabfm_model,
            tabpfn_model=tabpfn_model,
            acquisition=acquisition,
            quantile=quantile,
        )
        with open(run_path, "w") as f:
            json.dump(run_result, f, ensure_ascii=False, indent=4)
        return run_result

    with ThreadPoolExecutor(max_workers=min(max_parallel_runs, n_runs)) as executor:
        all_results = list(
            tqdm(
                executor.map(run_one, range(n_runs)),
                total=n_runs,
                desc="Runs",
            )
        )

    with open(
        RESULT_FOLDER / f"{label}_hotpotqa_multi_{n_samples}samples_{n_runs}runs.json",
        "w",
    ) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    save_hotpotqa_to_csv(all_results, label, n_samples, n_runs)

    return all_results


def make_plot(foundation: Algorithm, n_samples: int, n_runs: int) -> None:
    """Draw the paper's HotpotQA figure: the IMOSS family (TPE + a foundation
    model) vs UCB-AIR and Random.

    ``foundation`` is the foundation-model series to put in the comparison --
    ``Algorithm.IMOSS_TABFM`` for the TabFM figure or ``Algorithm.IMOSS_TABPFN``
    for the TabPFN one. The other three series are model-agnostic, so a single
    set of baseline checkpoints is reused for either figure. All four series
    must already be computed (see ``--algorithm`` runs below).
    """
    # Head-less: make the plotting helper's trailing ``plt.show()`` a no-op so
    # the PDF is written without a GUI (an interactive backend would block).
    import matplotlib

    matplotlib.use("Agg")
    from experiments.utils.plots.hotpotqa_plot import (
        ALGO_DISPLAY_NAMES,
        plot_hotpotqa_results,
    )

    tpe = algo_label(Algorithm.IMABO, beta=BETA)
    ucb = algo_label(Algorithm.UCB_AIR, beta=BETA)
    rnd = algo_label(Algorithm.RANDOM, beta=BETA)
    fm = algo_label(foundation, beta=BETA)
    display_overrides = {
        tpe: ALGO_DISPLAY_NAMES.get(Algorithm.IMABO.value, "IMOSS-TPE"),
        ucb: ALGO_DISPLAY_NAMES.get(Algorithm.UCB_AIR.value, "UCB-AIR"),
        fm: ALGO_DISPLAY_NAMES.get(foundation.value, foundation.value),
    }
    fig_slug = foundation.value.lower().replace("-", "_")
    print(f"Drawing HotpotQA paper figure ({display_overrides[fm]})...")
    plot_hotpotqa_results(
        algorithms=[tpe, fm, ucb, rnd],
        n_samples=n_samples,
        n_runs=n_runs,
        save_fig=True,
        display_overrides=display_overrides,
        fig_name=f"hotpotqa_imabo_family_{fig_slug}_{n_samples}samples.pdf",
        columns=1,
        conference="aaai",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--algorithm",
        default=Algorithm.IMOSS_TABFM.value,
        choices=[a.value for a in Algorithm],
        help=(
            "Which method to run. The paper's HotpotQA figure compares the "
            "IMOSS family with a foundation-model oracle -- run this once per "
            "method: 'IMABO' (IMOSS-TPE), 'UCB-AIR', 'Random', and one of "
            "'IMOSS-TABFM' / 'IMOSS-TABPFN'."
        ),
    )
    p.add_argument("--n-samples", type=int, default=5000, help="online HPO steps (T)")
    p.add_argument("--n-runs", type=int, default=5, help="independent seeds")
    p.add_argument("--n-holdout", type=int, default=500, help="held-out eval questions")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--max-parallel-runs",
        type=int,
        default=None,
        help="seeds evaluated concurrently (default: --n-runs)",
    )
    p.add_argument(
        "--acquisition",
        choices=["ucb", "quantile"],
        default="quantile",
        help=(
            "IMOSS-TABPFN acquisition (see imabo.tabpfn_optimizer.IMABOTabPFN): "
            "'quantile' (default) ranks candidates on the --quantile level of "
            "TabPFN's predictive distribution, 'ucb' on mean + kappa*std at "
            "kappa = Phi^-1(--quantile)."
        ),
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help=(
            "IMOSS-TABPFN exploration level in (0,1): the quantile of "
            "TabPFN's predictive distribution ranked on (default 0.99); for "
            "--acquisition ucb it is converted to kappa = Phi^-1(quantile)."
        ),
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help=(
            "after computing, draw the paper figure (uses --algorithm as the "
            "foundation-model series; the other three series must already be "
            "computed on disk)"
        ),
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="skip computing, only (re)draw the figure",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    algorithm = Algorithm(args.algorithm)

    if not args.plot_only:
        run_multiple_experiments(
            n_samples=args.n_samples,
            n_runs=args.n_runs,
            n_holdout=args.n_holdout,
            base_seed=args.base_seed,
            algorithm=algorithm,
            beta=BETA,
            max_parallel_runs=args.max_parallel_runs or args.n_runs,
            acquisition=args.acquisition,
            quantile=args.quantile,
        )

    if args.plot or args.plot_only:
        make_plot(algorithm, args.n_samples, args.n_runs)
