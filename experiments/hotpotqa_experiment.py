from pathlib import Path
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv
from enum import Enum
from tqdm import tqdm
from imabo import IMABO
from experiments.baselines.optuna_bandit import OptunaBandit
from experiments.baselines.random_search import RandomSearch
from experiments.benchmarks.hotpotqa.benchmark import HotpotQABenchmark
from experiments.benchmarks.hotpotqa.types import Result, BatchResult
from openrouter import OpenRouter
from openrouter.errors import (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
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


class Algorithm(Enum):
    IMABO = "IMABO"  # full method: TPE explore + MOSS exploit
    RANDOM = "Random"  # true uniform random search (RandomSearch)
    NO_TPE = "IMABO-noTPE"  # ablation: MOSS-only (IMABO with use_tpe=False)
    OPTUNA = "Optuna"  # sequential TPE with k-observation averaging


def build_optimizer(algorithm: Algorithm, seed: int, optuna_k: int = 1):
    """Construct the optimizer for ``algorithm``.

    All returned optimizers expose ``suggest()`` / ``observe(reward)``; use
    :func:`best_config_of` to read the incumbent uniformly across them.
    """
    if algorithm == Algorithm.IMABO:
        return IMABO(search_space=SEARCH_SPACE, seed=seed, use_tpe=True)
    elif algorithm == Algorithm.NO_TPE:
        return IMABO(search_space=SEARCH_SPACE, seed=seed, use_tpe=False)
    elif algorithm == Algorithm.RANDOM:
        return RandomSearch(search_space=SEARCH_SPACE, seed=seed)
    elif algorithm == Algorithm.OPTUNA:
        return OptunaBandit(search_space=SEARCH_SPACE, k=optuna_k, seed=seed)


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


def algo_label(algorithm: Algorithm, optuna_k: int = 1) -> str:
    """File/stats label; distinguishes Optuna runs with different k."""
    if algorithm == Algorithm.OPTUNA and optuna_k != 1:
        return f"{algorithm.value}-k{optuna_k}"
    return algorithm.value


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
        )
    ),
    wait=wait_random_exponential(multiplier=2, max=300),
    stop=stop_after_attempt(20),
    reraise=True,
)
@memory.cache(ignore=["passages"])
def call_llm(question: str, config: dict, passages: list[str]) -> str:

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


def run_single_experiment(
    n_samples: int = 10,
    algorithm: Algorithm = Algorithm.IMABO,
    seed: int = 42,
    n_holdout: int = 5,
    batch_size: int = 5,
    optuna_k: int = 1,
) -> dict:
    optimizer = build_optimizer(algorithm, seed, optuna_k=optuna_k)
    benchmark = HotpotQABenchmark(
        data_folder=DATA_FOLDER,
        n_samples=n_samples,
        n_holdout=n_holdout,
        seed=seed,
    )
    eval_qids = benchmark.train_qids
    holdout_qids = benchmark.holdout_qids

    n_steps = len(eval_qids) // batch_size
    ite_results = []
    for step in tqdm(range(n_steps), desc="Running experiment"):
        batch_qids = eval_qids[step * batch_size : (step + 1) * batch_size]
        config = optimizer.suggest()
        llm_config = {
            **FIXED_PARAMS,
            **config,
            "temperature": round(config["temperature"], 2),
        }
        result: BatchResult = benchmark.eval_batch(batch_qids, llm_config, call_llm)
        reward = result.avg_reward
        optimizer.observe(reward)
        ite_results.append(
            {
                "qids": batch_qids,
                "config": config,
                "reward": reward,
                "regret": 1.0 - reward,
                "best_config": best_config_of(optimizer),
                "raw_results": [asdict(r) for r in result.results],
            }
        )

    best_config = {**FIXED_PARAMS, **require_best_config_of(optimizer)}
    holdout_rewards = []
    for qid in tqdm(holdout_qids, desc="Evaluating best config"):
        result: Result = benchmark.eval_question(qid, best_config, call_llm)
        holdout_rewards.append(result.reward.weighted_f1)

    simple_regret = 1.0 - (sum(holdout_rewards) / len(holdout_rewards))

    with open(RESULT_FOLDER / f"hotpotqa_ite_results_{n_samples}.json", "w") as f:
        json.dump(ite_results, f, ensure_ascii=False, indent=4)

    with open(RESULT_FOLDER / f"hotpotqa_regret_{n_samples}.json", "w") as f:
        json.dump(
            {
                "best_config": best_config,
                "simple_regret": simple_regret,
                "regrets": [ir["regret"] for ir in ite_results],
                "holdout_rewards": holdout_rewards,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    return {
        "regret": [ir["regret"] for ir in ite_results],
        "simple_regret": simple_regret,
        "best_config": best_config,
        "holdout_rewards": holdout_rewards,
    }


def _run_algorithm(
    benchmark: HotpotQABenchmark,
    train_qids: list[str],
    holdout_qids: list[str],
    algorithm: Algorithm,
    seed: int,
    checkpoint_path: Path,
    batch_size: int = 1,
    max_workers: int = 4,
    optuna_k: int = 1,
) -> dict:
    optimizer = build_optimizer(algorithm, seed, optuna_k=optuna_k)
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

    n_steps = len(train_qids) // batch_size
    batches = [
        train_qids[s * batch_size : (s + 1) * batch_size] for s in range(n_steps)
    ]

    raw_results = []
    configs = []
    best_configs = []
    regrets = []
    ckpt = open(checkpoint_path, "a")
    try:
        for step, batch_qids in enumerate(
            tqdm(batches, desc=f"{algorithm.value} seed{seed}", leave=False)
        ):
            config = optimizer.suggest()
            if step < len(done):
                rec = done[step]
                if rec["qids"] != batch_qids:
                    raise ValueError(
                        f"Checkpoint mismatch at step {step}: expected qids {batch_qids}, "
                        f"found {rec['qids']}. Stale checkpoint for seed {seed} "
                        f"(different sampling/batch_size/search space?)."
                    )
                optimizer.observe(rec["reward"])
                regrets.append(rec["regret"])
                configs.append(rec["config"])
                best_configs.append(rec["best_config"])
                raw_results.extend(rec["raw_results"])
                continue
            llm_config = {
                **FIXED_PARAMS,
                **config,
                "temperature": round(config["temperature"], 2),
            }
            batch: BatchResult = benchmark.eval_batch(batch_qids, llm_config, call_llm)
            reward = batch.avg_reward
            optimizer.observe(reward)
            rec = {
                "qids": batch_qids,
                "config": config,
                "reward": reward,
                "regret": 1.0 - reward,
                "best_config": best_config_of(optimizer),
                "raw_results": [asdict(r) for r in batch.results],
            }
            regrets.append(rec["regret"])
            configs.append(config)
            best_configs.append(rec["best_config"])
            raw_results.extend(rec["raw_results"])
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
            as_completed(futures), total=len(futures), desc="Holdout", leave=False
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
    batch_size: int = 1,
    algorithm: Algorithm = Algorithm.IMABO,
    optuna_k: int = 1,
) -> list[dict]:
    # Built once; the Dense index spans the full corpus, so each run only
    # re-samples its (train, holdout) split per seed (no embedding rebuild).
    benchmark = HotpotQABenchmark(
        data_folder=DATA_FOLDER,
        n_samples=n_samples,
        n_holdout=n_holdout,
        seed=base_seed,
    )

    label = algo_label(algorithm, optuna_k)

    all_results = []
    for i in tqdm(range(n_runs), desc="Runs"):
        seed = base_seed + i * 1000
        stem = f"{label}_hotpotqa_{n_samples}samples"
        run_path = RESULT_FOLDER / f"{stem}_run{i}.json"
        # A finished run (training + holdout) is saved as run{i}.json — skip it.
        if run_path.exists():
            with open(run_path) as f:
                all_results.append(json.load(f))
            tqdm.write(f"--- Run {i} (seed {seed}) already complete, skipping ---")
            continue
        # Each run is an independent problem instance: draw a fresh train/holdout
        # split for this seed. Within a seed the split is still a stable prefix,
        # so the per-seed checkpoint stays valid across n_samples.
        benchmark.resample(seed)
        checkpoint_path = (
            RESULT_FOLDER
            / f"checkpoint_{label}_hotpotqa_seed{seed}_batch{batch_size}.jsonl"
        )
        run_result = _run_algorithm(
            benchmark,
            benchmark.train_qids,
            benchmark.holdout_qids,
            algorithm,
            seed,
            checkpoint_path,
            batch_size=batch_size,
            optuna_k=optuna_k,
        )
        all_results.append(run_result)
        with open(run_path, "w") as f:
            json.dump(run_result, f, ensure_ascii=False, indent=4)

    with open(
        RESULT_FOLDER / f"{label}_hotpotqa_multi_{n_samples}samples_{n_runs}runs.json",
        "w",
    ) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    save_hotpotqa_to_csv(all_results, label, n_samples, n_runs)

    return all_results


if __name__ == "__main__":
    n_samples = 2000
    n_runs = 5
    n_holdout = 200
    algorithm = Algorithm.NO_TPE

    run_multiple_experiments(
        n_samples=n_samples,
        n_runs=n_runs,
        n_holdout=n_holdout,
        algorithm=algorithm,
    )
