"""
IMABO hyperparameter optimization over tau2-bench.

Treats tau2's mean task-success rate as the reward signal and uses IMABO
to find the best agent LLM parameters.

Each IMABO step:
  - picks a different task from a seeded shuffle of the domain task list
  - runs num_trials trials on that task
  - computes mean reward across trials → observe to optimizer
  - records per-step regret = 1 - mean_reward

Simple regret is measured after optimization by re-evaluating the best config
on a held-out set of tasks (not seen during optimization).

Multiple independent runs (n_runs) are aggregated via calculate_statistics,
matching the pattern used in hpo_experiment.py.

Usage (from IMABO repo root):
    python -m experiments.tau2_experiment
"""

from dataclasses import dataclass
from enum import Enum
import json
import random
import sys
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import litellm

litellm.suppress_debug_info = True

load_dotenv()

from loguru import logger

logger.disable("tau2.utils.llm_utils")

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_tasks
from tau2.runner.helpers import get_tasks
from imabo import IMABO
from experiments.utils.stats import (
    calculate_statistics,
    save_results_to_csv,
    save_iterations_to_csv,
)

RESULT_DIR = Path(__file__).parent.parent / "results" / "tau2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BANKING_KNOWLEDGE_SEARCH_SPACE: dict = {
    "temperature": {"lower": 0.0, "upper": 1.0},
    "retrieval_config": {"choices": ["qwen_embeddings", "bm25"]},
    "top_k": {"lower": 5, "upper": 20, "int": True},
    "model_name": {
        "choices": [
            "qwen/qwen3.6-flash",
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash",
        ]
    },
}

RETAIL_SEARCH_SPACE: dict = {
    "temperature": {"lower": 0.0, "upper": 1.0},
    "model_name": {
        "choices": [
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash",
            "qwen/qwen3.6-flash",
        ]
    },
}

SEARCH_SPACES: dict = {
    "banking_knowledge": BANKING_KNOWLEDGE_SEARCH_SPACE,
    "retail": RETAIL_SEARCH_SPACE,
}


class Algorithm(Enum):
    IMABO = "IMABO"
    RANDOM = "RANDOM"


def mean_reward(results) -> float:
    rewards = [
        sim.reward_info.reward
        for sim in results.simulations
        if sim.reward_info is not None
    ]
    if not rewards:
        print("WARNING: no rewards found in results, returning 0.0", file=sys.stderr)
        return 0.0
    return sum(rewards) / len(rewards)


def evaluate(
    model_name: str,
    llm_args: dict,
    domain: str,
    llm_user: str,
    num_trials: int,
    task_ids: list[str],
    seed: int,
    max_steps: int,
    max_concurrency: int,
    save_to: str,
    retrieval_config: str | None = None,
    top_k: int | None = None,
) -> float:
    """Run tau2 on the given task_ids and return mean reward."""
    retrieval_config_kwargs = {"top_k": top_k} if top_k is not None else None
    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent="openrouter/" + model_name,
        llm_args_agent=llm_args,
        user="user_simulator",
        llm_user=llm_user,
        llm_args_user={"temperature": 0.0},
        num_trials=num_trials,
        task_ids=task_ids,
        seed=seed,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
        save_to=save_to,
        retrieval_config=retrieval_config,
        retrieval_config_kwargs=retrieval_config_kwargs,
        auto_resume=True,
    )
    tasks = get_tasks(
        task_set_name=config.task_set_name or config.domain,
        task_split_name=config.task_split_name,
        task_ids=task_ids,
    )
    results = run_tasks(
        config, tasks, save_path=Path(save_to) / "results.json", console_display=False
    )
    return mean_reward(results)


def run_single_experiment(
    domain: str,
    llm_user: str,
    n_steps: int,
    num_trials: int,
    seed: int,
    max_steps: int,
    max_concurrency: int,
    search_space: dict,
    beta: float,
    n_startup_trials: int,
    algorithm: Algorithm,
    exp_dir: str,
) -> dict:
    """Run one seed of the experiment. Returns RegretData dict per algorithm."""
    optimizer = IMABO(
        search_space=search_space,
        seed=seed,
        beta=beta,
        n_startup_trials=n_startup_trials,
        use_tpe=(algorithm == Algorithm.IMABO),
    )

    all_task_ids = [t.id for t in get_tasks(task_set_name=domain)]
    rng = random.Random(seed)
    rng.shuffle(all_task_ids)

    # Reserve last 10 tasks as held-out set for simple regret evaluation
    holdout_ids = all_task_ids[-10:]
    train_ids = all_task_ids[:-10]

    exp_path = RESULT_DIR / exp_dir / f"seed_{seed}"
    exp_path.mkdir(parents=True, exist_ok=True)

    regrets: list[float] = []

    for step in range(n_steps):
        task_id = train_ids[step % len(train_ids)]
        config = optimizer.suggest()
        print(
            f"[seed {seed}][step {step + 1:3d}/{n_steps}] task: {task_id}  suggested: {config}"
        )

        reward = evaluate(
            model_name=config["model_name"],
            llm_args={
                "temperature": config["temperature"],
            },
            domain=domain,
            llm_user=llm_user,
            num_trials=num_trials,
            task_ids=[task_id],
            seed=seed,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            save_to=str(exp_path / f"step_{step:04d}"),
            retrieval_config=(
                config.get("retrieval_config")
                if domain == "banking_knowledge"
                else None
            ),
            top_k=config.get("top_k") if domain == "banking_knowledge" else None,
        )

        optimizer.observe(reward)
        regret = 1.0 - reward
        regrets.append(regret)
        print(
            f"[seed {seed}][step {step + 1:3d}/{n_steps}] reward: {reward:.4f}  regret: {regret:.4f}"
        )

    # Simple regret: re-evaluate best config on held-out tasks
    best = optimizer.best_config
    print(
        f"[seed {seed}] Evaluating best config on {len(holdout_ids)} held-out tasks..."
    )
    holdout_reward = evaluate(
        model_name=best["model_name"],
        llm_args={
            "temperature": best["temperature"],
        },
        domain=domain,
        llm_user=llm_user,
        num_trials=num_trials,
        task_ids=holdout_ids,
        seed=seed,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
        save_to=str(exp_path / "holdout"),
        retrieval_config=(
            best.get("retrieval_config") if domain == "banking_knowledge" else None
        ),
        top_k=best.get("top_k") if domain == "banking_knowledge" else None,
    )
    simple_regret = 1.0 - holdout_reward
    print(f"[seed {seed}] simple regret: {simple_regret:.4f}")

    # Save per-seed summary
    summary = {
        "best_config": best,
        "holdout_reward": holdout_reward,
        "simple_regret": simple_regret,
        "regrets": regrets,
    }
    with open(exp_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        algorithm.value: {
            "regrets": regrets,
            "simple_regrets": simple_regret,
        }
    }


def run_experiment(
    domain: str,
    llm_user: str,
    n_steps: int,
    num_trials: int,
    n_runs: int,
    base_seed: int,
    max_steps: int,
    max_concurrency: int,
    search_space: dict,
    beta: float,
    n_startup_trials: int,
    algorithm: Algorithm,
    exp_dir: str,
) -> None:
    all_results = []
    for i in range(n_runs):
        seed = base_seed + i
        result = run_single_experiment(
            domain=domain,
            llm_user=llm_user,
            n_steps=n_steps,
            num_trials=num_trials,
            seed=seed,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            search_space=search_space,
            beta=beta,
            n_startup_trials=n_startup_trials,
            algorithm=algorithm,
            exp_dir=exp_dir,
        )
        all_results.append(result)

    key = f"{domain.replace('_', '-')}_0D_{n_steps}"
    results_dict = {key: calculate_statistics(all_results)}
    save_results_to_csv(
        results_dict,
        filename=f"{algorithm.value}_{domain}",
        exp_type="tau2",
        result_dir=RESULT_DIR,
    )
    save_iterations_to_csv(
        results_dict,
        filename=f"{algorithm.value}_{domain}",
        exp_type="tau2",
        result_dir=RESULT_DIR,
    )


@dataclass
class Args:
    llm_user: str = "openrouter/openai/gpt-4.1-mini"
    domain: Literal["retail", "airline", "telecom", "mock", "banking_knowledge"] = (
        "retail"
    )
    steps: int = 5
    trials: int = 1
    n_runs: int = 1
    base_seed: int = 42
    max_steps: int = 50
    max_concurrency: int = 20
    algorithm: Algorithm = Algorithm.IMABO
    beta: float = 0.8
    n_startup: int = 10


if __name__ == "__main__":
    args = Args()
    exp_dir = f"{args.algorithm.value.lower()}_{args.domain}_tau2"
    run_experiment(
        domain=args.domain,
        llm_user=args.llm_user,
        n_steps=args.steps,
        num_trials=args.trials,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        search_space=SEARCH_SPACES[args.domain],
        beta=args.beta,
        n_startup_trials=args.n_startup,
        algorithm=args.algorithm,
        exp_dir=exp_dir,
    )
