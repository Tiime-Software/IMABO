from dataclasses import dataclass


@dataclass
class Reward:
    joint_f1: float
    joint_precision: float
    joint_recall: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    sp_f1: float
    sp_precision: float
    sp_recall: float
    weighted_f1: float


@dataclass
class Result:
    qid: str
    question: str
    retrieved_titles: list[str]
    supporting_facts: list[list]
    passages: list[str]
    gold: str
    pred: str
    reward: Reward


@dataclass
class BatchResult:
    results: list[Result]
    rewards: list[float]
    avg_reward: float
