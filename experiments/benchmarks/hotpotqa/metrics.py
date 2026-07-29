import re
import string
from collections import Counter

from nltk.stem import PorterStemmer

from experiments.benchmarks.hotpotqa.types import Reward

_stemmer = PorterStemmer()


def _normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(_stemmer.stem(w) for w in s.split())


def answer_f1(pred: str, gold: str) -> tuple[float, float, float]:
    """Return (f1, precision, recall). Handles yes/no/noanswer edge cases."""
    pred_n, gold_n = _normalize_answer(pred), _normalize_answer(gold)
    yn = {"yes", "no", "noanswer"}
    # If gold is yes/no, snap prediction to its first word if it starts with yes/no
    if gold_n in yn and pred_n not in yn:
        first = pred_n.split()[0] if pred_n else ""
        if first in yn:
            pred_n = first
    if (pred_n in yn or gold_n in yn) and pred_n != gold_n:
        return 0.0, 0.0, 0.0
    p_tokens, g_tokens = pred_n.split(), gold_n.split()
    common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def supporting_facts_f1(
    retrieved_titles: list[str], gold_sp: list[list]
) -> tuple[float, float, float]:
    """Supporting-facts F1 comparing retrieved titles vs gold (title, para_idx) pairs.

    Comparison is title-level only (para_idx ignored) since retrievers return full docs.
    """
    pred_titles = set(retrieved_titles)
    gold_titles = {entry[0] for entry in gold_sp}
    tp = len(pred_titles & gold_titles)
    precision = tp / len(pred_titles) if pred_titles else 0.0
    recall = tp / len(gold_titles) if gold_titles else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return f1, precision, recall


def compute_reward(
    pred_answer: str,
    gold_answer: str,
    retrieved_titles: list[str],
    gold_sp: list[list],
    alpha: float = 0.5,  # weight for answer quality
) -> Reward:
    """Compute the reward for a given prediction and gold answer, retrieved titles and supporting facts."""
    a_f1, a_prec, a_recall = answer_f1(pred_answer, gold_answer)
    sp_f1, sp_prec, sp_recall = supporting_facts_f1(retrieved_titles, gold_sp)
    j_prec = a_prec * sp_prec
    j_recall = a_recall * sp_recall
    j_f1 = 2 * j_prec * j_recall / (j_prec + j_recall) if j_prec + j_recall > 0 else 0.0
    weighted_f1 = alpha * a_f1 + (1 - alpha) * sp_f1
    return Reward(
        joint_f1=j_f1,
        joint_precision=j_prec,
        joint_recall=j_recall,
        answer_f1=a_f1,
        answer_precision=a_prec,
        answer_recall=a_recall,
        sp_f1=sp_f1,
        sp_precision=sp_prec,
        sp_recall=sp_recall,
        weighted_f1=weighted_f1,
    )
