import json
import time
from pathlib import Path

import numpy as np
from beir.datasets.data_loader import GenericDataLoader

from experiments.benchmarks.hotpotqa.embedding import DenseIndex
from experiments.benchmarks.hotpotqa.metrics import Reward, compute_reward
from experiments.benchmarks.hotpotqa.types import Result


class HotpotQABenchmark:
    """
    BEIR-backed HotpotQA benchmark for RAG pipeline HPO.

    IMABO search space (retrieval side):
        top_k : int [1, 10]  passages passed to the LLM

    Add LLM params (e.g. temperature) to param_specs before passing to IMABO.

    Usage:
        bench = HotpotQABenchmark()

        def my_llm(question: str, passages: list[str]) -> str:
            # your OpenRouter call here
            ...

        reward = bench.eval_config(config, llm_fn=my_llm)
    """

    def __init__(
        self,
        data_folder: Path,
        seed: int = 42,
        n_samples: int = 1000,
        n_holdout: int = 0,
        n_corpus_sample: int | None = None,
        split: str = "test",
    ) -> None:
        """Initialize the HotpotQA benchmark.

        Train queries are a stable prefix of the seeded permutation and holdout
        queries a fixed tail, so a larger ``n_samples`` run reuses the cache and
        optimizer prefix of a smaller one (holdout depends only on ``n_holdout``).

        Args:
            data_folder: Path to the data folder.
            seed: Random seed for reproducibility.
            n_samples: Number of training queries (prefix of the permutation).
            n_holdout: Number of holdout queries (tail of the permutation).
            n_corpus_sample: Number of corpus samples to load for the experiment. If None, all corpus samples are loaded (approximately 5M passages).
            split: Split to use for the experiment. Options: "test", "train", "dev".
        """
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(
            split=split
        )

        # Gold answers and supporting facts live in queries.jsonl metadata
        gold: dict[str, str] = {}
        supporting_facts: dict[str, list] = {}
        with open(data_folder / "queries.jsonl") as f:
            for line in f:
                q = json.loads(line)
                meta = q.get("metadata", {})
                ans = meta.get("answer", "")
                if ans:
                    gold[q["_id"]] = ans
                sp = meta.get("supporting_facts", [])
                if sp:
                    supporting_facts[q["_id"]] = sp

        # Keep the full metadata maps so the question split can be re-sampled
        # per seed (resample()) without rebuilding the corpus/index.
        self._all_queries = queries
        self._all_gold = gold
        self._all_supporting = supporting_facts
        self._qrels = qrels
        self._valid = [qid for qid in queries if qid in qrels and qid in gold]
        self.n_corpus_sample = n_corpus_sample

        # Initial question split for this seed.
        self._sample_split(seed, n_samples, n_holdout)

        # Sample a subset of the corpus for the Dense index. Only when
        # n_corpus_sample is set: the subset then depends on the sampled
        # questions, which is why resample() is disallowed in that case.
        if n_corpus_sample is not None and n_corpus_sample < len(corpus):
            rng = np.random.RandomState(seed)
            chosen = self.train_qids + self.holdout_qids
            relevant = {d for qid in chosen for d in qrels[qid] if d in corpus}
            rest = [d for d in corpus if d not in relevant]
            extra = rng.choice(
                rest,
                size=min(n_corpus_sample - len(relevant), len(rest)),
                replace=False,
            ).tolist()
            corpus = {d: corpus[d] for d in list(relevant) + extra}
        print("--- Loading Embeddings ---")
        start_time = time.time()
        # self._bm25 = BM25Index(corpus)
        # bm25_time = time.time()
        self._dense = DenseIndex(corpus)
        dense_time = time.time()
        # print(f"--- BM25 index loaded in {bm25_time - start_time:.2f} seconds ---")
        print(f"--- Dense index loaded in {dense_time - start_time:.2f} seconds ---")
        self._corpus = corpus

    def _sample_split(self, seed: int, n_samples: int, n_holdout: int) -> None:
        """(Re)compute the train/holdout question split for ``seed``.

        Train is a stable prefix of the seeded permutation and holdout a fixed
        tail, so a larger ``n_samples`` run reuses a smaller run's prefix (and
        its cache/checkpoint) for the same seed. The two never overlap given the
        size check.
        """
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self._valid).tolist()
        n_train = min(n_samples, len(perm))
        if n_train + n_holdout > len(perm):
            raise ValueError(
                f"n_samples ({n_samples}) + n_holdout ({n_holdout}) exceeds "
                f"available queries ({len(perm)})."
            )
        self.train_qids = perm[:n_train]
        self.holdout_qids = perm[len(perm) - n_holdout :] if n_holdout > 0 else []
        print(
            f"--- {len(self.train_qids)} train + {len(self.holdout_qids)} "
            f"holdout queries sampled (seed {seed}) ---"
        )

    def resample(
        self, seed: int, n_samples: int | None = None, n_holdout: int | None = None
    ) -> "HotpotQABenchmark":
        """Draw a fresh train/holdout split for a new seed, reusing the index.

        Valid only with the full corpus (``n_corpus_sample=None``): the Dense
        index spans the whole corpus and is independent of the question sample,
        so each per-seed run reuses the same embeddings instead of rebuilding
        them. With a subsampled corpus the index depends on the sample and must
        be rebuilt (construct a new benchmark instead). ``n_samples`` /
        ``n_holdout`` default to the current split sizes.
        """
        if self.n_corpus_sample is not None:
            raise ValueError(
                "resample() requires the full corpus (n_corpus_sample=None); "
                "with a subsampled corpus the index must be rebuilt per seed."
            )
        n_samples = len(self.train_qids) if n_samples is None else n_samples
        n_holdout = len(self.holdout_qids) if n_holdout is None else n_holdout
        self._sample_split(seed, n_samples, n_holdout)
        return self

    def _retrieve_doc_ids(self, question: str, config: dict) -> list[str]:
        top_k = int(config["top_k"])
        # if config.get("retrieval") == "dense":
        return self._dense.retrieve(question, top_k=top_k)
        # return self._bm25.retrieve(question, top_k=top_k)

    def eval_question(self, qid: str, config: dict, llm_fn) -> Result:
        """Evaluate a single question.

        Reads from the full ``_all_*`` maps (built once in ``__init__`` and
        never mutated again) rather than the per-split ``train_qids``/
        ``holdout_qids`` state, so concurrent calls are safe even while
        another thread is mid-``resample()``.
        """
        question = self._all_queries[qid]
        supporting_facts = self._all_supporting.get(qid, [])
        gold = self._all_gold[qid]
        doc_ids = self._retrieve_doc_ids(question, config)
        passages = [
            self._corpus[d].get("title", "") + ": " + self._corpus[d].get("text", "")
            for d in doc_ids
        ]
        retrieved_titles = [self._corpus[d].get("title", "") for d in doc_ids]
        pred = llm_fn(question, config, passages)
        reward: Reward = compute_reward(pred, gold, retrieved_titles, supporting_facts)

        return Result(
            qid=qid,
            question=question,
            retrieved_titles=retrieved_titles,
            supporting_facts=supporting_facts,
            passages=passages,
            gold=gold,
            pred=pred,
            reward=reward,
        )
