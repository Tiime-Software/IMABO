import hashlib
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

CACHE_FOLDER = Path(__file__).parents[3] / "data" / ".cache"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, corpus: dict) -> None:
        self.doc_ids = list(corpus.keys())
        corpus_hash = hashlib.md5(" ".join(self.doc_ids).encode()).hexdigest()[:12]
        cache_path = CACHE_FOLDER / f"bm25_{corpus_hash}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self._bm25 = pickle.load(f)
        else:
            tokenized = [
                _tokenize(corpus[d].get("title", "") + " " + corpus[d].get("text", ""))
                for d in tqdm(self.doc_ids, desc="Tokenizing corpus")
            ]
            self._bm25 = BM25Okapi(tokenized)
            with open(cache_path, "wb") as f:
                pickle.dump(self._bm25, f)

        self._score_cache: dict[str, np.ndarray] = {}

    def retrieve(self, question: str, top_k: int) -> list[str]:
        if question not in self._score_cache:
            self._score_cache[question] = self._bm25.get_scores(_tokenize(question))
        scores = self._score_cache[question]
        k = min(top_k, len(self.doc_ids))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.doc_ids[i] for i in top]


DENSE_MODEL = "all-MiniLM-L6-v2"


class DenseIndex:
    def __init__(self, corpus: dict, model_name: str = DENSE_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.doc_ids = list(corpus.keys())
        corpus_hash = hashlib.md5(" ".join(self.doc_ids).encode()).hexdigest()[:12]
        cache_path = CACHE_FOLDER / f"{model_name.replace('/', '_')}_{corpus_hash}.npy"

        if cache_path.exists():
            self._embeddings = np.load(cache_path)
        else:
            texts = [
                corpus[d].get("title", "") + " " + corpus[d].get("text", "")
                for d in self.doc_ids
            ]
            model = SentenceTransformer(model_name)
            self._embeddings = model.encode(
                texts, batch_size=512, show_progress_bar=True, normalize_embeddings=True
            )
            np.save(cache_path, self._embeddings)

        self._model = SentenceTransformer(model_name)
        self._query_cache: dict[str, np.ndarray] = {}

    def retrieve(self, question: str, top_k: int) -> list[str]:
        if question not in self._query_cache:
            self._query_cache[question] = self._model.encode(
                [question], normalize_embeddings=True
            )[0]
        q_emb = self._query_cache[question]
        scores = self._embeddings @ q_emb
        k = min(top_k, len(self.doc_ids))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.doc_ids[i] for i in top]
