"""Search index supporting TF-IDF, BM25, and a hybrid ranker."""
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


class MovieSearchEngine:
    def __init__(self, data_path: str | Path):
        self.movies = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self._build_index()

    def _build_index(self) -> None:
        self.documents = [
            " ".join([
                m["title"],
                m.get("director", ""),
                " ".join(m.get("genres", [])),
                m.get("cast", ""),
                m["plot"],
            ])
            for m in self.movies
        ]

        self.tokenized_docs = [preprocess(d) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.vectorizer = TfidfVectorizer(
            tokenizer=preprocess,
            token_pattern=None,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def search(self, query: str, method: str = "bm25", top_k: int = 10, alpha: float = 0.5):
        query = query.strip()
        if not query:
            return []

        if method == "tfidf":
            scores = self._tfidf_scores(query)
        elif method == "bm25":
            scores = self._bm25_scores(query)
        elif method == "hybrid":
            scores = self._hybrid_scores(query, alpha)
        else:
            raise ValueError(f"Unknown ranking method: {method}")

        return self._rank(scores, top_k)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.tfidf_matrix).flatten()

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(preprocess(query)))

    def _hybrid_scores(self, query: str, alpha: float) -> np.ndarray:
        tfidf = self._tfidf_scores(query)
        bm25 = self._bm25_scores(query)
        return alpha * _minmax(tfidf) + (1.0 - alpha) * _minmax(bm25)

    def _rank(self, scores: np.ndarray, top_k: int) -> list[dict]:
        order = np.argsort(-scores)[:top_k]
        return [
            {**self.movies[i], "score": float(scores[i])}
            for i in order
            if scores[i] > 0
        ]


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
