"""Optional second-stage reranker: re-orders the retrieved pool before the top-k cut.

Uses a cross-encoder when sentence-transformers is installed (and the model loads),
otherwise a small lexical reranker so the with/without comparison still works offline.
"""
import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .retrievers import tokenize


def _order(scores) -> list[int]:
    s = np.asarray(scores, dtype=float)
    return [int(i) for i in np.lexsort((np.arange(len(s)), -s))]  # ties: lower index first


class CrossEncoderReranker:
    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.name = os.getenv("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(self.name)

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        return _order(self.model.predict([(query, t) for t in texts]))


class LexicalReranker:
    """Query-term coverage plus adjacent-pair matches; ignores chunk length, unlike BM25."""

    name = "lexical (offline fallback)"

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in ENGLISH_STOP_WORDS]

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        q = self._terms(query)
        qset, qpairs = set(q), set(zip(q, q[1:]))
        scores = []
        for t in texts:
            c = self._terms(t)
            cover = len(qset & set(c)) / (len(qset) or 1)
            pairs = len(qpairs & set(zip(c, c[1:]))) / (len(qpairs) or 1)
            scores.append(cover + 0.5 * pairs)
        return _order(scores)


@lru_cache(maxsize=1)  # loading a cross-encoder is slow; do it once per process
def get_reranker():
    try:
        return CrossEncoderReranker()
    except Exception:  # not installed, or the model could not be downloaded
        return LexicalReranker()
