"""BM25 (lexical), dense (embedding) and hybrid (reciprocal rank fusion) retrievers.

Every retriever returns a full ranking (best first) so recall@k for any k can be read
off one retrieval. The default dense embedder is TF-IDF + truncated SVD (LSA), which runs
offline. Set RAG_EMBEDDER=sentence-transformers to use a real embedding model instead.
"""
import math
import os
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.len.mean() if len(texts) else 0.0
        n = len(texts)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = 1 - self.b + self.b * self.len[i] / (self.avg or 1)
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def rank(self, query: str, n: int) -> list[int]:
        return _top(self.scores(query), n)


class DenseSVD:
    """Offline stand-in for an embedding model: TF-IDF followed by LSA."""

    def __init__(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        X = self.vec.fit_transform(texts)
        comps = min(128, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=comps, random_state=0) if comps >= 2 else None
        self.M = self._norm(self.svd.fit_transform(X) if self.svd else X.toarray())

    @staticmethod
    def _norm(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(n == 0, 1, n)

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        q = self.svd.transform(q) if self.svd else q.toarray()
        return self.M @ self._norm(q)[0]

    def rank(self, query: str, n: int) -> list[int]:
        return _top(self.scores(query), n)


class DenseST:
    def __init__(self, texts: list[str]):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(os.getenv("RAG_ST_MODEL", "all-MiniLM-L6-v2"))
        self.M = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        return self.M @ self.model.encode([query], normalize_embeddings=True)[0]

    def rank(self, query: str, n: int) -> list[int]:
        return _top(self.scores(query), n)


class Hybrid:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (c + rank)."""

    def __init__(self, bm25, dense, c: int = 60):
        self.bm25, self.dense, self.c = bm25, dense, c

    def rank(self, query: str, n: int) -> list[int]:
        pool = max(50, n * 3)
        fused: dict[int, float] = {}
        for lst in (self.bm25.rank(query, pool), self.dense.rank(query, pool)):
            for r, i in enumerate(lst, 1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.c + r)
        return [i for i, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))][:n]


def _top(scores: np.ndarray, n: int) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))  # ties: lower index first
    return [int(i) for i in order[:n]]


def build(texts: list[str], names: list[str]) -> dict:
    bm = BM25(texts)
    dense = DenseST(texts) if os.getenv("RAG_EMBEDDER") == "sentence-transformers" else DenseSVD(texts)
    all_r = {"bm25": bm, "dense": dense, "hybrid": Hybrid(bm, dense)}
    return {k: v for k, v in all_r.items() if k in names}
