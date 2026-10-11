"""Sweep chunk size x retriever x k and report recall@k, MRR and context cost."""
import re
from dataclasses import dataclass, field

from .chunking import chunk_text
from .rerank import get_reranker
from .retrievers import build

RETRIEVERS = ("bm25", "dense", "hybrid")


@dataclass
class Config:
    chunk_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    overlap_ratio: float = 0.125
    ks: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 12])
    retrievers: list[str] = field(default_factory=lambda: list(RETRIEVERS))
    target_recall: float = 0.9
    rerank: bool = False
    rerank_pool: int = 20  # candidates the reranker sees (at least max k)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def validate(docs, qa, cfg: Config):
    if not docs:
        raise ValueError("Add at least one document.")
    if not qa:
        raise ValueError("Add at least one question with an expected answer.")
    for q in qa:
        if not q.get("question") or not q.get("answer"):
            raise ValueError("Every QA item needs 'question' and 'answer'.")
    if not cfg.chunk_sizes or min(cfg.chunk_sizes) < 2:
        raise ValueError("Chunk sizes must be 2 or more words.")
    if not cfg.ks or min(cfg.ks) < 1:
        raise ValueError("k values must be 1 or more.")
    if not 0 <= cfg.overlap_ratio < 1:
        raise ValueError("Overlap ratio must be between 0 and 1.")
    bad = [r for r in cfg.retrievers if r not in RETRIEVERS]
    if bad or not cfg.retrievers:
        raise ValueError(f"Retrievers must be from {RETRIEVERS}.")


def _first_rank(order: list[int], gold: set[int]):
    return next((r for r, c in enumerate(order, 1) if c in gold), None)


def _rows(name, size, overlap, rerank, ranks, total, ks, avg_tokens, n_chunks) -> list[dict]:
    """`ranks` covers only answerable questions; `total` is every question asked.
    Unanswerable questions count as misses in recall/MRR so small chunk sizes aren't flattered."""
    answerable = len(ranks)
    mrr = sum(1 / r for r in ranks if r) / total
    rows = []
    for k in ks:
        hits = sum(1 for r in ranks if r and r <= k)
        rows.append({"retriever": name, "chunk_size": size, "overlap": overlap, "rerank": rerank, "k": k,
                     "recall": round(hits / total, 4),
                     "recall_answerable": round(hits / answerable, 4) if answerable else 0.0,
                     "mrr": round(mrr, 4), "context_tokens": round(k * avg_tokens), "chunks": n_chunks,
                     "evaluated": answerable, "total": total,
                     "unanswerable_share": round(1 - answerable / total, 4)})
    return rows


def position_check(ranks: list, k: int) -> dict:
    """Where the answer chunk lands among the k retrieved. The middle third is the
    zone models tend to under-use ("lost in the middle")."""
    head = round(k / 3)
    middle = (head + 1, k - head) if k >= 3 and k - 2 * head >= 1 else None
    found = [r for r in ranks if r and r <= k]
    return {"k": k, "scored": len(ranks), "found": len(found), "middle": middle,
            "in_middle": sum(1 for r in found if middle and middle[0] <= r <= middle[1]),
            "positions": {p: found.count(p) for p in range(1, k + 1)}}


def run(docs: list[dict], qa: list[dict], cfg: Config) -> dict:
    validate(docs, qa, cfg)
    ks = sorted(set(cfg.ks))
    kmax = ks[-1]
    reranker = get_reranker() if cfg.rerank else None
    pool = max(kmax, cfg.rerank_pool)
    rows, unanswerable, rank_log = [], {}, {}
    for size in sorted(set(cfg.chunk_sizes)):
        overlap = min(size - 1, int(size * cfg.overlap_ratio))
        chunks = [c for d in docs for c in chunk_text(d["name"], d["text"], size, overlap)]
        if not chunks:
            raise ValueError("Documents contain no text.")
        texts = [c.text for c in chunks]
        norm_texts = [_norm(t) for t in texts]
        gold = []
        for q in qa:
            ans = _norm(q["answer"])
            doc = q.get("doc")
            gold.append({i for i, t in enumerate(norm_texts)
                         if ans in t and (not doc or chunks[i].doc == doc)})
        unanswerable[size] = sum(1 for g in gold if not g)
        scored = [i for i, g in enumerate(gold) if g]
        avg_tokens = sum(c.tokens for c in chunks) / len(chunks)
        for name, retr in build(texts, cfg.retrievers).items():
            # 1-based rank of the first gold chunk, None if beyond kmax. One retrieval per question.
            plain, reranked = [], []
            for i in scored:
                order = retr.rank(qa[i]["question"], pool if reranker else kmax)
                plain.append(_first_rank(order[:kmax], gold[i]))
                if reranker:
                    new = reranker.rerank(qa[i]["question"], [texts[c] for c in order])
                    reranked.append(_first_rank([order[j] for j in new][:kmax], gold[i]))
            variants = [(False, plain)] + ([(True, reranked)] if reranker else [])
            for flag, ranks in variants:
                rank_log[(name, size, flag)] = ranks
                rows += _rows(name, size, overlap, flag, ranks, len(qa), ks, avg_tokens, len(chunks))
    best = recommend(rows, cfg.target_recall)
    ranks = rank_log[(best["retriever"], best["chunk_size"], best["rerank"])]
    return {"rows": rows, "unanswerable": unanswerable, "recommendation": best,
            "reranker": reranker.name if reranker else None,
            "rerank_summary": _rerank_summary(rows) if reranker else [],
            "position_check": position_check(ranks, best["k"])}


def _rerank_summary(rows: list[dict]) -> list[dict]:
    """Mean recall@k over all retriever x chunk-size settings, without vs with the reranker."""
    out = []
    for k in sorted({r["k"] for r in rows}):
        avg = {f: [r["recall"] for r in rows if r["k"] == k and r["rerank"] is f] for f in (False, True)}
        a, b = (sum(avg[f]) / len(avg[f]) for f in (False, True))
        out.append({"k": k, "without": round(a, 4), "with": round(b, 4), "gain": round(b - a, 4)})
    return out


def recommend(rows: list[dict], target: float) -> dict:
    ok = [r for r in rows if r["recall"] >= target]
    if ok:
        best = min(ok, key=lambda r: (r["context_tokens"], -r["recall"], r["k"]))
        return {"met_target": True, "target": target, **best}
    best = max(rows, key=lambda r: (r["recall"], -r["context_tokens"]))
    return {"met_target": False, "target": target, **best}
