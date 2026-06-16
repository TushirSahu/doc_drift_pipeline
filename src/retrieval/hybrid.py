"""
Hybrid search: combine dense (semantic) + sparse (keyword/BM25) scores.

Dense search is great for meaning ("how do I authenticate?")
but bad for exact tokens ("/api/v2/auth/refresh", "OAuth2").

BM25 catches exact keyword matches; we fuse both with a weighted average.
"""
from typing import List, Tuple

from src.core.settings import cfg


def bm25_search(query: str, corpus: List[str], limit: int) -> List[Tuple[str, float]]:
    """Run BM25 over the local corpus. Returns (doc, score) pairs."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    if not corpus:
        return []

    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)

    ranked = sorted(
        [(corpus[i], float(scores[i])) for i in range(len(corpus))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:limit]


def hybrid_fuse(
    dense_scores: List[Tuple[str, float]],
    sparse_scores: List[Tuple[str, float]],
    alpha: float | None = None,
    limit: int = 5,
) -> List[str]:
    """
    Fuse dense and sparse rankings.
    alpha=0.5 means equal weight; higher alpha favours semantic search.
    """
    alpha = alpha if alpha is not None else cfg("retrieval", "hybrid_alpha", default=0.5)

    def _normalize(scores: List[Tuple[str, float]]) -> dict[str, float]:
        if not scores:
            return {}
        vals = [s for _, s in scores]
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        return {text: (score - lo) / span for text, score in scores}

    d_norm = _normalize(dense_scores)
    s_norm = _normalize(sparse_scores)
    combined: dict[str, float] = {}
    for text in set(d_norm) | set(s_norm):
        combined[text] = alpha * d_norm.get(text, 0.0) + (1 - alpha) * s_norm.get(text, 0.0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:limit]]
