"""
Maximal Marginal Relevance (MMR) selection.

Problem: dense search often returns near-duplicate chunks
("Auth uses JWT" and "JWT tokens are used for auth").

MMR balances relevance to the query vs. diversity among selected chunks.
lambda=1.0 → pure relevance; lambda=0.0 → pure diversity.
"""
import math
from typing import List, Tuple

from src.core.settings import cfg


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


def mmr_select(
    query_embedding: List[float],
    candidates: List[Tuple[str, List[float]]],
    limit: int,
    lambda_param: float | None = None,
) -> List[str]:
    lambda_param = lambda_param or cfg("retrieval", "mmr_lambda", default=0.5)
    if not candidates:
        return []

    selected: List[Tuple[str, List[float]]] = []
    remaining = list(candidates)

    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = float("-inf")
        for idx, (text, emb) in enumerate(remaining):
            relevance = cosine_similarity(query_embedding, emb)
            redundancy = 0.0
            if selected:
                redundancy = max(
                    cosine_similarity(emb, s_emb) for _, s_emb in selected
                )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return [text for text, _ in selected]
