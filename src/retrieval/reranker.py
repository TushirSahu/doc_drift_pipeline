"""
Two-stage retrieval: retrieve wide, rerank narrow.

Stage 1: fetch top-N candidates cheaply (vector search).
Stage 2: re-order them by true relevance and keep the best `limit`.

Two reranker strategies, chosen by config (`retrieval.reranker`):

  * ``cross_encoder`` (production default) — a cross-encoder model
    (e.g. ``BAAI/bge-reranker-base``) scores each (query, chunk) pair jointly.
    This is the industry-standard reranking approach: far more accurate than
    bi-encoder vector similarity because the model sees the query and the chunk
    *together* instead of comparing two independently-computed embeddings.
  * ``llm`` — ask the generation LLM to rank the chunks. A cheap stand-in when
    no reranker model is available; kept for environments without
    sentence-transformers/torch.

``sentence-transformers`` (and torch) are heavy and not always installed, so the
cross-encoder import is deferred; if it's missing we log and fall back to the
original order rather than crashing the request.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from src.core.settings import cfg

logger = logging.getLogger(__name__)

# Cache one CrossEncoder per model name — loading weights is expensive.
_CROSS_ENCODERS: dict = {}


def _rank_by_scores(candidates: List[str], scores: List[float], limit: int) -> List[str]:
    """Pure helper: order candidates by descending score, keep `limit`.

    Separated out so the ranking logic is unit-testable without a model.
    """
    paired: List[Tuple[str, float]] = list(zip(candidates, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in paired[:limit]]


def _get_cross_encoder(model_name: str):
    """Load (and cache) a sentence-transformers CrossEncoder."""
    model = _CROSS_ENCODERS.get(model_name)
    if model is None:
        from sentence_transformers import CrossEncoder  # deferred heavy import

        model = CrossEncoder(model_name)
        _CROSS_ENCODERS[model_name] = model
    return model


def cross_encoder_rerank(
    question: str,
    candidates: List[str],
    limit: int,
    model_name: Optional[str] = None,
    score_fn: Optional[Callable[[List[Tuple[str, str]]], List[float]]] = None,
) -> List[str]:
    """Rerank with a cross-encoder. Falls back to original order if unavailable.

    ``score_fn`` is injectable for testing — given a list of (query, chunk) pairs
    it returns a relevance score per pair.
    """
    if len(candidates) <= limit:
        return candidates

    model_name = model_name or cfg(
        "retrieval", "reranker_model", default="BAAI/bge-reranker-base"
    )
    pairs = [(question, c) for c in candidates]
    try:
        if score_fn is None:
            model = _get_cross_encoder(model_name)
            scores = list(model.predict(pairs))
        else:
            scores = score_fn(pairs)
    except Exception as e:  # noqa: BLE001 — never fail the request on a reranker
        logger.warning("Cross-encoder rerank unavailable (%s); using original order", e)
        return candidates[:limit]

    return _rank_by_scores(candidates, [float(s) for s in scores], limit)


def llm_rerank(
    question: str,
    candidates: List[str],
    model_name: str,
    limit: int,
) -> List[str]:
    if len(candidates) <= limit:
        return candidates

    from ollama import chat  # deferred so importing this module needs no backend

    numbered = "\n".join(f"{i + 1}. {c[:300]}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {question}\n\n"
        f"Rank these document chunks by relevance (most relevant first). "
        f"Reply with only comma-separated numbers, e.g. 3,1,2\n\n{numbered}"
    )
    try:
        response = chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        order_str = response.message.content.strip()
        indices = [
            int(x.strip()) - 1
            for x in order_str.split(",")
            if x.strip().isdigit()
        ]
        reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        for c in candidates:
            if c not in reranked:
                reranked.append(c)
        return reranked[:limit]
    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)
        return candidates[:limit]


def rerank(
    question: str,
    candidates: List[str],
    limit: int,
    model_name: Optional[str] = None,
    strategy: Optional[str] = None,
) -> List[str]:
    """Dispatch to the configured reranker (`cross_encoder` or `llm`)."""
    strategy = strategy or cfg("retrieval", "reranker", default="cross_encoder")
    if strategy == "llm":
        llm_model = model_name or cfg("models", "llm", default="llama3.2:3b")
        return llm_rerank(question, candidates, llm_model, limit)
    return cross_encoder_rerank(question, candidates, limit, model_name=model_name)
