"""
Multi-query retrieval.

A user's wording often misses relevant chunks ("how do I pay?" vs the doc's
"process a payment transaction"). We ask the LLM to rephrase the question into a
few alternative search queries, retrieve for each, and merge the results. This
widens recall without changing how the user asks.

Kept deliberately small: ``expand_query`` (LLM call with a safe fallback) and
``merge_unique`` (pure list de-duplication, easy to unit-test).
"""
from __future__ import annotations

import logging
from typing import List

from src.core import llm
from src.core.settings import cfg

logger = logging.getLogger(__name__)


def merge_unique(result_lists: List[List[str]], limit: int) -> List[str]:
    """Flatten several result lists into one, dropping duplicates, keep order."""
    seen: set[str] = set()
    merged: List[str] = []
    for results in result_lists:
        for text in results:
            if text not in seen:
                seen.add(text)
                merged.append(text)
    return merged[:limit]


def expand_query(query: str, n: int | None = None, model: str | None = None) -> List[str]:
    """Return the original query plus up to ``n`` LLM-generated paraphrases.

    Falls back to just ``[query]`` if the model is unavailable or returns junk,
    so retrieval never breaks because of the rewrite step.
    """
    n = n or cfg("retrieval", "multi_query_count", default=3)
    prompt = (
        f"Rewrite the question below as {n} alternative search queries that mean "
        f"the same thing but use different words. One per line, no numbering.\n\n"
        f"Question: {query}"
    )
    try:
        text = llm.chat([{"role": "user", "content": prompt}], model=model)
        variants = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001 - rewrite is best-effort
        logger.warning("Query expansion failed (%s); using original only", e)
        variants = []

    # Always include the original; de-dupe; cap to n + 1.
    out: List[str] = [query]
    for v in variants:
        if v and v.lower() != query.lower() and v not in out:
            out.append(v)
    return out[: n + 1]
