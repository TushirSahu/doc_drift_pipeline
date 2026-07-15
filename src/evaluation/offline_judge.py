"""
Offline eval judge — score answers without a live LLM.

The Ragas judge makes several LLM calls per row per metric, which burns API
credits and fails (NaN) the moment they run out. For CI, local dev, or a
credit-free smoke run, this computes lexical proxies from token overlap instead
— no LLM, no embeddings, deterministic. It is a rough signal, NOT a replacement
for Ragas quality scoring, so its numbers are not comparable to a Ragas baseline.

Enabled with ``evaluation.offline: true``. Reuses the same token-overlap
grounding the serving guardrails already use.
"""
from __future__ import annotations

from src.agentic.guardrails import grounding_score


def _row_scores(question: str, response: str, contexts: list, reference: str) -> dict:
    return {
        # answer supported by the retrieved context
        "faithfulness": grounding_score(response, contexts),
        # answer actually addresses the question
        "answer_relevancy": grounding_score(response, [question]),
        # answer matches the gold reference
        "answer_correctness": grounding_score(response, [reference or ""]),
    }


def lexical_frame(data: dict):
    """Per-row lexical scores as a DataFrame (same shape callers expect from Ragas)."""
    import pandas as pd

    rows = [
        _row_scores(q, r, c, ref)
        for q, r, c, ref in zip(
            data["user_input"], data["response"],
            data["retrieved_contexts"], data["reference"],
        )
    ]
    return pd.DataFrame(rows)


class LexicalResult:
    """Mimics a Ragas result's ``.to_pandas()`` so callers need no changes."""
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df
