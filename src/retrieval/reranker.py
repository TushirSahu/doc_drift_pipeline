"""
Two-stage retrieval: retrieve wide, rerank narrow.

Stage 1: fetch top-10 candidates cheaply (vector search).
Stage 2: ask the LLM to rank them by relevance to the question.

This is a lightweight alternative to cross-encoder rerankers used in production.
"""
import logging
from typing import List
from ollama import chat

logger = logging.getLogger(__name__)


def llm_rerank(question: str,candidates: List[str],model_name: str,limit: int,) -> List[str]:

    if len(candidates) <= limit:
        return candidates


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
