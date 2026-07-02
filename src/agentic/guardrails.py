"""
Answer guardrails.

Why: The system prompt *asks* the LLM for grounded, cited answers — but nothing
*enforces* it. The model can ignore the rules and hallucinate. For a
documentation tool, an unsourced or invented answer is worse than "I don't know."

These checks run on the final answer using the ``retrieved_contexts`` the
controller already collects, and produce a measurable verdict the API can act on
(warn, downgrade, or refuse) instead of shipping a bad answer silently.

The grounding check is intentionally lightweight (lexical token overlap), so it
adds no extra LLM call and no dependency. It's a guardrail, not a proof — it
catches the obvious failure where an answer shares almost no vocabulary with
anything that was retrieved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List

_WORD_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[source:[^\]]+\]", re.IGNORECASE)

# Tokens too common to count as evidence of grounding.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for",
    "with", "by", "you", "your", "it", "this", "that", "as", "at", "be", "from",
    "use", "uses", "used", "if", "not", "no", "do", "does", "can", "will",
}

_IDK_MARKERS = ("i don't know", "i do not know", "no relevant", "cannot find",
                "not found", "no information")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def grounding_score(answer: str, contexts: List[str]) -> float:
    """Fraction of the answer's meaningful tokens that appear in the context.

    1.0 = every content word is supported by retrieved text; 0.0 = none are.
    """
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= _tokens(c)
    if not ctx_tokens:
        return 0.0
    overlap = ans_tokens & ctx_tokens
    return round(len(overlap) / len(ans_tokens), 3)


@dataclass
class GuardrailResult:
    grounded: bool
    grounding_score: float
    has_citation: bool
    is_idk: bool
    reasons: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def check_answer(
    answer: str,
    contexts: List[str],
    min_grounding: float = 0.3,
    require_citation: bool = True,
) -> GuardrailResult:
    """Validate a final answer against the context it was supposed to use."""
    answer = (answer or "").strip()
    is_idk = any(m in answer.lower() for m in _IDK_MARKERS)
    score = grounding_score(answer, contexts)
    has_citation = bool(_CITATION_RE.search(answer))

    reasons: List[str] = []

    # An honest "I don't know" is acceptable and not held to grounding/citation.
    if is_idk:
        return GuardrailResult(
            grounded=True, grounding_score=score, has_citation=has_citation,
            is_idk=True, reasons=["answer abstained (I don't know)"],
        )

    if not contexts:
        reasons.append("no context was retrieved")
    if score < min_grounding:
        reasons.append(
            f"low grounding score {score:.2f} < {min_grounding} (possible hallucination)"
        )
    if require_citation and not has_citation:
        reasons.append("missing [Source: ...] citation")

    grounded = len([r for r in reasons if "grounding" in r or "no context" in r]) == 0
    return GuardrailResult(
        grounded=grounded,
        grounding_score=score,
        has_citation=has_citation,
        is_idk=False,
        reasons=reasons or ["passed"],
    )
