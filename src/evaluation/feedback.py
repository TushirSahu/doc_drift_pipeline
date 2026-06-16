"""
Human feedback loop — the data flywheel.

Real production failures are the most valuable test cases you have, but they
usually evaporate (a user thumbs-downs an answer and it's gone). This module
captures feedback and turns the bad ones into a **regression set**: questions the
system must keep answering well. Every future eval can fold these in, so a fix
for one real failure is protected forever.

Two files (under ``metrics/``, JSONL, append-only):
  * ``feedback.jsonl``        — every rating, for analytics.
  * ``regression_cases.jsonl`` — down-voted questions promoted to test cases.

A regression case carries the question and, when the user supplied a correction,
a ``reference`` answer. Cases with a reference can be scored by Ragas; cases
without one are still replayed and monitored.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _metrics_dir() -> Path:
    return ROOT_DIR / cfg("paths", "metrics_dir", default="metrics")


def feedback_path() -> Path:
    return _metrics_dir() / "feedback.jsonl"


def regression_path() -> Path:
    return _metrics_dir() / "regression_cases.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def record_feedback(
    *,
    question: str,
    answer: str,
    rating: str,
    trace_id: Optional[str] = None,
    correct_answer: Optional[str] = None,
    comment: Optional[str] = None,
    fb_path: Optional[Path] = None,
    reg_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist one feedback entry. Down-votes are promoted to regression cases.

    ``rating`` is "up" or "down". Returns the stored feedback record.
    """
    rating = rating.lower().strip()
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")

    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": now,
        "trace_id": trace_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    _append_jsonl(fb_path or feedback_path(), entry)

    promoted = False
    if rating == "down":
        case = {
            "question": question,
            "reference": correct_answer,  # may be None
            "source": "feedback",
            "trace_id": trace_id,
            "created_at": now,
        }
        _append_jsonl(reg_path or regression_path(), case)
        promoted = True
        logger.info("Down-voted answer promoted to regression set: %.60s", question)

    entry["promoted_to_regression"] = promoted
    return entry


def load_regression_cases(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all stored regression cases (deduplicated by question, last wins)."""
    target = path or regression_path()
    if not target.exists():
        return []
    by_question: Dict[str, Dict[str, Any]] = {}
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_question[case.get("question", "")] = case
    return [c for q, c in by_question.items() if q]


def regression_qa_pairs(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Regression cases that have a reference answer, as {question, answer} pairs.

    These are ready to merge into the evaluation QA set so the drift gate covers
    real reported failures, not just synthetic questions.
    """
    pairs = []
    for case in load_regression_cases(path):
        if case.get("reference"):
            pairs.append({"question": case["question"], "answer": case["reference"]})
    return pairs
