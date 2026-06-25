"""
Human feedback loop — the data flywheel.

Real production failures are the most valuable test cases you have, but they
usually evaporate. This captures every rating and promotes down-votes into a
**regression set**: questions the system must keep answering well.

Storage is pluggable behind a tiny ``FeedbackStore`` interface:
  * ``JsonlFeedbackStore`` (default) — append-only JSONL files under ``metrics/``.
    Zero dependencies; great for local/dev.
  * ``PostgresFeedbackStore`` — used automatically when ``DATABASE_URL`` is set.
    Durable and queryable for production. ``psycopg`` is imported lazily.

Public API (``record_feedback`` / ``load_regression_cases`` / ``regression_qa_pairs``)
is unchanged, so callers don't care which backend is active.
"""
from __future__ import annotations

import json
import logging
import os
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


# --------------------------------------------------------------------------- #
# JSONL backend (default)
# --------------------------------------------------------------------------- #
class JsonlFeedbackStore:
    def __init__(self, fb_path: Optional[Path] = None, reg_path: Optional[Path] = None):
        self.fb_path = fb_path or _metrics_dir() / "feedback.jsonl"
        self.reg_path = reg_path or _metrics_dir() / "regression_cases.jsonl"

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def save_feedback(self, entry: Dict[str, Any]) -> None:
        self._append(self.fb_path, entry)

    def save_regression_case(self, case: Dict[str, Any]) -> None:
        self._append(self.reg_path, case)

    def load_regression_cases(self) -> List[Dict[str, Any]]:
        if not self.reg_path.exists():
            return []
        by_question: Dict[str, Dict[str, Any]] = {}
        with open(self.reg_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_question[case.get("question", "")] = case  # last wins
        return [c for q, c in by_question.items() if q]


# --------------------------------------------------------------------------- #
# Postgres backend (when DATABASE_URL is set)
# --------------------------------------------------------------------------- #
class PostgresFeedbackStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._ready = False  # tables created lazily on first use

    def _conn(self):
        import psycopg  # lazy

        return psycopg.connect(self.dsn)

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, ts TIMESTAMPTZ, trace_id TEXT,
                    question TEXT, answer TEXT, rating TEXT, comment TEXT
                );
                CREATE TABLE IF NOT EXISTS regression_cases (
                    question TEXT PRIMARY KEY, reference TEXT, source TEXT,
                    trace_id TEXT, created_at TIMESTAMPTZ
                );
                """
            )
        self._ready = True

    def save_feedback(self, entry: Dict[str, Any]) -> None:
        self._ensure()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (id, ts, trace_id, question, answer, rating, comment) "
                "VALUES (%(id)s, %(timestamp)s, %(trace_id)s, %(question)s, %(answer)s, "
                "%(rating)s, %(comment)s) ON CONFLICT (id) DO NOTHING",
                entry,
            )

    def save_regression_case(self, case: Dict[str, Any]) -> None:
        self._ensure()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO regression_cases (question, reference, source, trace_id, created_at) "
                "VALUES (%(question)s, %(reference)s, %(source)s, %(trace_id)s, %(created_at)s) "
                "ON CONFLICT (question) DO UPDATE SET reference = EXCLUDED.reference, "
                "source = EXCLUDED.source, trace_id = EXCLUDED.trace_id, "
                "created_at = EXCLUDED.created_at",
                case,
            )

    def load_regression_cases(self) -> List[Dict[str, Any]]:
        self._ensure()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT question, reference, source, trace_id, created_at FROM regression_cases"
            )
            cols = ["question", "reference", "source", "trace_id", "created_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_store() -> "JsonlFeedbackStore | PostgresFeedbackStore":
    """Postgres when DATABASE_URL is set, otherwise JSONL files."""
    dsn = os.getenv("DATABASE_URL")
    return PostgresFeedbackStore(dsn) if dsn else JsonlFeedbackStore()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def record_feedback(
    *,
    question: str,
    answer: str,
    rating: str,
    trace_id: Optional[str] = None,
    correct_answer: Optional[str] = None,
    comment: Optional[str] = None,
    store=None,
    fb_path: Optional[Path] = None,
    reg_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist one feedback entry. Down-votes become regression cases."""
    rating = rating.lower().strip()
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")

    if store is None:
        store = JsonlFeedbackStore(fb_path, reg_path) if (fb_path or reg_path) else get_store()

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
    store.save_feedback(entry)

    promoted = False
    if rating == "down":
        store.save_regression_case({
            "question": question,
            "reference": correct_answer,
            "source": "feedback",
            "trace_id": trace_id,
            "created_at": now,
        })
        promoted = True
        logger.info("Down-voted answer promoted to regression set: %.60s", question)

    entry["promoted_to_regression"] = promoted
    return entry


def load_regression_cases(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    store = JsonlFeedbackStore(reg_path=path) if path else get_store()
    return store.load_regression_cases()


def regression_qa_pairs(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Regression cases that carry a reference answer, as {question, answer} pairs."""
    return [
        {"question": c["question"], "answer": c["reference"]}
        for c in load_regression_cases(path)
        if c.get("reference")
    ]
