"""
Request tracing.

Why: In production you cannot debug what you cannot see. When a user reports a
bad answer, you need the trace: the question, what was retrieved, how many agent
steps ran, which tools were called, how long it took, and whether guardrails
passed. Each trace is one JSON line appended to ``metrics/traces.jsonl`` —
greppable, append-only, no database required.

Usage:
    with Tracer("query") as t:
        result = controller.run(question)
        t.update(question=question, steps=result["steps"], ...)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()


def traces_path() -> Path:
    rel = cfg("paths", "metrics_dir", default="metrics")
    return ROOT_DIR / rel / "traces.jsonl"


def record_trace(record: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Append a single trace record as one JSON line."""
    target = path or traces_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    with _WRITE_LOCK:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Tracer:
    """Context manager that times an operation and writes a trace on exit."""

    def __init__(self, operation: str, path: Optional[Path] = None):
        self.operation = operation
        self.path = path
        self.trace_id = uuid.uuid4().hex[:12]
        self.fields: Dict[str, Any] = {}
        self._start = 0.0

    def update(self, **fields: Any) -> "Tracer":
        self.fields.update(fields)
        return self

    def __enter__(self) -> "Tracer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        latency_ms = round((time.perf_counter() - self._start) * 1000, 2)
        record = {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "ok": exc is None,
            **self.fields,
        }
        if exc is not None:
            record["error"] = f"{exc_type.__name__}: {exc}"
        try:
            record_trace(record, self.path)
        except Exception as e:  # never let tracing break the request
            logger.warning("Failed to write trace: %s", e)
        return False  # don't suppress exceptions
