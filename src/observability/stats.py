"""
Trace aggregation.

Why: Individual traces help you debug one request; aggregate stats are what you
put on a dashboard or alert on — p50/p95 latency, error rate, tool-usage
frequency, guardrail pass rate. ``/metrics`` and the CLI summarizer call this.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import pg
from src.observability.tracing import load_traces_pg, traces_path


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 2)


def load_traces(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    if path is None and pg.pg_enabled():
        return load_traces_pg()
    target = path or traces_path()
    if not target.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_traces(path: Optional[Path] = None) -> Dict[str, Any]:
    records = load_traces(path)
    if not records:
        return {"count": 0}

    latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
    errors = sum(1 for r in records if not r.get("ok", True))
    grounded = [r for r in records if "grounded" in r]
    grounded_ok = sum(1 for r in grounded if r.get("grounded"))

    tool_counts: Dict[str, int] = {}
    for r in records:
        for tool in r.get("tools_used", []) or []:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return {
        "count": len(records),
        "error_rate": round(errors / len(records), 3),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "avg_steps": round(
            sum(r.get("steps", 0) for r in records) / len(records), 2
        ),
        "tool_usage": tool_counts,
        "grounding_pass_rate": (
            round(grounded_ok / len(grounded), 3) if grounded else None
        ),
    }
