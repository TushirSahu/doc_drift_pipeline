import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)


def finite_only(scores: Dict[str, float]) -> Dict[str, float]:
    """Drop NaN/inf metrics. A weak judge can NaN a Ragas metric; keeping it would
    write invalid JSON and silently skew the gate. Skip it instead."""
    return {
        k: float(v) for k, v in scores.items()
        if isinstance(v, (int, float)) and math.isfinite(v)
    }

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def baseline_path() -> Path:
    rel = cfg("drift", "baseline_path", default="metrics/baseline.json")
    return ROOT_DIR / rel


def save_baseline(scores: Dict[str, float], path: Path | None = None) -> Path:
    target = path or baseline_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": finite_only(scores),   # never commit NaN into the baseline
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Baseline saved to %s", target)
    return target


def load_baseline(path: Path | None = None) -> Dict[str, float] | None:
    target = path or baseline_path()
    if not target.exists():
        return None
    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("scores", data)


def check_drift(
    current: Dict[str, float],
    baseline: Dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    regression_threshold = cfg("drift", "regression_threshold", default=0.05)
    faithfulness_threshold = cfg("drift", "faithfulness_threshold", default=0.8)
    current = finite_only(current)   # a NaN metric is skipped, not silently compared
    reasons: list[str] = []

    faith = current.get("faithfulness")
    if faith is not None and faith < faithfulness_threshold:
        reasons.append(
            f"faithfulness {faith:.3f} below threshold {faithfulness_threshold}"
        )

    if baseline:
        for metric in METRICS:
            if metric not in current or metric not in baseline:
                continue
            delta = baseline[metric] - current[metric]
            if delta > regression_threshold:
                reasons.append(
                    f"{metric} regressed by {delta:.3f} "
                    f"(baseline {baseline[metric]:.3f} -> current {current[metric]:.3f})"
                )

    return len(reasons) == 0, reasons


def enforce_drift_or_exit(current: Dict[str, float], set_baseline: bool = False) -> None:
    if set_baseline:
        save_baseline(current)
        return

    baseline = load_baseline()
    if baseline is None:
        logger.warning("No baseline found — saving current run as baseline.")
        save_baseline(current)
        return

    passed, reasons = check_drift(current, baseline)
    if not passed:
        for reason in reasons:
            logger.error("DRIFT DETECTED: %s", reason)
        sys.exit(1)

    logger.info("Drift check passed.")
