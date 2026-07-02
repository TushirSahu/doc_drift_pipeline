"""Export evaluation results to metrics/ for tracking over time."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)


def metrics_dir() -> Path:
    rel = cfg("paths", "metrics_dir", default="metrics")
    return ROOT_DIR / rel


def export_csv(df, filename: str) -> Path:
    out = metrics_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    logger.info("Exported CSV → %s", path)
    return path


def export_json(data: dict, filename: str) -> Path:
    out = metrics_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Exported JSON → %s", path)
    return path
