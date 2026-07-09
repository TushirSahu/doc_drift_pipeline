"""Export evaluation results to metrics/ for tracking over time."""
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.core.blob_store import write_metrics_json
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
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    # Writes the file and, when DATABASE_URL is set, a durable Postgres blob.
    path = write_metrics_json(filename, payload)
    logger.info("Exported JSON → %s", path)
    return path
