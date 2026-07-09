"""
Durable store for the small JSON snapshots under ``metrics/``.

The benchmark and eval steps write latest-wins blobs — ``model_scores.json``,
``champion.json``, ``latest_eval.json`` — that the API and dashboard read back.
On an ephemeral host those files vanish on restart. :func:`write_metrics_json`
always writes the file (so local tooling and the static dashboard keep working)
and additionally upserts the blob into Postgres when ``DATABASE_URL`` is set;
:func:`read_metrics_json` prefers the durable blob, falling back to the file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.core import pg
from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)


def metrics_dir() -> Path:
    return ROOT_DIR / cfg("paths", "metrics_dir", default="metrics")


_READY = False


def _ensure_table() -> None:
    global _READY
    if _READY:
        return
    pg.execute(
        "CREATE TABLE IF NOT EXISTS json_blobs ("
        "name TEXT PRIMARY KEY, data JSONB, updated_at TIMESTAMPTZ DEFAULT now())"
    )
    _READY = True


def write_metrics_json(name: str, payload: dict) -> Path:
    """Write ``metrics/<name>`` and, when Postgres is enabled, upsert the blob."""
    out = metrics_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if pg.pg_enabled():
        try:
            _ensure_table()
            pg.execute(
                "INSERT INTO json_blobs (name, data, updated_at) "
                "VALUES (%(name)s, %(data)s::jsonb, now()) "
                "ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data, "
                "updated_at = now()",
                {"name": name, "data": json.dumps(payload, default=str)},
            )
        except Exception as e:  # noqa: BLE001 - durability is best-effort
            logger.warning("Postgres blob write failed for %s: %s", name, e)
    return path


def read_metrics_json(name: str) -> Optional[dict]:
    """Return the blob from Postgres when enabled, else the local file, else None."""
    if pg.pg_enabled():
        try:
            _ensure_table()
            rows = pg.query("SELECT data FROM json_blobs WHERE name = %(name)s",
                            {"name": name})
            if rows:
                return rows[0][0]
        except Exception as e:  # noqa: BLE001
            logger.warning("Postgres blob read failed for %s: %s", name, e)

    path = metrics_dir() / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
    return None
