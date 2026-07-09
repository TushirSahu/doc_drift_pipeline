"""
Optional Postgres backing for runtime state.

Hugging Face Spaces (and most container hosts) have an ephemeral filesystem, so
anything written under ``metrics/`` at runtime is lost on restart. When
``DATABASE_URL`` is set, the trace/blob/ingest-state stores persist to Postgres
instead of local files; when it is unset, everything falls back to the original
file behavior, so local dev and tests are unchanged.

This module is the single seam the stores talk to. ``psycopg`` and the pool are
imported lazily so importing it costs nothing when Postgres is disabled, and
tests can monkeypatch :func:`execute` / :func:`query` without a live database.
"""
from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Sequence, Tuple


def database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or None


def pg_enabled() -> bool:
    return bool(database_url())


def _dsn_with_ssl(dsn: str) -> str:
    # Managed Postgres requires TLS; enforce it unless the caller set a mode.
    if "sslmode=" in dsn:
        return dsn
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}sslmode=require"


_POOL = None
_POOL_LOCK = threading.Lock()


def get_pool():
    """Lazy singleton connection pool (autocommit) — one per process."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from psycopg_pool import ConnectionPool  # lazy

                _POOL = ConnectionPool(
                    _dsn_with_ssl(database_url()),
                    min_size=1,
                    max_size=4,
                    kwargs={"autocommit": True},
                )
    return _POOL


def execute(sql: str, params: Optional[dict] = None) -> None:
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params or {})


def query(sql: str, params: Optional[dict] = None) -> List[Tuple[Any, ...]]:
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params or {})
        return cur.fetchall()
