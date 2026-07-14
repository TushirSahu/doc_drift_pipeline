"""
Optional Redis, for state that must be shared across app instances.

The in-process rate limiter is correct for one process, but behind a load
balancer with N instances the per-IP limit effectively becomes N× — a hole an
attacker spreading requests across instances walks through. When ``REDIS_URL``
is set, the limiter counts in Redis so the cap is global; when it is unset,
everything falls back to the in-process limiter (local dev/tests unchanged).

Mirrors ``pg.py``: the client is imported lazily so importing this costs nothing
when Redis is disabled, and tests monkeypatch :func:`get_redis`.
"""
from __future__ import annotations

import os
import threading
from typing import Optional


def redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL") or None


def redis_enabled() -> bool:
    return bool(redis_url())


_CLIENT = None
_LOCK = threading.Lock()


def get_redis():
    """Lazy singleton Redis client (one per process)."""
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                import redis  # lazy

                _CLIENT = redis.from_url(redis_url(), decode_responses=True)
    return _CLIENT
