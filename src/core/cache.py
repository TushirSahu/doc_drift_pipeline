"""
Lightweight in-process TTL + LRU cache.

Why: Embeddings are deterministic — embedding the same chunk or the same
repeated user query twice is wasted latency and compute. In an eval loop or a
busy API the same questions recur constantly. Caching cuts both response time
and load on Ollama/Qdrant, which are the two costs production users feel most.

This is intentionally dependency-free (no Redis). For a single-process service
it's plenty; swapping in a shared cache later only touches this file.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional


def make_key(*parts: Any) -> str:
    """Stable hash key from arbitrary parts (text, params, ...)."""
    raw = "\x1f".join(repr(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TTLCache:
    """Thread-safe cache with per-entry TTL and LRU eviction."""

    def __init__(self, maxsize: int = 512, ttl: float = 3600.0):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self.ttl = ttl
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _expired(self, ts: float) -> bool:
        return self.ttl > 0 and (time.monotonic() - ts) > self.ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, value = item
            if self._expired(ts):
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }


# Shared instances used across the app.
embedding_cache = TTLCache(maxsize=2048, ttl=86400.0)  # embeddings are stable
retrieval_cache = TTLCache(maxsize=512, ttl=600.0)      # results can go stale on re-ingest
