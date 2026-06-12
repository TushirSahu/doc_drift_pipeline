"""
Resilience helpers — retry with exponential backoff + jitter.

Why: Ollama and Qdrant are network services. They throw transient errors —
connection resets, timeouts, cold starts. A single blip should not fail the
whole request. Retrying a couple of times with growing, jittered delays turns
most transient failures into invisible hiccups. This is the single biggest
reliability win for an LLM app, and it needs no extra dependency.

Usage:
    @retry(attempts=3, base_delay=0.5)
    def get_embeddings(...):
        ...
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a callable on the given exceptions with exponential backoff.

    The final attempt's exception is re-raised so callers still see real failures.
    ``sleep`` is injectable so tests don't actually wait.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad retry
                    last_exc = exc
                    if attempt == attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__, attempts, exc,
                        )
                        raise
                    sleep_for = min(delay, max_delay)
                    sleep_for += random.uniform(0, jitter * sleep_for)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                        fn.__name__, attempt, attempts, exc, sleep_for,
                    )
                    sleep(sleep_for)
                    delay *= backoff
            # Unreachable, but keeps type-checkers happy.
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
