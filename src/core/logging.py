"""
Central logging configuration.

Every CLI was calling ``logging.basicConfig`` independently, which means
inconsistent formats and double-configuration when modules are composed. This
gives one entry point with a consistent, timestamped format and an env override
(``DOCDRIFT_LOG_LEVEL``) so log verbosity is operable in production without code
changes.
"""
from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging once. Safe to call from any entry point.

    Resolution order: explicit ``level`` arg, then ``DOCDRIFT_LOG_LEVEL`` env,
    then INFO.
    """
    resolved = level or os.getenv("DOCDRIFT_LOG_LEVEL") or logging.INFO
    if isinstance(resolved, str):
        resolved = logging.getLevelName(resolved.upper())
    logging.basicConfig(level=resolved, format=_DEFAULT_FORMAT)
