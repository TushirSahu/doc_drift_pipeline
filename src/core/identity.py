"""
Document identity helpers.

Content hashing and doc-id derivation were duplicated across the ingestion
service, the vector store, and the automation layer — and they *must* agree, or
change detection silently breaks (a doc looks "new" because two modules derived
its id differently). Centralizing them here makes that contract explicit and
single-sourced.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.settings import ROOT_DIR


def content_hash(text: str) -> str:
    """Stable SHA-256 of a document's text. Identical text -> identical hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_for(path: str | Path, root: Path | None = None) -> str:
    """Derive the canonical doc id for a file.

    A path-relative id (``data/auth_service_v2.md`` -> ``data_auth_service_v2.md``)
    so it's stable, filesystem-safe, and human-readable in state files. Falls back
    to the bare filename for paths outside ``root``.
    """
    root = root or ROOT_DIR
    path = Path(path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.name
    return str(rel).replace("/", "_")
