import re
from typing import List

from src.core.settings import cfg


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """
    Split documents into chunks for embedding.
    """
    chunk_size = chunk_size or cfg("chunking", "chunk_size", default=500)
    overlap = overlap or cfg("chunking", "overlap", default=50)
    strategy = cfg("chunking", "strategy", default="markdown")

    if strategy == "markdown":
        return _markdown_chunk(text, chunk_size, overlap)
    return _word_chunk(text, chunk_size, overlap)


def _word_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def _split_by_headers(text: str) -> List[tuple[str, str]]:
    """Break markdown into (header, body) sections."""
    pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    parts = pattern.split(text)
    if len(parts) == 1:
        return [("", text)]

    sections: List[tuple[str, str]] = []
    preamble = parts[0].strip()
    if preamble:
        sections.append(("", preamble))

    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2
    return sections


def _markdown_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    sections = _split_by_headers(text)
    chunks: List[str] = []

    for header, body in sections:
        body = body.strip()
        # Skip header-only sections (e.g. a "## Endpoints" parent whose content
        # lives in its sub-sections). They embed as near-empty chunks and pollute
        # retrieval with no information to ground an answer on.
        if header and not body:
            continue

        section_text = f"{header}\n{body}".strip() if header else body
        if not section_text:
            continue

        words = section_text.split()
        if len(words) <= chunk_size:
            chunks.append(section_text)
            continue

        # Section still too long — fall back to word chunks within this section
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + chunk_size])
            if piece.strip():
                chunks.append(piece)
            if i + chunk_size >= len(words):
                break

    return chunks or _word_chunk(text, chunk_size, overlap)
