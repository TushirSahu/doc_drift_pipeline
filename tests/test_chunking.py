from src.ingestion.chunking import _split_by_headers, chunk_text


def test_word_chunk_splits_long_document():
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_markdown_splits_on_headers():
    text = "# Auth Service\n\nUses OAuth2.\n\n## Endpoints\n\nPOST /api/v2/auth/refresh"
    sections = _split_by_headers(text)
    assert len(sections) >= 2
    headers = [h for h, _ in sections if h]
    assert any("Auth Service" in h for h in headers)


def test_markdown_chunk_keeps_section_together():
    text = "# Payment API\n\nAll transactions use Stripe webhooks."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert "Stripe" in chunks[0]
