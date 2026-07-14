import pytest


@pytest.fixture(autouse=True)
def _no_live_db(monkeypatch):
    """Tests must never touch a live backend. A DATABASE_URL / REDIS_URL in the
    environment (or .env) would flip the durability stores onto Postgres or the
    rate limiter onto Redis; unset both so every test uses the local fallback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
