import pytest


@pytest.fixture(autouse=True)
def _no_live_db(monkeypatch):
    """Tests must never touch a live backend. A DATABASE_URL in the environment
    (or .env) would flip the durability stores onto Postgres; unset it so every
    test uses the file fallback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
