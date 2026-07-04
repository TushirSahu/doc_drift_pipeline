import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api.app as app_module  # noqa: E402
from src.api.app import app  # noqa: E402


def _patch_limit(monkeypatch, value):
    real = app_module.cfg
    monkeypatch.setattr(
        app_module, "cfg",
        lambda *a, **k: value if a[:2] == ("api", "rate_limit_per_min") else real(*a, **k),
    )


def test_health_is_exempt_and_untracked(monkeypatch):
    _patch_limit(monkeypatch, 1)
    app_module._RL.clear()
    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/health").status_code == 200
    assert len(app_module._RL) == 0  # exempt path is never tracked


def test_rate_limit_triggers_and_stays_bounded(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    _patch_limit(monkeypatch, 2)
    app_module._RL.clear()
    with TestClient(app) as client:
        codes = [client.get("/metrics").status_code for _ in range(4)]
    assert 429 in codes                 # limiter kicks in past the limit
    assert len(app_module._RL) <= 1     # single client IP → memory stays bounded
