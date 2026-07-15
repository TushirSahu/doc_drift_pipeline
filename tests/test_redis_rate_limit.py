"""Redis-backed rate limiter (shared across instances) + in-process fallback."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api.app as app_module  # noqa: E402
from src.api.app import app  # noqa: E402
from src.core import redis_client  # noqa: E402


class _FakeRedis:
    """Minimal fixed-window counter store — enough for the limiter."""
    def __init__(self):
        self.counts = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, seconds):
        return True

    def ttl(self, key):
        return 42

    def eval(self, script, numkeys, key, arg):
        # Stand in for the INCR(+EXPIRE) Lua script: only the count matters here.
        return self.incr(key)


def _patch_limit(monkeypatch, value):
    real = app_module.cfg
    monkeypatch.setattr(
        app_module, "cfg",
        lambda *a, **k: value if a[:2] == ("api", "rate_limit_per_min") else real(*a, **k),
    )


def test_redis_enabled_reflects_env(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert redis_client.redis_enabled() is False
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert redis_client.redis_enabled() is True


def test_redis_rate_limited_blocks_past_limit(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    assert app_module._redis_rate_limited("1.2.3.4", 2)[0] is False  # 1
    assert app_module._redis_rate_limited("1.2.3.4", 2)[0] is False  # 2
    blocked, retry = app_module._redis_rate_limited("1.2.3.4", 2)    # 3 > 2
    assert blocked and retry >= 1


def test_middleware_uses_redis_when_enabled(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    _patch_limit(monkeypatch, 2)
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)
    fake = _FakeRedis()
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    app_module._RL.clear()
    with TestClient(app) as client:
        codes = [client.get("/metrics").status_code for _ in range(4)]
    assert 429 in codes
    assert len(app_module._RL) == 0  # counted in Redis, not the in-process store


def test_middleware_falls_back_when_redis_errors(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    _patch_limit(monkeypatch, 2)
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)

    def boom():
        raise RuntimeError("redis down")

    monkeypatch.setattr(redis_client, "get_redis", boom)
    app_module._RL.clear()
    with TestClient(app) as client:
        codes = [client.get("/metrics").status_code for _ in range(4)]
    assert 429 in codes                 # still limited via the in-process fallback
    assert len(app_module._RL) == 1     # fallback path did the counting


class _PingOK:
    def ping(self):
        return True


class _PingBad:
    def ping(self):
        raise RuntimeError("redis down")


def test_health_omits_redis_when_disabled(monkeypatch):
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: False)
    with TestClient(app) as client:
        checks = client.get("/health").json()["checks"]
    assert "redis" not in checks


def test_health_reports_redis_ok(monkeypatch):
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: _PingOK())
    with TestClient(app) as client:
        checks = client.get("/health").json()["checks"]
    assert checks["redis"] == "ok"


def test_health_redis_unreachable_is_degraded(monkeypatch):
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: _PingBad())
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["checks"]["redis"].startswith("unreachable")
    assert body["status"] == "degraded"


class _Req:
    def __init__(self, xff=None, host="9.9.9.9"):
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client = type("C", (), {"host": host})()


def test_client_ip_uses_socket_when_proxy_untrusted(monkeypatch):
    monkeypatch.setattr(app_module, "_TRUST_PROXY", False)
    # XFF is ignored — a naked deploy must not trust a spoofable header.
    assert app_module._client_ip(_Req(xff="1.1.1.1", host="9.9.9.9")) == "9.9.9.9"


def test_client_ip_uses_xff_rightmost_when_trusted(monkeypatch):
    monkeypatch.setattr(app_module, "_TRUST_PROXY", True)
    # Client prepended "evil"; the proxy appended the real 5.5.5.5 — take that.
    assert app_module._client_ip(_Req(xff="evil, 5.5.5.5", host="proxy")) == "5.5.5.5"


def test_client_ip_falls_back_without_xff(monkeypatch):
    monkeypatch.setattr(app_module, "_TRUST_PROXY", True)
    assert app_module._client_ip(_Req(xff=None, host="9.9.9.9")) == "9.9.9.9"
