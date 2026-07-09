import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app as app_module  # noqa: E402
from src.api.app import app  # noqa: E402


FAKE_RESULT = {
    "answer": "Auth v2 uses OAuth2 and JWT tokens. [Source: auth_service_v2.md]",
    "steps": 2,
    "tools_used": ["search_docs"],
    "retrieved_contexts": ["Auth Service v2.0 uses OAuth2 and JWT tokens."],
    "guardrails": {
        "grounded": True,
        "grounding_score": 0.8,
        "has_citation": True,
        "is_idk": False,
        "reasons": ["passed"],
    },
}


class _FakeController:
    def __init__(self, *a, **k):
        pass

    def run(self, question):
        return FAKE_RESULT


def test_health_does_not_crash():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "config" in body["checks"]


def test_query_returns_answer_and_guardrails(monkeypatch):
    monkeypatch.setattr(app_module, "AgenticController", _FakeController)
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "What auth does v2 use?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "OAuth2" in body["answer"]
    assert body["guardrails"]["grounded"] is True
    assert body["warning"] is None


def test_query_requires_api_key_when_set(monkeypatch):
    monkeypatch.setattr(app_module, "AgenticController", _FakeController)
    monkeypatch.setenv("DOCDRIFT_API_KEY", "secret")
    with TestClient(app) as client:
        bad = client.post("/query", json={"question": "hi"})
        good = client.post(
            "/query", json={"question": "hi"}, headers={"X-API-Key": "secret"}
        )
    assert bad.status_code == 401
    assert good.status_code == 200


def test_query_rejects_empty_question(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic validation


def test_feedback_downvote_promoted(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)

    # Patch the storage layer so the test doesn't touch the real metrics dir.
    import src.evaluation.feedback as fb

    def fake_record(**kwargs):
        return {"id": "abc123", "rating": kwargs["rating"],
                "promoted_to_regression": kwargs["rating"] == "down"}

    monkeypatch.setattr(fb, "record_feedback", fake_record)
    with TestClient(app) as client:
        resp = client.post("/feedback", json={
            "question": "How long is the admin session?",
            "answer": "5 minutes",
            "rating": "down",
            "correct_answer": "12 hours",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["rating"] == "down"
    assert body["promoted_to_regression"] is True


def test_feedback_rejects_bad_rating(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.post("/feedback", json={
            "question": "q", "answer": "a", "rating": "maybe",
        })
    assert resp.status_code == 422  # Literal["up","down"] validation


def test_security_headers_present():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in resp.headers["Content-Security-Policy"]
    assert resp.headers["Cache-Control"] == "no-store"
    # Framework/version is not advertised.
    assert resp.headers["Server"] == "DocDrift"


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(app_module, "AgenticController", _FakeController)
    monkeypatch.setenv("DOCDRIFT_API_KEY", "secret")
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "hi"})  # no header
    assert resp.status_code == 401


def test_unhandled_error_is_opaque(monkeypatch):
    """A crash returns a generic message + correlation id — never a stack trace."""
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)

    class _BoomController:
        def __init__(self, *a, **k):
            pass

        def run(self, question):
            raise RuntimeError("secret path /Users/db/creds leaked here")

    monkeypatch.setattr(app_module, "AgenticController", _BoomController)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Internal server error"
    assert "error_id" in body
    assert "secret path" not in resp.text  # internals never reach the client


def test_query_rejected_when_saturated(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "AgenticController", _FakeController)

    class _FullSem:
        def acquire(self, blocking=False):
            return False

        def release(self):
            pass

    monkeypatch.setattr(app_module, "_QUERY_SEM", _FullSem())
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"


def test_feedback_rejects_oversized_answer(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.post("/feedback", json={
            "question": "q", "answer": "x" * 9000, "rating": "up",
        })
    assert resp.status_code == 422  # exceeds max_length


def test_env_flag_parses_truthy_values():
    from src.api.app import _env_flag
    import os as _os

    _os.environ["_DOCDRIFT_TEST_FLAG"] = "0"
    assert _env_flag("_DOCDRIFT_TEST_FLAG", True) is False
    _os.environ["_DOCDRIFT_TEST_FLAG"] = "on"
    assert _env_flag("_DOCDRIFT_TEST_FLAG", False) is True
    del _os.environ["_DOCDRIFT_TEST_FLAG"]
    assert _env_flag("_DOCDRIFT_TEST_FLAG", True) is True


def test_eval_returns_scores_and_baseline(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.get("/eval")
    assert resp.status_code == 200
    body = resp.json()
    # Shape is stable even when the metric files are missing/empty.
    assert isinstance(body["scores"], dict)
    assert isinstance(body["baseline"], dict)
    assert "updated_at" in body


def _cfg_with_gate(gate: bool):
    """cfg() stub: control the benchmark gate, leave everything else at default
    (so the rate limiter stays disabled)."""
    def fake(section, key, default=None):
        if (section, key) == ("api", "allow_benchmark_trigger"):
            return gate
        return default
    return fake


def test_benchmark_trigger_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    # Gate off (the default) → the endpoint refuses to spawn anything.
    monkeypatch.setattr(app_module, "cfg", _cfg_with_gate(False))
    with TestClient(app) as client:
        resp = client.post("/models/benchmark")
    assert resp.status_code == 403


def test_benchmark_start_and_double_start(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "cfg", _cfg_with_gate(True))
    # Don't actually launch a job — freeze it in the "running" state.
    monkeypatch.setattr(app_module, "_spawn", lambda fn: None)
    app_module._reset_benchmark_state()
    with TestClient(app) as client:
        first = client.post("/models/benchmark")
        second = client.post("/models/benchmark")
        status = client.get("/models/benchmark/status")
    assert first.status_code == 200
    assert first.json()["state"] == "running"
    assert second.status_code == 409          # a job is already running
    assert status.json()["state"] == "running"
    app_module._reset_benchmark_state()


def test_benchmark_records_done_on_success(monkeypatch):
    monkeypatch.delenv("DOCDRIFT_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "cfg", _cfg_with_gate(True))
    # Run the job body synchronously so the state transition is deterministic.
    monkeypatch.setattr(app_module, "_spawn", lambda fn: fn())

    class _FakeProc:
        def wait(self):
            return 0

    monkeypatch.setattr(app_module.subprocess, "Popen", lambda *a, **k: _FakeProc())
    app_module._reset_benchmark_state()
    with TestClient(app) as client:
        start = client.post("/models/benchmark")
        status = client.get("/models/benchmark/status")
    assert start.status_code == 200
    assert status.json()["state"] == "done"
    assert status.json()["returncode"] == 0
    app_module._reset_benchmark_state()
