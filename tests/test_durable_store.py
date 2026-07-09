"""Durable stores (Postgres-backed when DATABASE_URL is set).

No live database: pg.execute / pg.query are monkeypatched, so these assert the
backend-selection and SQL wiring without any real connection.
"""
import src.core.blob_store as blob
import src.core.pg as pg
import src.observability.stats as stats
import src.observability.tracing as tracing


# ── blob store ──────────────────────────────────────────────────────────────
def test_blob_file_roundtrip_without_pg(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "pg_enabled", lambda: False)
    monkeypatch.setattr(blob, "metrics_dir", lambda: tmp_path)
    blob.write_metrics_json("x.json", {"a": 1})
    assert blob.read_metrics_json("x.json") == {"a": 1}
    assert blob.read_metrics_json("missing.json") is None


def test_blob_prefers_pg_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "pg_enabled", lambda: True)
    monkeypatch.setattr(blob, "_READY", True)  # skip DDL
    monkeypatch.setattr(blob, "metrics_dir", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(pg, "execute", lambda sql, params=None: calls.append(sql))
    monkeypatch.setattr(pg, "query", lambda sql, params=None: [({"champion": "z"},)])

    blob.write_metrics_json("model_scores.json", {"champion": "z"})
    assert any("json_blobs" in s for s in calls)          # upserted to pg
    assert (tmp_path / "model_scores.json").exists()      # and still wrote the file
    assert blob.read_metrics_json("model_scores.json") == {"champion": "z"}  # pg wins


# ── trace store ─────────────────────────────────────────────────────────────
def test_trace_pg_write_and_load(monkeypatch):
    monkeypatch.setattr(pg, "pg_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_PG_READY", True)
    inserted = []
    monkeypatch.setattr(pg, "execute", lambda sql, params=None: inserted.append(params))
    tracing.record_trace({"trace_id": "t1", "operation": "q", "timestamp": "now",
                          "latency_ms": 5.0, "ok": True, "steps": 2})
    assert inserted and inserted[0]["trace_id"] == "t1"

    monkeypatch.setattr(
        pg, "query",
        lambda sql, params=None: [({"trace_id": "t1", "latency_ms": 5.0, "ok": True},)],
    )
    assert stats.load_traces() == [{"trace_id": "t1", "latency_ms": 5.0, "ok": True}]


def test_trace_explicit_path_bypasses_pg(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "pg_enabled", lambda: True)

    def boom(*a, **k):
        raise AssertionError("pg must not be used when an explicit path is given")

    monkeypatch.setattr(pg, "execute", boom)
    p = tmp_path / "traces.jsonl"
    tracing.record_trace({"trace_id": "t", "operation": "q"}, path=p)
    assert p.exists()


def test_trace_pg_failure_falls_back_to_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "pg_enabled", lambda: True)
    monkeypatch.setattr(tracing, "_PG_READY", True)

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(pg, "execute", boom)
    monkeypatch.setattr(tracing, "traces_path", lambda: tmp_path / "traces.jsonl")
    tracing.record_trace({"trace_id": "t", "operation": "q"})
    assert (tmp_path / "traces.jsonl").exists()  # request never loses its trace


def test_pg_enabled_reflects_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert pg.pg_enabled() is False
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    assert pg.pg_enabled() is True


def test_dsn_gets_sslmode(monkeypatch):
    assert "sslmode=require" in pg._dsn_with_ssl("postgresql://u:p@h/db")
    # respects an explicit mode
    assert pg._dsn_with_ssl("postgresql://u:p@h/db?sslmode=disable").endswith("disable")
