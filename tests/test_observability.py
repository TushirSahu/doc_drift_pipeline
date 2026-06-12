from src.observability.stats import _percentile, summarize_traces
from src.observability.tracing import Tracer, record_trace


def test_percentile():
    data = [10, 20, 30, 40, 50]
    assert _percentile(data, 0.5) == 30
    assert _percentile([], 0.5) == 0.0


def test_tracer_writes_record(tmp_path):
    path = tmp_path / "traces.jsonl"
    with Tracer("query", path=path) as t:
        t.update(question="q", steps=2, tools_used=["search_docs"])
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"operation": "query"' in lines[0]
    assert '"latency_ms"' in lines[0]


def test_tracer_records_errors_without_suppressing(tmp_path):
    path = tmp_path / "traces.jsonl"
    raised = False
    try:
        with Tracer("query", path=path):
            raise RuntimeError("boom")
    except RuntimeError:
        raised = True
    assert raised  # exception propagated
    content = path.read_text()
    assert '"ok": false' in content
    assert "boom" in content


def test_summarize_traces(tmp_path):
    path = tmp_path / "traces.jsonl"
    for i in range(3):
        record_trace(
            {
                "operation": "query",
                "latency_ms": (i + 1) * 100,
                "ok": True,
                "steps": 2,
                "tools_used": ["search_docs"],
                "grounded": i != 0,
            },
            path=path,
        )
    summary = summarize_traces(path)
    assert summary["count"] == 3
    assert summary["error_rate"] == 0.0
    assert summary["tool_usage"]["search_docs"] == 3
    assert summary["avg_steps"] == 2.0
    assert summary["grounding_pass_rate"] == round(2 / 3, 3)


def test_summarize_empty(tmp_path):
    assert summarize_traces(tmp_path / "none.jsonl") == {"count": 0}
