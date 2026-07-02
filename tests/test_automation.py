import json

from src.automation.reingest import (
    AutoReport,
    detect_changes,
    run_auto_reingest,
    watch,
)


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_detect_changes_classifies_files(tmp_path):
    root = tmp_path
    data = root / "data"
    _write(data / "a.md", "alpha")
    _write(data / "b.md", "bravo")

    # State knows a.md (matching) and an old c.md that no longer exists.
    import hashlib
    a_hash = hashlib.sha256("alpha".encode()).hexdigest()
    state = {"data_a.md": a_hash, "data_c.md": "deadbeef"}
    state_path = root / "ingest_state.json"
    state_path.write_text(json.dumps(state))

    cs = detect_changes(data_dir="data", state_path=state_path, root=root)
    assert cs.unchanged == ["data_a.md"]
    assert cs.added == ["data_b.md"]
    assert cs.removed == ["data_c.md"]
    assert cs.has_work is True


def test_detect_changes_marks_modified(tmp_path):
    root = tmp_path
    _write(root / "data" / "a.md", "new content")
    state_path = root / "state.json"
    state_path.write_text(json.dumps({"data_a.md": "oldhash"}))

    cs = detect_changes(data_dir="data", state_path=state_path, root=root)
    assert cs.changed == ["data_a.md"]


def test_run_auto_reingest_skips_when_no_changes(monkeypatch):
    import src.automation.reingest as mod
    monkeypatch.setattr(mod, "detect_changes", lambda **k: mod.ChangeSet())

    called = {"ingest": 0, "eval": 0}

    def ingest():
        called["ingest"] += 1
        return 0

    def evaluate():
        called["eval"] += 1
        return {}, True, []

    report = run_auto_reingest(ingest_fn=ingest, evaluate_fn=evaluate)
    assert report.status == "no_changes"
    assert called == {"ingest": 0, "eval": 0}  # nothing ran


def test_run_auto_reingest_force_runs_eval(monkeypatch):
    import src.automation.reingest as mod
    monkeypatch.setattr(mod, "detect_changes", lambda **k: mod.ChangeSet())

    report = run_auto_reingest(
        force=True,
        ingest_fn=lambda: 7,
        evaluate_fn=lambda: ({"faithfulness": 0.9}, True, []),
    )
    assert report.status == "evaluated"
    assert report.ingested_chunks == 7
    assert report.drift_passed is True


def test_run_auto_reingest_reports_drift(monkeypatch):
    import src.automation.reingest as mod
    cs = mod.ChangeSet(changed=["data_a.md"])
    monkeypatch.setattr(mod, "detect_changes", lambda **k: cs)

    report = run_auto_reingest(
        ingest_fn=lambda: 3,
        evaluate_fn=lambda: ({"faithfulness": 0.5}, False, ["faithfulness 0.5 below 0.8"]),
    )
    assert report.status == "evaluated"
    assert report.drift_passed is False
    assert report.drift_reasons


def test_run_auto_reingest_handles_ingest_failure(monkeypatch):
    import src.automation.reingest as mod
    monkeypatch.setattr(mod, "detect_changes", lambda **k: mod.ChangeSet(added=["x"]))

    def boom():
        raise RuntimeError("qdrant down")

    report = run_auto_reingest(ingest_fn=boom, evaluate_fn=lambda: ({}, True, []))
    assert report.status == "ingest_failed"
    assert "qdrant down" in report.error


def test_watch_stops_after_max_iterations(monkeypatch):
    import src.automation.reingest as mod
    monkeypatch.setattr(mod, "detect_changes", lambda **k: mod.ChangeSet())

    reports = []
    watch(
        interval=0,
        max_iterations=3,
        on_report=reports.append,
        sleep=lambda s: None,
    )
    assert len(reports) == 3
    assert all(isinstance(r, AutoReport) for r in reports)
