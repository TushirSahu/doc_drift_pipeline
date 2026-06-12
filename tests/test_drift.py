from pathlib import Path

from src.evaluation.drift import check_drift, load_baseline, save_baseline


def test_check_drift_passes_within_threshold():
    baseline = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_precision": 0.75}
    current = {"faithfulness": 0.84, "answer_relevancy": 0.79, "context_precision": 0.74}
    passed, reasons = check_drift(current, baseline)
    assert passed is True
    assert reasons == []


def test_check_drift_fails_on_faithfulness():
    current = {"faithfulness": 0.5}
    passed, reasons = check_drift(current, baseline=None)
    assert passed is False


def test_check_drift_fails_on_regression():
    baseline = {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_precision": 0.80}
    current = {"faithfulness": 0.70, "answer_relevancy": 0.84, "context_precision": 0.79}
    passed, reasons = check_drift(current, baseline)
    assert passed is False
    assert any("faithfulness" in r for r in reasons)


def test_save_and_load_baseline(tmp_path: Path):
    path = tmp_path / "baseline.json"
    scores = {"faithfulness": 0.88, "answer_relevancy": 0.82, "context_precision": 0.77}
    save_baseline(scores, path=path)
    loaded = load_baseline(path=path)
    assert loaded["faithfulness"] == 0.88
