"""Tests for the multi-LLM benchmark (evaluation/model_bench.py).

The scorer is injected, so these run without Ragas/Ollama/Qdrant. They cover the
two things that must be correct: champion selection and the written artifacts.
"""
import json

from src.core.llm import ModelSpec
from src.evaluation import model_bench
from src.evaluation.model_bench import benchmark_models, select_champion


def test_select_champion_by_primary_metric():
    results = {
        "a": {"answer_correctness": 0.60, "faithfulness": 0.99},
        "b": {"answer_correctness": 0.82, "faithfulness": 0.70},
        "c": {"answer_correctness": 0.75, "faithfulness": 0.90},
    }
    assert select_champion(results, "answer_correctness") == "b"


def test_select_champion_tie_breaks_on_mean():
    results = {
        "a": {"answer_correctness": 0.80, "faithfulness": 0.60},
        "b": {"answer_correctness": 0.80, "faithfulness": 0.95},  # same primary, higher mean
    }
    assert select_champion(results, "answer_correctness") == "b"


def test_select_champion_ignores_errored_models():
    results = {
        "a": {"error": "boom"},
        "b": {"answer_correctness": 0.5},
    }
    assert select_champion(results, "answer_correctness") == "b"


def test_select_champion_none_when_all_failed():
    assert select_champion({"a": {"error": "x"}}, "answer_correctness") is None


def test_benchmark_writes_scores_and_champion(monkeypatch, tmp_path):
    # Redirect all writes into a temp metrics dir.
    monkeypatch.setattr(model_bench, "metrics_dir", lambda: tmp_path)

    written = {}

    def _fake_export(data, filename):
        (tmp_path / filename).write_text(json.dumps(data), encoding="utf-8")
        written[filename] = data
        return tmp_path / filename

    monkeypatch.setattr(model_bench, "export_json", _fake_export)

    def _fake_write(name, payload):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        written[name] = payload
        return tmp_path / name

    monkeypatch.setattr(model_bench, "write_metrics_json", _fake_write)

    specs = [
        ModelSpec("weak", "ollama", "llama3.2:3b"),
        ModelSpec("strong", "openai", "big", base_url="https://r/v1", api_key_env="HF_TOKEN"),
    ]
    canned = {
        "weak": {"answer_correctness": 0.40, "faithfulness": 0.90},
        "strong": {"answer_correctness": 0.88, "faithfulness": 0.92},
    }
    summary = benchmark_models(
        specs, ["q1"], ["a1"],
        primary_metric="answer_correctness",
        evaluate_fn=lambda spec: canned[spec.name],
    )

    assert summary["champion"] == "strong"
    assert "model_scores.json" in written
    # Champion file carries the full spec so the serving path can rebuild it.
    champ = json.loads((tmp_path / "champion.json").read_text())
    assert champ["name"] == "strong"
    assert champ["spec"]["model"] == "big"
    assert champ["spec"]["api_key_env"] == "HF_TOKEN"


def test_benchmark_survives_one_failing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(model_bench, "metrics_dir", lambda: tmp_path)
    monkeypatch.setattr(model_bench, "export_json", lambda data, filename: tmp_path / filename)
    monkeypatch.setattr(model_bench, "write_metrics_json",
                        lambda name, payload: tmp_path / name)

    specs = [ModelSpec("bad", "openai", "x"), ModelSpec("good", "ollama", "y")]

    def _flaky(spec):
        if spec.name == "bad":
            raise RuntimeError("401 unauthorized")
        return {"answer_correctness": 0.7}

    summary = benchmark_models(specs, ["q"], ["a"],
                               primary_metric="answer_correctness", evaluate_fn=_flaky)
    assert summary["champion"] == "good"
    assert "error" in summary["models"]["bad"]
