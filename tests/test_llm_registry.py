"""Tests for the named-model registry + champion routing in core/llm.py.

Pure-logic: no LLM/vector backend. Config and filesystem are monkeypatched.
"""
import json

from src.core import llm
from src.core.llm import ModelSpec


def test_registry_parses_entries(monkeypatch):
    entries = [
        {"name": "a", "provider": "openai", "model": "m1",
         "base_url": "https://x/v1", "api_key_env": "HF_TOKEN"},
        {"name": "b", "provider": "ollama", "model": "llama3.2:3b"},
    ]
    monkeypatch.setattr(llm, "cfg", lambda *a, **k: entries)
    specs = llm.registry()
    assert [s.name for s in specs] == ["a", "b"]
    assert specs[0].base_url == "https://x/v1"
    assert specs[0].api_key_env == "HF_TOKEN"
    # Defaults applied to the minimal Ollama entry.
    assert specs[1].provider == "ollama"
    assert specs[1].api_key_env == "OPENAI_API_KEY"


def test_resolve_by_name(monkeypatch):
    monkeypatch.setattr(llm, "registry",
                        lambda: [ModelSpec("a", "openai", "m1"), ModelSpec("b", "ollama", "m2")])
    assert llm.resolve("b").model == "m2"


def test_resolve_unknown_raises(monkeypatch):
    monkeypatch.setattr(llm, "registry", lambda: [])
    try:
        llm.resolve("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_api_key_reads_named_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-123")
    spec = ModelSpec("a", "openai", "m1", api_key_env="HF_TOKEN")
    assert spec.api_key() == "secret-123"


def test_chat_with_spec_routes_to_that_endpoint(monkeypatch):
    captured = {}

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "spec-answer"})})]

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _Resp()

    seen = {}

    def _client(spec=None):
        seen["spec"] = spec
        return _Client()

    monkeypatch.setattr(llm, "_openai_client", _client)
    spec = ModelSpec("hf", "openai", "meta-llama/X", base_url="https://router/v1", api_key_env="HF_TOKEN")
    out = llm.chat([{"role": "user", "content": "hi"}], spec=spec)
    assert out == "spec-answer"
    assert captured["model"] == "meta-llama/X"   # spec.model used, not global
    assert seen["spec"] is spec                   # client built from the spec


def test_default_chat_spec_prefers_champion(monkeypatch):
    champion = {
        "name": "winner",
        "spec": {"name": "winner", "provider": "openai", "model": "best-model",
                 "base_url": "https://router/v1", "api_key_env": "HF_TOKEN"},
    }
    monkeypatch.setattr(llm, "read_metrics_json", lambda name: champion)
    spec = llm.default_chat_spec()
    assert spec.name == "winner"
    assert spec.model == "best-model"


def test_default_chat_spec_falls_back_to_config(monkeypatch):
    monkeypatch.setattr(llm, "read_metrics_json", lambda name: None)
    monkeypatch.setattr(llm, "provider", lambda: "ollama")
    monkeypatch.setattr(llm, "chat_model", lambda: "llama3.2:3b")
    spec = llm.default_chat_spec()
    assert spec.name == "configured"
    assert spec.provider == "ollama"
    assert spec.model == "llama3.2:3b"
