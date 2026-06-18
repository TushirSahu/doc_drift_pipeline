from src.core import llm


def test_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm.provider() == "ollama"  # config default


def test_provider_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    assert llm.provider() == "openai"  # case-insensitive


def test_chat_routes_to_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    captured = {}

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "hi"})})]

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _Resp()

    monkeypatch.setattr(llm, "_openai_client", lambda: _Client())
    out = llm.chat([{"role": "user", "content": "yo"}], model="gpt-x")
    assert out == "hi"
    assert captured["model"] == "gpt-x"


def test_embed_routes_to_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    class _Resp:
        data = [type("D", (), {"embedding": [0.1, 0.2, 0.3]})]

    class _Client:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                return _Resp()

    monkeypatch.setattr(llm, "_openai_client", lambda: _Client())
    assert llm.embed("text", model="emb") == [0.1, 0.2, 0.3]
