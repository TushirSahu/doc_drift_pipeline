"""
Provider-agnostic model access (chat + embeddings).

Why: the local stack uses Ollama, but free cloud hosts have no GPU. Routing the
serving-path model calls through one tiny interface lets the *same* code run
locally on Ollama or in the cloud on any OpenAI-compatible endpoint (OpenAI,
Groq, Together, a local vLLM) — a config/env change, not a rewrite.

Selected by ``models.provider`` (or the ``LLM_PROVIDER`` env override). Heavy
SDKs are imported lazily so this module stays importable without them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.blob_store import read_metrics_json
from src.core.settings import ROOT_DIR, cfg


@dataclass(frozen=True)
class ModelSpec:
    """A fully-resolved chat model: everything needed to talk to it in one place.

    The serving path historically read a *single* global provider/model. To
    compare several LLMs (and route to the winner) each candidate must carry its
    own provider + endpoint + credentials, otherwise two "openai" models would
    fight over the shared OPENAI_BASE_URL / OPENAI_API_KEY env. A ModelSpec makes
    a model self-describing so ``chat(spec=...)`` needs no globals.
    """

    name: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"

    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or cfg("models", "provider", default="ollama")).lower()


def embed_provider() -> str:
    # Decoupled from the chat provider: HF's router is chat-only, so a cloud
    # deploy can use HF for chat + local sentence-transformers for embeddings.
    return (os.getenv("EMBED_PROVIDER") or cfg("models", "embed_provider", default="ollama")).lower()


_ST_MODELS: dict = {}


def _sentence_transformer(model: str):
    st = _ST_MODELS.get(model)
    if st is None:
        from sentence_transformers import SentenceTransformer  # lazy

        st = SentenceTransformer(model)
        _ST_MODELS[model] = st
    return st


def _openai_client(spec: ModelSpec | None = None):
    from openai import OpenAI  # lazy

    if spec is not None:
        base_url = spec.base_url or os.getenv("OPENAI_BASE_URL") or cfg("models", "base_url", default=None)
        api_key = spec.api_key()
    else:
        base_url = os.getenv("OPENAI_BASE_URL") or cfg("models", "base_url", default=None)
        api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key)


# --- Named-model registry (multi-LLM benchmark + champion routing) -----------
def registry() -> List[ModelSpec]:
    """Candidate chat models from ``models.registry`` in config, in file order."""
    entries = cfg("models", "registry", default=[]) or []
    specs: List[ModelSpec] = []
    for e in entries:
        specs.append(
            ModelSpec(
                name=e["name"],
                provider=str(e.get("provider", "openai")).lower(),
                model=e["model"],
                base_url=e.get("base_url"),
                api_key_env=e.get("api_key_env") or "OPENAI_API_KEY",
            )
        )
    return specs


def resolve(name: str) -> ModelSpec:
    """Look up a registry model by name (raises if it isn't defined)."""
    for spec in registry():
        if spec.name == name:
            return spec
    raise KeyError(f"model '{name}' not in models.registry")


def _legacy_spec() -> ModelSpec:
    """The single configured model, as a spec (back-compat default)."""
    return ModelSpec(
        name="configured",
        provider=provider(),
        model=chat_model(),
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    )


def champion_spec() -> ModelSpec | None:
    """The benchmark-winning model, if a champion has been recorded."""
    data = read_metrics_json("champion.json")
    if not data:
        return None
    try:
        s = data["spec"]
        return ModelSpec(
            name=s["name"],
            provider=str(s.get("provider", "openai")).lower(),
            model=s["model"],
            base_url=s.get("base_url"),
            api_key_env=s.get("api_key_env") or "OPENAI_API_KEY",
        )
    except (KeyError, ValueError, TypeError):
        return None


def default_chat_spec() -> ModelSpec:
    """Model the serving path should use: the champion if one exists, else config."""
    return champion_spec() or _legacy_spec()


def chat_model() -> str:
    """Chat model id: LLM_MODEL env → models.llm config."""
    return os.getenv("LLM_MODEL") or cfg("models", "llm", default="llama3.2:3b")


def embed_model() -> str:
    """Embedding model id: EMBED_MODEL env → models.embed config."""
    return os.getenv("EMBED_MODEL") or cfg("models", "embed", default="nomic-embed-text")


def chat(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    spec: ModelSpec | None = None,
) -> str:
    """Return the assistant's text for a chat completion.

    ``spec`` selects a self-contained model (provider + endpoint + key); when it
    is given ``model`` is ignored. Without a spec the call keeps the original
    behavior (global provider + ``model``/config), so existing callers are
    unaffected.
    """
    if spec is not None:
        if spec.provider == "openai":
            resp = _openai_client(spec).chat.completions.create(
                model=spec.model, messages=messages, temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        from ollama import chat as ollama_chat  # lazy

        resp = ollama_chat(model=spec.model, messages=messages, options={"temperature": temperature})
        return resp.message.content

    model = model or chat_model()
    if provider() == "openai":
        resp = _openai_client().chat.completions.create(
            model=model, messages=messages, temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    from ollama import chat as ollama_chat  # lazy

    resp = ollama_chat(model=model, messages=messages, options={"temperature": temperature})
    return resp.message.content


def embed(text: str, model: str | None = None) -> List[float]:
    """Return the embedding vector for a piece of text."""
    model = model or embed_model()
    p = embed_provider()
    if p in ("sentence_transformers", "st", "local"):
        # Runs locally on CPU (no embedding API) — ideal for GPU-free hosts.
        return _sentence_transformer(model).encode(text, normalize_embeddings=True).tolist()
    if p == "openai":
        resp = _openai_client().embeddings.create(model=model, input=text)
        return resp.data[0].embedding
    from ollama import embeddings as ollama_embeddings  # lazy

    return ollama_embeddings(model=model, prompt=text)["embedding"]


# --- LangChain objects for Ragas (the evaluator needs LC-shaped LLM/embeddings) ---
def eval_llm(temperature: float = 0.0):
    """A LangChain chat model for Ragas, matching the configured provider."""
    model = chat_model()
    if provider() == "openai":
        from langchain_openai import ChatOpenAI  # lazy

        return ChatOpenAI(
            model=model, temperature=temperature,
            base_url=os.getenv("OPENAI_BASE_URL") or cfg("models", "base_url", default=None),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    from langchain_ollama import ChatOllama  # lazy

    return ChatOllama(model=model, temperature=temperature)


def eval_embeddings():
    """A LangChain embeddings object for Ragas, matching the embed provider."""
    model = embed_model()
    p = embed_provider()
    if p in ("sentence_transformers", "st", "local"):
        from langchain_community.embeddings import HuggingFaceEmbeddings  # lazy

        return HuggingFaceEmbeddings(model_name=model)
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings  # lazy

        return OpenAIEmbeddings(
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL") or cfg("models", "base_url", default=None),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    from langchain_ollama import OllamaEmbeddings  # lazy

    return OllamaEmbeddings(model=model)
