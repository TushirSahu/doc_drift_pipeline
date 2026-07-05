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

import os
from typing import Dict, List

from src.core.settings import cfg


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


def _openai_client():
    from openai import OpenAI  # lazy

    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL") or cfg("models", "base_url", default=None),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def chat_model() -> str:
    """Chat model id: LLM_MODEL env → models.llm config."""
    return os.getenv("LLM_MODEL") or cfg("models", "llm", default="llama3.2:3b")


def embed_model() -> str:
    """Embedding model id: EMBED_MODEL env → models.embed config."""
    return os.getenv("EMBED_MODEL") or cfg("models", "embed", default="nomic-embed-text")


def chat(messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.0) -> str:
    """Return the assistant's text for a chat completion."""
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
