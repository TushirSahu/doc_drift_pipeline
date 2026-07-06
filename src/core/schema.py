"""
Config validation.

Why: ``cfg("retrieval", "top_k")`` silently returns a default if a key is
missing or mistyped. In production a typo like ``top_k: "five"`` should fail
loudly *at startup* — not produce confusing behavior several calls later.

This module defines a Pydantic schema for ``config.yaml`` and a
``validate_config`` helper. Call ``validate_config()`` once at boot (the API and
pipeline do this). The rest of the code can keep using ``cfg()`` as before.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.core.settings import get_config


class ModelEntry(BaseModel):
    """One candidate chat model in the multi-LLM benchmark registry.

    Self-contained on purpose: carrying its own provider + credentials is what
    lets an HF-router model and a local Ollama model be compared in a single run
    without fighting over the shared OPENAI_BASE_URL / OPENAI_API_KEY env vars.
    """

    model_config = {"extra": "forbid"}
    name: str
    provider: Literal["ollama", "openai"] = "openai"
    model: str
    base_url: Optional[str] = None      # openai-compatible host; None → OPENAI_BASE_URL
    api_key_env: Optional[str] = None   # env var holding the key; None → OPENAI_API_KEY


class ModelsCfg(BaseModel):
    model_config = {"extra": "allow"}  # base_url and other optional keys
    llm: str
    embed: str
    provider: Literal["ollama", "openai"] = "ollama"
    embed_provider: Literal["ollama", "openai", "sentence_transformers"] = "ollama"
    embed_dim: int = Field(default=768, gt=0)
    registry: List[ModelEntry] = Field(default_factory=list)
    primary_metric: str = "answer_correctness"


class ChunkingCfg(BaseModel):
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    strategy: Literal["markdown", "word"] = "markdown"


class RetrievalCfg(BaseModel):
    top_k: int = Field(gt=0)
    mmr_lambda: float = Field(ge=0.0, le=1.0)
    use_mmr: bool = False
    use_hybrid: bool = False
    hybrid_alpha: float = Field(ge=0.0, le=1.0)
    rerank: bool = False
    rerank_candidates: int = Field(gt=0)
    reranker: Literal["cross_encoder", "llm"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    multi_query: bool = False
    multi_query_count: int = Field(default=3, gt=0)


class EvaluationCfg(BaseModel):
    num_questions: int = Field(gt=0)
    metrics: List[str]
    timeout: int = Field(gt=0)
    max_workers: int = Field(gt=0)


class DriftCfg(BaseModel):
    faithfulness_threshold: float = Field(ge=0.0, le=1.0)
    regression_threshold: float = Field(ge=0.0, le=1.0)
    baseline_path: str


class AgenticCfg(BaseModel):
    max_steps: int = Field(gt=0)
    tools: List[str]


class Settings(BaseModel):
    """Validated view of config.yaml. Extra keys (paths, prompts) are allowed."""

    model_config = {"extra": "allow"}

    models: ModelsCfg
    chunking: ChunkingCfg
    retrieval: RetrievalCfg
    evaluation: EvaluationCfg
    drift: DriftCfg
    agentic: AgenticCfg


class ConfigError(Exception):
    """Raised when config.yaml fails validation."""


def validate_config(raw: dict | None = None) -> Settings:
    """Validate config and return a typed Settings object.

    Raises ConfigError with a readable message if validation fails.
    """
    data = raw if raw is not None else get_config()
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config.yaml:\n{exc}") from exc
