"""Request/response schemas for the API — typed contracts for clients."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class GuardrailInfo(BaseModel):
    grounded: bool
    grounding_score: float
    has_citation: bool
    is_idk: bool
    reasons: List[str]


class QueryResponse(BaseModel):
    answer: str
    steps: int
    tools_used: List[str]
    retrieved_contexts: List[str]
    guardrails: GuardrailInfo
    warning: Optional[str] = None


class IngestResponse(BaseModel):
    ingested_chunks: int


class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, Any]


class MetricsResponse(BaseModel):
    traces: Dict[str, Any]
    embedding_cache: Dict[str, Any]
    retrieval_cache: Dict[str, Any]
