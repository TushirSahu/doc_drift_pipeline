"""Request/response schemas for the API — typed contracts for clients."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    tool_calls: List[Dict[str, Any]] = []
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


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1)
    rating: Literal["up", "down"]
    trace_id: Optional[str] = None
    correct_answer: Optional[str] = None  # a correction turns this into a gold case
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    rating: str
    promoted_to_regression: bool
