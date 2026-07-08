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


class ModelsResponse(BaseModel):
    """Latest multi-LLM benchmark: who's serving + every model's scores."""

    champion: Optional[str] = None            # model name currently serving answers
    primary_metric: str                       # metric the champion was chosen by
    updated_at: Optional[str] = None          # when the benchmark last ran
    models: Dict[str, Dict[str, Any]] = {}    # name -> {metric: score} | {"error": ...}
    specs: Dict[str, Dict[str, str]] = {}     # name -> {provider, model} from config


class EvalResponse(BaseModel):
    """Latest single-model Ragas eval vs the committed drift baseline."""

    scores: Dict[str, float] = {}             # metric -> latest score
    baseline: Dict[str, float] = {}           # metric -> baseline score
    updated_at: Optional[str] = None          # when latest_eval was written


class BenchmarkStatusResponse(BaseModel):
    """State of the background multi-LLM benchmark job (--compare-models)."""

    state: Literal["idle", "running", "done", "error"]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None


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


class SourcesResponse(BaseModel):
    documents: List[str]
