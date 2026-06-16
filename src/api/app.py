"""
FastAPI service layer for DocDrift.

Why: until now the only way to use DocDrift was the CLI. A production tool needs
a stable, language-agnostic interface so a web app, a Slack bot, or another
service can ask questions. This app is a thin layer over the existing code —
it duplicates no business logic.

Endpoints:
  GET  /health   — liveness + dependency checks (Qdrant, Ollama, config)
  POST /query    — run agentic RAG, return answer + trace + guardrail verdict
  POST /ingest   — (re)ingest documents from the data dir
  GET  /metrics  — aggregate observability stats + cache hit rates

Auth: all endpoints except /health require an `X-API-Key` header matching the
`DOCDRIFT_API_KEY` env var. If the var is unset, auth is disabled (dev mode).

Run:  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.agentic.controller import AgenticController
from src.api.models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from src.core.cache import embedding_cache, retrieval_cache
from src.core.logging import configure_logging
from src.core.schema import ConfigError, validate_config
from src.observability.stats import summarize_traces

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on a bad config at boot rather than at the first request.
    configure_logging()
    try:
        validate_config()
    except ConfigError as exc:
        logger.error("Config validation failed at startup: %s", exc)
        raise
    yield


app = FastAPI(
    title="DocDrift API",
    version="1.0.0",
    description="Agentic RAG over your documentation, with drift detection.",
    lifespan=lifespan,
)

# Allow the browser demo (and any frontend) to call the API. For a real
# deployment, set DOCDRIFT_CORS_ORIGINS to a comma-separated allowlist.
_cors = os.getenv("DOCDRIFT_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors == "*" else [o.strip() for o in _cors.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests without the configured API key. No key set = dev mode."""
    expected = os.getenv("DOCDRIFT_API_KEY")
    if not expected:
        return  # auth disabled in dev
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checks: dict = {}

    try:
        validate_config()
        checks["config"] = "ok"
    except ConfigError as exc:
        checks["config"] = f"invalid: {exc}"

    # Qdrant reachability (best-effort; never crash the health endpoint).
    try:
        from src.ingestion.vectorstore import get_vectorstore

        get_vectorstore().client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = f"unreachable: {type(exc).__name__}"

    healthy = all(v == "ok" for v in checks.values())
    return HealthResponse(status="ok" if healthy else "degraded", checks=checks)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    # Controller construction is cheap; the shared vector store handles connection
    # reuse. Constructing per request keeps the handler easy to test/mock.
    result = AgenticController().run(req.question)

    guard = result["guardrails"]
    warning = None
    if not guard["grounded"]:
        warning = "Answer may not be fully grounded in the documentation. " + \
                  "; ".join(guard["reasons"])

    return QueryResponse(
        answer=result["answer"],
        steps=result["steps"],
        tools_used=result["tools_used"],
        tool_calls=result.get("tool_calls", []),
        retrieved_contexts=result.get("retrieved_contexts", []),
        guardrails=guard,
        warning=warning,
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest() -> IngestResponse:
    from src.ingestion.service import ingest_all

    total = ingest_all()
    # New docs may change retrieval results — drop the stale cache.
    retrieval_cache.clear()
    return IngestResponse(ingested_chunks=total)


@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
def metrics() -> MetricsResponse:
    return MetricsResponse(
        traces=summarize_traces(),
        embedding_cache=embedding_cache.stats(),
        retrieval_cache=retrieval_cache.stats(),
    )


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    # Import here to avoid pulling the heavy evaluator package on every import.
    from src.evaluation.feedback import record_feedback

    entry = record_feedback(
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        trace_id=req.trace_id,
        correct_answer=req.correct_answer,
        comment=req.comment,
    )
    return FeedbackResponse(
        id=entry["id"],
        rating=entry["rating"],
        promoted_to_regression=entry["promoted_to_regression"],
    )
