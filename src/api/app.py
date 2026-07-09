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

import hmac
import logging
import math
import os
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import json
import time
from collections import OrderedDict, deque

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.settings import ROOT_DIR, cfg

from src.agentic.controller import AgenticController
from src.api.models import (
    BenchmarkStatusResponse,
    EvalResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
    SourcesResponse,
)
from src.core.cache import answer_cache, embedding_cache, retrieval_cache
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


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Interactive API docs expose the full endpoint/schema surface. Fine for a demo,
# but a hardened deployment can hide them with DOCDRIFT_EXPOSE_DOCS=0.
_EXPOSE_DOCS = _env_flag("DOCDRIFT_EXPOSE_DOCS", True)

app = FastAPI(
    title="DocDrift API",
    version="1.0.0",
    description="Agentic RAG over your documentation, with drift detection.",
    lifespan=lifespan,
    docs_url="/docs" if _EXPOSE_DOCS else None,
    redoc_url="/redoc" if _EXPOSE_DOCS else None,
    openapi_url="/openapi.json" if _EXPOSE_DOCS else None,
)

# Reject requests with a spoofed/unexpected Host header (defeats Host-header
# poisoning and cache attacks). Default "*" keeps local dev + the demo working;
# set DOCDRIFT_ALLOWED_HOSTS to a comma-separated allowlist in production.
_hosts = os.getenv("DOCDRIFT_ALLOWED_HOSTS", "*")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if _hosts == "*" else [h.strip() for h in _hosts.split(",")],
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


# Sent on every response (including errors): content-type/framing hygiene,
# no-store on API JSON, and a generic Server header.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store",
}


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    if cfg("api", "security_headers", default=True):
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        # Don't advertise the server implementation/version.
        response.headers["Server"] = "DocDrift"
    return response

# Per-IP rate limit (fixed 60s window) so a keyless public demo is safe.
# Bounded LRU of IPs so memory can't grow without limit. Per-process; for
# multi-instance use a shared store (Redis). Set api.rate_limit_per_min (0 disables).
_RL_MAX_IPS = 10_000
_RL: "OrderedDict[str, deque]" = OrderedDict()

# Global ceiling on concurrent agent runs — each /query fans out to several LLM
# calls, so an unbounded flood exhausts memory/threads. Excess is shed with 503.
_MAX_CONCURRENT_QUERIES = max(1, int(cfg("api", "max_concurrent_queries", default=16)))
_QUERY_SEM = threading.BoundedSemaphore(_MAX_CONCURRENT_QUERIES)


def _acquire_query_slot() -> None:
    """Take a query slot or reject with 503 when the server is saturated."""
    if not _QUERY_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server busy — too many concurrent queries. Retry shortly.",
            headers={"Retry-After": "1"},
        )


@app.middleware("http")
async def _rate_limit(request, call_next):
    limit = cfg("api", "rate_limit_per_min", default=0)
    if limit and request.url.path != "/health":
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = _RL.get(ip)
        if hits is None:
            hits = deque()
            _RL[ip] = hits
        _RL.move_to_end(ip)  # mark recently used
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = max(1, int(60 - (now - hits[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded — try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        while len(_RL) > _RL_MAX_IPS:   # evict least-recently-used IPs
            _RL.popitem(last=False)
    return await call_next(request)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests without the configured API key. No key set = dev mode."""
    expected = os.getenv("DOCDRIFT_API_KEY")
    if not expected:
        return
    # Constant-time compare so timing can't recover the key byte by byte.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace, file path, or backend error to the client.

    The full error is logged server-side with a short id; the client gets only
    that id so an operator can correlate a report to the log without exposing
    internals to an attacker.
    """
    error_id = uuid.uuid4().hex[:12]
    logger.exception("Unhandled error %s on %s %s", error_id,
                     request.method, request.url.path)
    resp = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_id": error_id},
    )
    for k, v in _SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    return resp


# ── Background multi-LLM benchmark job ──────────────────────────────────────
# One benchmark runs at a time per process. State is a module global guarded by
# a lock; a watcher thread records the outcome when the subprocess exits. The
# `_spawn` seam lets tests run the job body synchronously (or not at all).
_BENCH_LOCK = threading.Lock()
_BENCH: dict = {
    "state": "idle", "started_at": None, "finished_at": None,
    "returncode": None, "error": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_benchmark_state() -> None:
    with _BENCH_LOCK:
        _BENCH.update(state="idle", started_at=None, finished_at=None,
                      returncode=None, error=None)


def _spawn(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def _run_benchmark_job() -> None:
    """Run `pipeline.py --compare-models` and record the outcome. Fixed argv —
    no user input reaches the command line, so there is nothing to inject."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "pipeline.py", "--compare-models"], cwd=str(ROOT_DIR)
        )
        rc = proc.wait()
        with _BENCH_LOCK:
            _BENCH["returncode"] = rc
            _BENCH["finished_at"] = _now()
            _BENCH["state"] = "done" if rc == 0 else "error"
            _BENCH["error"] = None if rc == 0 else f"pipeline exited with code {rc}"
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the thread
        with _BENCH_LOCK:
            _BENCH.update(state="error", finished_at=_now(), error=str(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checks: dict = {}

    try:
        validate_config()
        checks["config"] = "ok"
    except ConfigError as exc:
        # Detail can contain file paths — log it, expose only a generic status.
        logger.error("Config validation failed at /health: %s", exc)
        checks["config"] = "invalid"

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
    _acquire_query_slot()
    try:
        result = AgenticController().run(req.question)
    finally:
        _QUERY_SEM.release()

    guard = result["guardrails"]
    warning = None
    if result.get("blocked"):
        warning = "Request blocked by input guardrail (possible prompt injection)."
    elif not guard["grounded"]:
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
        cached=result.get("cached", False),
    )


@app.post("/query/stream", dependencies=[Depends(require_api_key)])
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Same as /query but streamed as Server-Sent Events (step → token → done)."""
    # Hold a concurrency slot for the whole stream; release when it ends.
    _acquire_query_slot()

    def events():
        try:
            for event in AgenticController().run_stream(req.question):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            _QUERY_SEM.release()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest() -> IngestResponse:
    from src.ingestion.service import ingest_all

    total = ingest_all()
    # New docs may change retrieval results and answers — drop the stale caches.
    retrieval_cache.clear()
    answer_cache.clear()
    return IngestResponse(ingested_chunks=total)


@app.get("/sources", response_model=SourcesResponse, dependencies=[Depends(require_api_key)])
def sources() -> SourcesResponse:
    """List documents actually present in the vector store (source of truth)."""
    from src.ingestion.vectorstore import get_vectorstore

    try:
        doc_ids = get_vectorstore().list_documents()
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint
        logger.warning("Could not list documents: %s", exc)
        doc_ids = []
    # Prettify: "data_auth_service_v2.md" -> "auth_service_v2.md"
    names = sorted({d[5:] if d.startswith("data_") else d for d in doc_ids})
    return SourcesResponse(documents=names)


@app.get("/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
def models() -> ModelsResponse:
    """The latest multi-LLM benchmark: the serving champion + every model's scores.

    Reads ``metrics/model_scores.json`` (written by ``pipeline.py --compare-models``).
    These numbers change only when the benchmark re-runs — not per request — so the
    page can poll this cheaply. ``specs`` is joined from config so the UI can show
    provider + model id next to each name.
    """
    from src.core import llm

    primary_default = cfg("models", "primary_metric", default="answer_correctness")
    specs = {s.name: {"provider": s.provider, "model": s.model} for s in llm.registry()}
    path = ROOT_DIR / cfg("paths", "metrics_dir", default="metrics") / "model_scores.json"
    if not path.exists():
        return ModelsResponse(champion=None, primary_metric=primary_default, specs=specs)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ModelsResponse(champion=None, primary_metric=primary_default, specs=specs)
    return ModelsResponse(
        champion=data.get("champion"),
        primary_metric=data.get("primary_metric", primary_default),
        updated_at=data.get("timestamp"),
        models=data.get("models", {}),
        specs=specs,
    )


def _finite_scores(path) -> dict:
    """Load a `{scores: {...}}` metrics file, keeping only finite float values.
    NaN/inf (e.g. Ragas failing a metric) are dropped so the JSON stays valid."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get("scores", {})
    except (ValueError, OSError):
        return {}
    out = {}
    for name, val in raw.items():
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[name] = f
    return out


@app.get("/eval", response_model=EvalResponse, dependencies=[Depends(require_api_key)])
def eval_scores() -> EvalResponse:
    """Latest single-model Ragas scores vs the committed drift baseline.

    Reads ``metrics/latest_eval.json`` and ``metrics/baseline.json`` (written by
    ``pipeline.py``). Lets the dashboard render the quality panel from live data
    instead of a hardcoded metric list.
    """
    metrics_dir = ROOT_DIR / cfg("paths", "metrics_dir", default="metrics")
    latest_path = metrics_dir / "latest_eval.json"
    updated_at = None
    if latest_path.exists():
        try:
            updated_at = json.loads(latest_path.read_text(encoding="utf-8")).get("timestamp")
        except (ValueError, OSError):
            updated_at = None
    return EvalResponse(
        scores=_finite_scores(latest_path),
        baseline=_finite_scores(metrics_dir / "baseline.json"),
        updated_at=updated_at,
    )


@app.post(
    "/models/benchmark",
    response_model=BenchmarkStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def start_benchmark() -> BenchmarkStatusResponse:
    """Kick off a fresh multi-LLM benchmark in the background.

    Gated behind ``api.allow_benchmark_trigger`` (default off) so a public,
    keyless demo can't be made to spawn expensive jobs. One job per process:
    a second start while one is running returns 409.
    """
    if not cfg("api", "allow_benchmark_trigger", default=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Benchmark trigger disabled. Run `python pipeline.py "
                   "--compare-models` locally, or set api.allow_benchmark_trigger.",
        )
    with _BENCH_LOCK:
        if _BENCH["state"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A benchmark is already running.",
            )
        _BENCH.update(state="running", started_at=_now(), finished_at=None,
                      returncode=None, error=None)
        snapshot = dict(_BENCH)
    _spawn(_run_benchmark_job)
    return BenchmarkStatusResponse(**snapshot)


@app.get(
    "/models/benchmark/status",
    response_model=BenchmarkStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def benchmark_status() -> BenchmarkStatusResponse:
    with _BENCH_LOCK:
        return BenchmarkStatusResponse(**_BENCH)


@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
def metrics() -> MetricsResponse:
    return MetricsResponse(
        traces=summarize_traces(),
        embedding_cache=embedding_cache.stats(),
        retrieval_cache=retrieval_cache.stats(),
        answer_cache=answer_cache.stats(),
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
