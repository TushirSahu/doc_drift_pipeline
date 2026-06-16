# DocDrift: Automated Documentation Drift Detection

DocDrift is an LLMOps platform for documentation question-answering. It ingests
markdown docs, serves grounded answers through an agentic RAG service, and
continuously evaluates answer quality with Ragas — failing the build when
documentation changes cause a measurable regression ("drift").

It combines two things most RAG projects keep separate: a **production serving
path** (API, caching, retries, guardrails, observability) and an **offline
quality gate** (synthetic QA generation, Ragas metrics, baseline drift checks in
CI).

## Architecture

```
doc_drift_pipeline/
├── config/config.yaml              # single source of truth for all knobs
├── prompts/                        # versioned agent system prompts
├── src/
│   ├── core/
│   │   ├── settings.py             # config loader
│   │   ├── schema.py               # config validation (fail fast at boot)
│   │   ├── resilience.py           # retry/backoff for Ollama + Qdrant
│   │   ├── cache.py                # TTL/LRU cache for embeddings + retrieval
│   │   └── prompts.py              # versioned prompt loader
│   ├── ingestion/                  # markdown chunking, embedding, vector store
│   ├── retrieval/                  # dense search + MMR, hybrid (BM25), rerank
│   ├── agentic/
│   │   ├── tools.py                # whitelisted tools (search_docs, calculator)
│   │   ├── controller.py           # agent loop: retrieve → reason → act → repeat
│   │   ├── guardrails.py           # grounding + citation checks on answers
│   │   └── cli.py                  # interactive Q&A
│   ├── observability/              # per-request tracing + aggregate stats
│   ├── evaluation/                 # QA generation, Ragas eval, drift gate
│   ├── automation/                 # change detection + auto re-ingest + re-eval
│   └── api/                        # FastAPI service (/health /query /ingest /metrics)
├── Dockerfile
├── docker-compose.yml              # one-command stack: api + qdrant + ollama
├── .github/workflows/              # CI: unit tests + drift-gated evaluation
├── FEATURE_PLAN.md                 # roadmap + design rationale
├── tests/
└── pipeline.py                     # evaluation + drift CLI
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python -m src.ingestion.cli --all
```

### Vector store modes

DocDrift talks to Qdrant in one of three modes, chosen by env var — no code change:

| Mode | How | When |
|------|-----|------|
| **Embedded on-disk** (default) | leave `QDRANT_URL` empty | local dev / demos — **no server or cloud needed** |
| **In-memory** | `QDRANT_PATH=:memory:` | quick tests (data not persisted) |
| **Remote / Cloud** | set `QDRANT_URL` (+ `QDRANT_API_KEY`) | production / shared deployments |

Embedded mode stores vectors under `qdrant_storage/` and requires nothing
running. You still need **Ollama** up (for embeddings + the LLM).

## Run as a service

```bash
# Local
uvicorn src.api.app:app --reload --port 8000

# Or the whole stack (api + qdrant + ollama)
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' -H 'X-API-Key: $DOCDRIFT_API_KEY' \
  -d '{"question": "What auth method does v2 use?"}'
curl localhost:8000/metrics -H 'X-API-Key: $DOCDRIFT_API_KEY'
```

Every `/query` returns the answer plus a **guardrail verdict** (is it grounded
in the retrieved docs? does it cite a source? grounding score) and is recorded
as a structured trace in `metrics/traces.jsonl`. `/metrics` aggregates p50/p95
latency, error rate, tool usage, and cache hit rates.

## Ask questions from the CLI

```bash
# Watch the agent search, reason, and answer
./scripts/run_rag.sh "What auth method does v2 use?"
python -m src.agentic.cli "How long is the admin session in minutes?"
```

## Evaluate quality and detect drift

```bash
# Generate synthetic QA, score with Ragas, fail on regression vs. baseline
python pipeline.py

# Set the current scores as the baseline
python pipeline.py --set-baseline

# Compare retrieval strategies side-by-side
python pipeline.py --compare-retrievers

# Compare naive single-shot RAG vs. agentic RAG on the same QA set
python pipeline.py --compare-agentic
```

The same evaluation runs in CI on every push that touches docs, code, or config
(`.github/workflows/`). If faithfulness drops below threshold or any metric
regresses past the baseline, the build fails — so documentation drift is caught
before it ships.

## Continuous re-ingestion

Drift detection shouldn't wait for a human to push. The automation layer detects
which docs changed (by content hash), re-ingests only those, re-runs the
evaluation, and reports drift:

```bash
# One-shot — re-ingest changed docs, re-eval, exit non-zero on drift.
# Wire into cron or a scheduled CI job.
python -m src.automation.cli --once

# Watch the data dir locally and react to edits (never exits on drift).
python -m src.automation.cli --watch --interval 60
```

Exit codes: `0` clean, `1` drift detected, `2` ingestion failed — so `--once`
doubles as a quality gate. A daily scheduled run ships in
`.github/workflows/scheduled-reingest.yml`.

## Agentic RAG vs. naive RAG

Naive RAG retrieves once and answers. The agentic controller lets the LLM decide
*what to do next* — it can search again with a refined query, run a calculator,
or combine multiple retrieval rounds before answering.

| | Naive RAG | Agentic RAG |
|--|-----------|-------------|
| **Flow** | retrieve once → answer | LLM decides each step |
| **Multi-hop** | can't re-search | re-queries with refined terms |
| **Tools** | none | calculator, search_docs, … |
| **Complex questions** | often fails | breaks the question into steps |
| **Control** | fixed pipeline | LLM is the controller |

Naive RAG is a **pipe**. Agentic RAG is an **agent** that uses retrieval as one
of its tools.

## Production engineering

| Capability | Why it matters |
|------------|----------------|
| **REST API** (`src/api/`) | Lets other systems use DocDrift, not just a human at a terminal |
| **Config validation** (`core/schema.py`) | Fails fast on a bad `config.yaml` instead of silently using wrong defaults |
| **Retry/backoff** (`core/resilience.py`) | Transient Ollama/Qdrant blips retry instead of failing the request |
| **Caching** (`core/cache.py`) | Deterministic embeddings + repeat queries served from memory — lower latency and cost |
| **Tracing** (`observability/`) | Debug any answer; alert on p95 latency / error rate |
| **Guardrails** (`agentic/guardrails.py`) | Flags ungrounded or uncited answers instead of shipping hallucinations |
| **Drift gate** (`evaluation/`) | CI blocks quality regressions caused by doc or prompt changes |
| **Docker** | Reproducible one-command deploys (api + qdrant + ollama) |

## Tech stack

Python · FastAPI · Qdrant (vector store) · Ollama (local LLM + embeddings) ·
Ragas (evaluation) · BM25 hybrid search · Pydantic · pytest · GitHub Actions ·
Docker Compose.
<!-- 
## Roadmap

See `FEATURE_PLAN.md` for the full design rationale. Next up: prompt/model A-B
evaluation, streaming responses with conversation memory, a real cross-encoder
reranker, and per-request token/cost accounting. -->
