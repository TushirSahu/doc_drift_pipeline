# CLAUDE.md

Standing context for AI assistants working in this repo. Read this first.

## Project goal

Make DocDrift a production-ready system that solves a real problem and is
strong enough to showcase in interviews. Favor robustness, clear engineering
rationale, and clean git/docs over breadth of half-built features.

## What this project is

**DocDrift** is an LLMOps platform for documentation question-answering. It
ingests markdown docs, serves grounded answers through an agentic RAG service,
and continuously evaluates answer quality with Ragas — failing the build when a
doc/prompt/config change causes a measurable quality regression ("drift").

Two halves work together:
- **Serving path** — API, agentic retrieval, caching, retries, guardrails, tracing.
- **Quality gate** — synthetic QA generation, Ragas metrics, baseline drift checks in CI.

## Architecture (where things live)

- `src/core/` — `settings.py` (YAML config loader, use `cfg(...)`), `schema.py`
  (Pydantic config validation), `resilience.py` (retry/backoff), `cache.py`
  (TTL/LRU caches), `prompts.py` (versioned prompt loader).
- `src/ingestion/` — markdown chunking, embedding (Ollama), Qdrant vector store,
  ingest state tracking by content hash (`metrics/ingest_state.json`).
- `src/retrieval/` — dense search + MMR, hybrid (BM25) fusion, reranking
  (`reranker.py`: cross-encoder via `rerank()` dispatcher, or LLM fallback), all
  orchestrated by `engine.py`.
- `src/agentic/` — `controller.py` (the agent loop), `tools.py` (whitelisted
  tools: `search_docs`, `calculator`), `guardrails.py` (grounding + citation
  checks), `cli.py`.
- `src/evaluation/` — QA generation, Ragas `evaluator.py`, `drift.py` (baseline
  compare + gate).
- `src/observability/` — per-request JSONL tracing (`metrics/traces.jsonl`) +
  aggregate stats (p50/p95 latency, error rate, tool usage).
- `src/api/` — FastAPI app: `/health`, `/query`, `/ingest`, `/metrics`,
  `/feedback` (thumbs up/down; down-votes become regression cases).
- `src/automation/` — change detection + auto re-ingest + re-eval (`--once` for
  cron/CI, `--watch` for local).
- `pipeline.py` — top-level evaluation/drift CLI.
- `tests/` — pytest; `config/config.yaml` — all tunable knobs.

## Stack

Python 3.11 · FastAPI · Qdrant (vectors) · Ollama (local LLM `llama3.2:3b` +
`nomic-embed-text`) · Ragas (eval) · rank_bm25 · Pydantic · pytest ·
GitHub Actions · Docker Compose.

## Conventions

- **Config over constants.** New tunables go in `config/config.yaml` and are read
  via `cfg("section", "key", default=...)`. Add validation to `core/schema.py`.
- **External calls are wrapped.** Ollama/Qdrant calls use the `@retry` decorator
  and, where deterministic, the caches in `core/cache.py`.
- **Keep heavy imports lazy** in modules that need to stay testable. Ollama,
  Qdrant, Ragas, and datasets are not always installed in CI/dev sandboxes, so
  defer those imports into the functions that use them (see `automation/reingest.py`).
- **Testability via injection.** Orchestration functions accept injectable
  callables (e.g. `ingest_fn`, `evaluate_fn`) so tests don't need live backends.
- **Tracing & guardrails are not optional plumbing** — every served answer gets a
  guardrail verdict and a trace. Preserve this when editing the controller/API.
- Prefer prose-y, explanatory module docstrings that say *why* a module exists.

## Common commands

```bash
pip install -r requirements.txt
cp .env.example .env

python -m src.ingestion.cli --all            # ingest docs
python -m src.agentic.cli "..."              # ask a question
python pipeline.py                           # eval + drift gate
python pipeline.py --compare-agentic         # naive vs agentic
python -m src.automation.cli --once          # auto re-ingest + eval (cron/CI)
python -m src.automation.cli --watch         # local file watcher
uvicorn src.api.app:app --reload --port 8000 # run the service
docker compose up -d                         # full stack (api+qdrant+ollama)
pytest tests/ -v                             # tests
```

## Testing notes

- `pytest tests/` is the full suite. Many tests are pure-logic and need no
  backends; `test_api.py` needs `fastapi`+`httpx`, `test_config_schema.py` needs
  `pydantic`. All are in `requirements.txt`.
- Tests must not require a live Ollama/Qdrant. If a new test would, mock at the
  boundary or inject a fake callable instead.

## Git / workflow conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`...).
- Feature work goes on a `feat/<slug>` branch (git refs can't contain `:`, so the
  branch uses `/` while the commit message uses the `feat:` prefix).
- `metrics/` is gitignored except `metrics/baseline.json`, which is committed so
  the CI drift gate has a reference.
- Don't commit `.env`. Secrets (`QDRANT_URL`, `QDRANT_API_KEY`, `DOCDRIFT_API_KEY`)
  come from env / GitHub Actions secrets.

## CI

- `.github/workflows/llmops-eval.yml` — unit tests + drift-gated eval on push/PR
  touching docs/code/config.
- `.github/workflows/scheduled-reingest.yml` — daily (cron) auto re-ingest + drift
  check, also runnable on demand.
