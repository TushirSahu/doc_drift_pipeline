# DocDrift — Production Feature Plan

This document proposes the features that turn DocDrift from a learning-oriented
RAG pipeline into a **production-ready documentation service**, explains *why*
each one matters, and *how* it fits the existing code.

The project today is excellent at *evaluating* RAG quality (Ragas + drift gate)
and *experimenting* with retrieval strategies. To run it in production you need
three things it doesn't have yet:

1. A **service interface** (so other systems can call it, not just a human at a CLI).
2. **Observability** (so you can see latency, cost, and bad answers in the wild).
3. **Robustness** (so a flaky model server or a hallucinated answer doesn't take you down).

The features below are grouped by priority. Tiers 1 and 2 are implemented in this
change; Tier 3 is the next-step backlog.

---

## Tier 1 — Make it a service (implemented)

### 1. REST API (`src/api/`)
**What:** A FastAPI app exposing `/health`, `/query`, `/ingest`, and `/metrics`,
protected by an API key. `/query` runs the agentic RAG loop and returns the
answer plus its trace; `/ingest` triggers document ingestion; `/metrics`
returns live observability stats.

**Why:** Right now the only way to use DocDrift is `python -m src.agentic.cli`.
A production tool needs a stable, language-agnostic interface so a web app, a
Slack bot, or another microservice can ask questions. The CLI stays for humans;
the API serves machines. This is "Week 4 — api/" on your roadmap.

**How it fits:** The API is a thin layer over the code you already have. `/query`
calls `AgenticController.run()`; `/ingest` calls `ingest_all()`; `/metrics` reads
the observability store. No business logic is duplicated.

### 2. Config validation (`src/core/schema.py`)
**What:** A Pydantic model that validates `config.yaml` at startup — types,
ranges (e.g. `0 ≤ mmr_lambda ≤ 1`), and enum values (e.g. chunking strategy).

**Why:** Today `cfg("retrieval", "top_k")` silently returns a default if a key is
missing or mistyped. In production a typo like `top_k: "five"` should fail loudly
*at boot*, not produce mysterious behavior three calls later. Fail fast, fail clear.

**How it fits:** Wraps the existing `load_config()`; the rest of the code can keep
using `cfg()`, but a validated `Settings` object is now available and validation
runs on startup.

### 3. Resilience: retry with backoff (`src/core/resilience.py`)
**What:** A `@retry` decorator with exponential backoff + jitter for calls that
hit the network (Ollama, Qdrant).

**Why:** Model servers and vector DBs throw transient errors — connection resets,
timeouts, cold starts. A single blip currently fails the whole request. Retrying
2–3 times with backoff turns most transient failures into invisible hiccups,
which is the single biggest reliability win for an LLM app.

**How it fits:** Decorate `LocalEmbedder.get_embeddings`, the Qdrant query calls,
and the `ollama.chat` calls. Pure-Python, no new dependency.

### 4. Caching (`src/core/cache.py`)
**What:** A TTL + LRU cache for embeddings (keyed by text hash) and for retrieval
results (keyed by query + params).

**Why:** Embeddings are deterministic — embedding the same chunk or the same
repeated user query twice is wasted latency and compute. In an eval loop or a
busy API, the same questions recur constantly. Caching cuts both response time
and load on Ollama/Qdrant. Cost and latency are the two metrics production users
feel most.

**How it fits:** `LocalEmbedder` and `RetrievalEngine` consult the cache before
calling out. Transparent — callers don't change.

---

## Tier 2 — See it and trust it (implemented)

### 5. Observability / tracing (`src/observability/`)
**What:** Every query produces a structured trace — request id, question,
latency, number of agent steps, tools called, retrieved chunk count, whether
guardrails passed — appended to `metrics/traces.jsonl`. A summarizer computes
p50/p95 latency, tool-usage frequency, and error rate.

**Why:** In production you cannot debug what you cannot see. When a user says "the
bot gave a wrong answer at 2pm," you need the trace: what it retrieved, how many
steps it took, which tools it called. Aggregate stats (p95 latency, error rate)
are what you'd put on a dashboard or alert on. This is "Week 4 — observability/."

**How it fits:** A lightweight `Tracer` context manager wraps
`AgenticController.run()` and the API handler. Writes JSONL (greppable, no DB
needed); `/metrics` and a CLI summarizer read it back.

### 6. Answer guardrails (`src/agentic/guardrails.py`)
**What:** Post-generation checks: (a) is the answer grounded — does it overlap
with the retrieved context, or is it likely hallucinated? (b) did the agent
include the required `[Source: ...]` citation? (c) is it an "I don't know" that
should be flagged rather than presented as fact?

**Why:** Your system prompt *asks* for citations and grounding, but nothing
*enforces* it — the LLM can ignore the rules. For a documentation tool, an
unsourced or hallucinated answer is worse than "I don't know." Guardrails give
you a measurable grounding score per answer and let the API flag low-confidence
responses instead of shipping them silently.

**How it fits:** Runs on the controller's final answer using the
`retrieved_contexts` it already collects. Adds `grounded`, `has_citation`, and
`grounding_score` to the result dict; the API can downgrade or warn on failures.

---

## Tier 3 — Next-step backlog (designed, not yet built)

### 7. Docker deployment (`Dockerfile`, `docker-compose.yml`)
**What:** One-command stack — app + Qdrant + Ollama — so the whole system runs
identically on a laptop or a server. *(Compose files are included in this change;
treat as Tier 2.5.)*

**Why:** "Works on my machine" is the enemy of production. Pinned, containerized
dependencies make deploys reproducible and onboarding instant.

### 8. Incremental / scheduled re-ingestion
**What:** A watcher or cron that re-ingests changed docs and re-runs the drift
eval automatically. You already hash files (`ingest_state.json`) — extend that to
trigger evals when docs change.

**Why:** "Documentation drift detection" should run *continuously*, not only when
someone pushes. This closes the loop on the project's core promise.

### 9. Prompt & model versioning + A/B eval
**What:** Track which prompt version (`prompts/v2/...`) and model produced each
eval, and add a `--compare-prompts` mode like the existing retriever/agentic
comparisons.

**Why:** In production you'll tune prompts and swap models. You need to prove a
change is an improvement, not a regression — your drift harness is the perfect
foundation.

### 10. Streaming responses + conversation memory
**What:** Server-sent-events streaming on `/query`, and optional multi-turn
session memory.

**Why:** Streaming makes the API feel responsive; memory enables follow-up
questions ("what about admin sessions?") without restating context.

### 11. Cost & token accounting
**What:** Record prompt/completion token counts per request and roll them into
the metrics summary.

**Why:** Even with local models, tokens map to latency and hardware cost. If you
later swap in a paid API, this becomes your billing guardrail.

---

## Summary

| # | Feature | Tier | New module |
|---|---------|------|------------|
| 1 | REST API | 1 | `src/api/` |
| 2 | Config validation | 1 | `src/core/schema.py` |
| 3 | Retry/backoff | 1 | `src/core/resilience.py` |
| 4 | Caching | 1 | `src/core/cache.py` |
| 5 | Observability/tracing | 2 | `src/observability/` |
| 6 | Answer guardrails | 2 | `src/agentic/guardrails.py` |
| 7 | Docker deployment | 2.5 | `Dockerfile`, `docker-compose.yml` |
| 8–11 | Re-ingestion, versioning, streaming, cost | 3 | backlog |

The throughline: DocDrift already *measures* quality offline. These features let
it *serve* answers online, *observe* itself, and *defend* against the two failure
modes that matter most in production — infrastructure flakiness and ungrounded
answers.
