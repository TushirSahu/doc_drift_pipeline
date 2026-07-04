# Edge-case audit & hardening issues

Whole-project edge-case review with severity, plus ready-to-open GitHub issues.
Sev: 🔴 critical · 🟠 high · 🟡 medium.

## Audit by component

### Ingestion & chunking
- 🟠 Deleted/renamed docs leave **orphan chunks** (no GC) → stale answers.
- 🟠 **Concurrent ingest + serve** on embedded Qdrant → file-lock crash.
- 🟡 Empty/whitespace or binary-as-`.md` file → 0 or garbage chunks, no validation.
- 🟡 Duplicate content across files → duplicate chunks inflate false confidence.
- 🟡 `ingest_state.json` partial write/corruption → skip logic misbehaves.

### Retrieval
- 🟠 `retrieval_cache` not cleared on **CLI/automation** ingest (only on API `/ingest`) → stale results.
- 🟡 Query longer than embed-model max tokens → silently truncated.
- 🟡 Look-alike docs (shared vocab) blend → mitigated by MMR/rerank but not eliminated.

### Agentic / tools
- ✅ **RESOLVED** `calculator` allows `**` → `9**9**9` DoS. Fixed: exponent capped
  at 100 and result magnitude at 1e100 (`tools.py`), with tests.
- 🟡 Client disconnect mid-stream → server generator keeps running (wasted compute).
- 🟡 Oversized tool args (huge `search_docs` query) → cost/latency.

### API
- ✅ **RESOLVED** Rate-limiter `_RL` was unbounded. Fixed: bounded LRU
  (`_RL_MAX_IPS`, evicts least-recently-used), with tests. Still per-process
  (multi-instance needs Redis — separate issue).
- 🟠 `/ingest` and `/query` are **synchronous with no timeout** → hung requests, worker starvation.
- 🟡 API key compared with `!=` → timing side-channel; use `secrets.compare_digest`.
- 🟡 Key accepted via `?key=` URL param → leaks into access logs.

### Evaluation & drift
- 🟠 Ragas metric returns **NaN** → `mean` is NaN → drift comparison silently wrong.
- 🟠 First run with no baseline **auto-saves** current as baseline → can enshrine a bad baseline.
- 🟡 Ragas/LLM-judge variance on tiny sets → flaky gate.

### Feedback loop (from prior review)
- 🔴 Regressions not auto-fed into automation eval; reference-less down-votes dropped.
- 🟠 Stale regression cases contradict updated docs → false drift.

### Observability
- 🟠 `traces.jsonl` / feedback JSONL grow **unbounded** → disk fill; `/metrics` reads whole file each call (O(n)).

### Security
- 🔴 **Prompt injection via documents** (a doc says "ignore instructions…") → no defense.
- 🟠 **Destructive-command answers** (DROP/rm -rf) served without a tripwire.

---

## Issues to open (paste-ready + gh)

Create labels once:
```bash
for l in "correctness:d93f0b" "security:b60205" "performance:fbca04" "reliability:0e8a16" "enhancement:a2eeef"; do
  gh label create "${l%%:*}" --color "${l##*:}" 2>/dev/null; done
```

```bash
gh issue create --title "Sandbox the calculator tool (block ** / huge exponents)" --label "security,correctness" \
  --body "calculator uses ast eval allowing **; 9**9**9 is a CPU/memory DoS. Cap operand/exponent size and reject ** above a threshold. File: src/agentic/tools.py."

gh issue create --title "Rate limiter leaks memory and is per-process" --label "reliability,performance" \
  --body "src/api/app.py _RL dict keeps one deque per IP forever (no eviction) and is per-process so it doesn't hold across instances. Add TTL eviction/max size, and a Redis backend option for multi-instance."

gh issue create --title "Add timeouts + async job for /ingest and /query" --label "reliability" \
  --body "Synchronous handlers with no timeout can hang workers. Add per-request timeouts on LLM/Qdrant calls and move /ingest to a background job with status polling."

gh issue create --title "Orphan-chunk GC + empty-collection UX" --label "correctness" \
  --body "Deleted/renamed docs leave orphan chunks; empty collections yield ungrounded answers. Add a GC that removes chunks whose source doc no longer exists, and a clear 'no docs ingested' state in API/demo."

gh issue create --title "Invalidate retrieval cache on CLI/automation ingest" --label "correctness" \
  --body "retrieval_cache is only cleared on API /ingest; CLI and automation ingest leave stale cached results. Clear the cache after any successful ingest."

gh issue create --title "Handle NaN metrics and missing-baseline in drift gate" --label "correctness" \
  --body "Ragas can return NaN → mean NaN → drift comparison silently wrong. Treat NaN as a failure; and don't silently enshrine the first run as baseline — require an explicit --set-baseline."

gh issue create --title "Rotate/cap traces.jsonl and feedback logs; stream /metrics" --label "performance,reliability" \
  --body "Append-only JSONL grows unbounded and /metrics reads the whole file per call. Add size-based rotation (or move to Postgres) and incremental aggregation."

gh issue create --title "Prompt-injection defense for retrieved docs" --label "security" \
  --body "A malicious/instructional doc chunk can hijack the answer. Add input isolation (delimit context, instruction to ignore embedded directives) and a post-answer check."

gh issue create --title "Destructive-answer tripwire" --label "safety,security" \
  --body "Flag answers containing irreversible ops (DROP, rm -rf, DELETE) in the guardrail verdict and require extra grounding/confirmation. Patterns in config.yaml."

gh issue create --title "Constant-time API key compare; stop accepting key via URL" --label "security" \
  --body "require_api_key uses != (timing side-channel) — use secrets.compare_digest. The demo passes ?key= which leaks into access logs; prefer header or one-time exchange."
```
