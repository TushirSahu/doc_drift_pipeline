# Good first issues

Ready-to-open issues for open-source contributors. Create them from the GitHub UI,
or run the `gh` commands at the bottom (requires the GitHub CLI, `gh auth login`).

---

### 1. Add a dark-mode toggle to the demo UI
**Labels:** `good first issue`, `frontend`
The demo (`demo/index.html`) is a minimalist black-and-white theme. Add a toggle
that switches to a true-dark palette by swapping the `:root` CSS variables. Persist
the choice in `localStorage`. No build step — it's a single HTML file.
**Acceptance:** toggle flips light/dark, choice survives refresh, both modes readable.

### 2. Show retrieval scores next to each source chunk
**Labels:** `good first issue`, `backend`, `frontend`
`/query` returns `retrieved_contexts` but not their similarity scores. Include the
score per chunk in the API response and render it in the demo's source chips.
**Acceptance:** each source chip shows a 0–1 score; hidden gracefully when absent.

### 3. Add a `/warmup` endpoint + call it on demo load
**Labels:** `good first issue`, `backend`
Free hosts cold-start, so the first real query is slow. Add a lightweight
`/warmup` that embeds one token and pings the LLM, and have the demo call it on
load so the first user query is fast.
**Acceptance:** endpoint returns 200; demo calls it once on connect.

### 4. Support more vector stores (pgvector / Chroma)
**Labels:** `enhancement`
`vectorstore.py` is Qdrant-specific. Extract a small interface and add a second
backend (pgvector or Chroma) selected by config, mirroring the existing
embedded/remote modes.
**Acceptance:** ingestion + retrieval pass against the new backend behind a config flag.

### 5. Config-driven destructive-answer guardrail
**Labels:** `enhancement`, `safety`
Flag answers containing irreversible operations (`DROP`, `rm -rf`, `DELETE`) and
attach a warning in the guardrail verdict, so a RAG bot never silently suggests a
destructive command. Add patterns to `config.yaml`.
**Acceptance:** an answer with a destructive command is flagged; safe answers are not.

### 6. Add a CONTRIBUTING guide + dev setup
**Labels:** `good first issue`, `docs`
Document how to run locally (embedded Qdrant + Ollama), run tests, and the branch/
commit conventions. Link from the README.
**Acceptance:** a new contributor can go from clone to passing tests using only the guide.

---

## Create them with the GitHub CLI

```bash
gh label create "good first issue" --color 7057ff 2>/dev/null; true
gh issue create --title "Add a dark-mode toggle to the demo UI" --label "good first issue,frontend" --body "The demo (demo/index.html) is a minimalist B/W theme. Add a toggle that swaps the :root CSS variables to a dark palette and persists the choice in localStorage."
gh issue create --title "Show retrieval scores next to each source chunk" --label "good first issue,backend" --body "/query returns retrieved_contexts but not scores. Include the similarity score per chunk and render it in the demo source chips."
gh issue create --title "Add a /warmup endpoint and call it on demo load" --label "good first issue,backend" --body "Free hosts cold-start. Add a lightweight /warmup that embeds one token and pings the LLM; call it from the demo on connect so the first query is fast."
gh issue create --title "Support more vector stores (pgvector / Chroma)" --label "enhancement" --body "Extract a small interface from vectorstore.py and add a second backend selected by config, mirroring the embedded/remote modes."
gh issue create --title "Config-driven destructive-answer guardrail" --label "enhancement,safety" --body "Flag answers containing irreversible ops (DROP, rm -rf, DELETE) in the guardrail verdict. Patterns configurable in config.yaml."
gh issue create --title "Add a CONTRIBUTING guide and dev setup" --label "good first issue,documentation" --body "Document local run (embedded Qdrant + Ollama), tests, and branch/commit conventions. Link from the README."
```
