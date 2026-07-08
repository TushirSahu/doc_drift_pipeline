# Dynamic dashboard metrics + re-run benchmark button

**Date:** 2026-07-08
**Branch:** `feat/dynamic-metrics-rerun`

## Problem

1. Dashboard (`demo/index.html`) hardcodes which metrics it displays. New Ragas
   metrics added to `evaluation.metrics` never appear; the top "quality metrics"
   panel is a static array wired to no data.
2. No way to trigger a fresh multi-LLM benchmark without shelling into the box
   and running `python pipeline.py --compare-models`.

## Scope

Dashboard-only for the dynamic-metrics change (leave `demo/eval_dashboard.html`
untouched). Add one new feature: a re-run benchmark button.

## Part A — dynamic metrics (`demo/index.html`)

- **Model cards** (`renderModels`): stop iterating `Object.keys(METRIC_LABELS)`.
  Instead iterate the metric keys present on each model's own score object, so a
  new metric shows automatically. `METRIC_LABELS` stays as an *optional*
  pretty-label override; unknown keys fall back to `prettify(key)`
  (`answer_correctness` → `Answer correctness`).
- **Top quality panel** (`renderMetrics`): feed from real data instead of the
  static `METRICS` array. No file currently serves `baseline.json` /
  `latest_eval.json` over the API, so add a small read-only `GET /eval`. Panel
  computes each metric's value-vs-baseline from live data; the existing static
  array remains only as the offline demo fallback.

## Part B — re-run benchmark button

- **`POST /models/benchmark`** (api-key guarded): starts a background job that
  runs `subprocess.Popen([sys.executable, "pipeline.py", "--compare-models"])`.
  Fixed argv, no user input → no shell injection. Returns
  `{state, started_at}`. Second start while one runs → `409`.
- **`GET /models/benchmark/status`**: `{state: idle|running|done|error,
  started_at, finished_at, returncode, error}`. Dashboard polls ~3s while
  running, then calls `loadModels()` to refresh the board.
- **Safety gate**: `cfg("api", "allow_benchmark_trigger", default=False)`.
  Disabled → `403` with a hint to run the CLI locally. Prevents abuse on the
  public keyless Render demo. Enable locally via `config.yaml` or env.
- Job state is a module-global guarded by a `threading.Lock`; a watcher thread
  records the return code on exit. Single job per process (matches the existing
  single-process rate-limiter assumption).

## Registry — free lightweight models

Registry is already all free/lightweight. Add **one** more so the re-run has a
fuller board: `Qwen/Qwen2.5-3B-Instruct:cheapest` (HF router free tier). Keeps
CI cost and flakiness low.

## Files

- `demo/index.html` — dynamic metric rendering, re-run button, status poll.
- `src/api/app.py` — `/eval`, `/models/benchmark`, `/models/benchmark/status`.
- `src/api/models.py` — `EvalResponse`, `BenchmarkStatusResponse`.
- `config/config.yaml` — `api.allow_benchmark_trigger`, one registry entry.
- `tests/test_api.py` — endpoint tests (mock `subprocess.Popen`; gate on/off;
  409 on double-start; `/eval` shape). No live backend required.

## Non-goals

- No change to `eval_dashboard.html`.
- No queue / multi-job orchestration — one job per process is enough.
- No ingest trigger from the button (benchmark only; `--compare-models` runs
  without re-ingesting when the store is warm).

## Testing

- Unit: mock `subprocess.Popen`, assert start/gate/409/status transitions;
  assert `/eval` returns `{scores, baseline, updated_at}`.
- Manual: run API, click re-run with gate enabled, watch board refresh; add a
  metric to `evaluation.metrics`, confirm it appears without touching JS.
