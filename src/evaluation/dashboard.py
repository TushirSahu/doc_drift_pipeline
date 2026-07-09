"""
Render the multi-LLM benchmark as a self-contained HTML dashboard.

Why: ``metrics/model_scores.json`` is the source of truth, but a JSON blob is not
something you show a stakeholder (or a hiring panel). ``build_dashboard`` turns
the last benchmark into a single ``metrics/dashboard.html`` — a grouped bar chart
per metric, a sortable score table, and a badge for the champion that the serving
path is actually using. No server, no build step: open the file in a browser.

Chart.js is loaded from a CDN; everything else is inlined so the file is portable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.core.blob_store import read_metrics_json
from src.evaluation.drift import METRICS
from src.evaluation.export import metrics_dir

logger = logging.getLogger(__name__)


def _load_scores() -> Optional[dict]:
    return read_metrics_json("model_scores.json")


def build_dashboard(out_name: str = "dashboard.html") -> Optional[Path]:
    """Write metrics/<out_name> from model_scores.json. Returns path, or None."""
    data = _load_scores()
    if not data:
        return None

    models: dict = data.get("models", {})
    champion = data.get("champion")
    primary = data.get("primary_metric", "answer_correctness")
    generated = data.get("timestamp") or datetime.now(timezone.utc).isoformat()

    # Metrics actually present across the successful runs, in canonical order.
    ok = {n: s for n, s in models.items() if "error" not in s}
    metrics = [m for m in METRICS if any(m in s for s in ok.values())]
    names = list(models.keys())

    payload = {
        "labels": [m.replace("_", " ").title() for m in metrics],
        "metrics": metrics,
        "datasets": [
            {"name": n, "scores": [round(float(ok.get(n, {}).get(m, 0)) * 100, 2) for m in metrics]}
            for n in names if n in ok
        ],
        "champion": champion,
        "primary": primary,
    }

    # Table rows (include errored models so failures are visible).
    rows = []
    for n in names:
        s = models[n]
        crown = " 👑" if n == champion else ""
        if "error" in s:
            rows.append(
                f'<tr class="err"><td>{n}{crown}</td>'
                f'<td colspan="{len(metrics) + 1}">error: {s["error"]}</td></tr>'
            )
            continue
        cells = "".join(f"<td>{float(s.get(m, 0)) * 100:.1f}%</td>" for m in metrics)
        mean = sum(float(s.get(m, 0)) for m in metrics) / (len(metrics) or 1) * 100
        cls = ' class="champ"' if n == champion else ""
        rows.append(f"<tr{cls}><td>{n}{crown}</td>{cells}<td><b>{mean:.1f}%</b></td></tr>")

    header_cells = "".join(f"<th>{m.replace('_', ' ').title()}</th>" for m in metrics)
    champ_line = (
        f'Champion: <span class="badge">{champion} 👑</span> '
        f'(best <b>{primary.replace("_", " ")}</b>)'
        if champion else '<span class="badge warn">no successful model</span>'
    )

    html = _TEMPLATE.format(
        generated=generated,
        champ_line=champ_line,
        header_cells=header_cells,
        rows="\n".join(rows),
        data_json=json.dumps(payload),
    )
    out = metrics_dir() / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Dashboard → %s", out)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DocDrift — LLM Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{ --bg:#0f1117; --card:#1a1d27; --ink:#e7e9ee; --muted:#9aa3b2; --line:#2a2e3a; --accent:#6ea8fe; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
           padding:20px; margin-bottom:22px; }}
  .badge {{ background:#1f3a5f; color:#cfe1ff; padding:3px 10px; border-radius:999px;
            font-weight:600; font-size:13px; }}
  .badge.warn {{ background:#4a2b12; color:#ffd7a8; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  td:not(:first-child), th:not(:first-child) {{ text-align:right; }}
  tr.champ {{ background:rgba(110,168,254,.08); }}
  tr.err td {{ color:#ff9b9b; }}
  .foot {{ color:var(--muted); font-size:12px; text-align:center; margin-top:8px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>DocDrift — Multi-LLM Benchmark</h1>
  <div class="sub">{champ_line} &nbsp;·&nbsp; generated {generated}</div>

  <div class="card"><canvas id="chart" height="130"></canvas></div>

  <div class="card">
    <table>
      <thead><tr><th>Model</th>{header_cells}<th>Mean</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
  <div class="foot">Scores are Ragas metrics (higher is better). The judge model is held
  fixed across candidates; only the answer-generating model changes.</div>
</div>
<script>
  const DATA = {data_json};
  const palette = ["#6ea8fe","#7ee787","#f0883e","#d2a8ff","#ff7b72","#79c0ff","#e3b341"];
  new Chart(document.getElementById("chart"), {{
    type: "bar",
    data: {{
      labels: DATA.labels,
      datasets: DATA.datasets.map((d, i) => ({{
        label: d.name + (d.name === DATA.champion ? " 👑" : ""),
        data: d.scores,
        backgroundColor: palette[i % palette.length],
        borderRadius: 4,
      }})),
    }},
    options: {{
      responsive: true,
      scales: {{
        y: {{ beginAtZero:true, max:100, ticks:{{ color:"#9aa3b2", callback:v=>v+"%" }},
              grid:{{ color:"#2a2e3a" }} }},
        x: {{ ticks:{{ color:"#9aa3b2" }}, grid:{{ display:false }} }},
      }},
      plugins: {{ legend:{{ labels:{{ color:"#e7e9ee" }} }} }},
    }},
  }});
</script>
</body>
</html>
"""
