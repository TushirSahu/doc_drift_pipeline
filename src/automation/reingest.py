"""
Automated re-ingestion + evaluation.

Why: "documentation drift detection" should run *continuously*, not only when a
human remembers to push. This module closes the loop:

    detect changed docs  ->  re-ingest only those  ->  re-run Ragas eval
                          ->  compare to baseline   ->  report drift

It powers two entry points (see ``cli.py``):
  * ``--once``  : a single check, ideal for cron or a CI schedule. Exits non-zero
                  if drift is detected, so it can gate a pipeline.
  * ``--watch`` : poll the data dir on an interval and react to changes locally.

Change detection reuses the same content hashing the vector store already uses
(``metrics/ingest_state.json``: ``{doc_id: sha256}``), so "changed" means exactly
"would be re-embedded." The heavy ingestion/eval imports are deferred into the
functions that need them, so detection (and the tests) stay dependency-free.
"""
from __future__ import annotations

import glob
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.core.identity import content_hash, doc_id_for
from src.core.settings import ROOT_DIR, cfg

logger = logging.getLogger(__name__)

# Same location the vector store writes to (see ingestion/vectorstore.py).
INGEST_STATE_FILE = ROOT_DIR / "metrics" / "ingest_state.json"


# --------------------------------------------------------------------------- #
# Change detection (pure, no external services)
# --------------------------------------------------------------------------- #
def _load_state(state_path: Path) -> Dict[str, str]:
    if not state_path.exists():
        return {}
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ChangeSet:
    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "added": self.added,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "removed": self.removed,
        }


def detect_changes(
    data_dir: Optional[str] = None,
    state_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> ChangeSet:
    """Compare the docs on disk against the recorded ingest state."""
    root = root or ROOT_DIR
    data_dir = data_dir or cfg("paths", "data_dir", default="data")
    state_path = state_path or INGEST_STATE_FILE

    state = _load_state(state_path)
    pattern = str(root / data_dir / "**" / "*.md")
    files = sorted(glob.glob(pattern, recursive=True))

    cs = ChangeSet()
    seen_ids = set()
    for fp in files:
        path = Path(fp)
        doc_id = doc_id_for(path, root=root)
        seen_ids.add(doc_id)
        digest = content_hash(path.read_text(encoding="utf-8"))
        if doc_id not in state:
            cs.added.append(doc_id)
        elif state[doc_id] != digest:
            cs.changed.append(doc_id)
        else:
            cs.unchanged.append(doc_id)

    # Docs we ingested before but that no longer exist on disk.
    cs.removed = [doc_id for doc_id in state if doc_id not in seen_ids]
    return cs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AutoReport:
    status: str                      # "no_changes" | "evaluated" | "ingest_failed"
    changes: Dict[str, List[str]]
    ingested_chunks: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    drift_passed: Optional[bool] = None
    drift_reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "changes": self.changes,
            "ingested_chunks": self.ingested_chunks,
            "scores": self.scores,
            "drift_passed": self.drift_passed,
            "drift_reasons": self.drift_reasons,
            "error": self.error,
        }


def _default_ingest() -> int:
    """Re-ingest all docs (the store skips unchanged files by hash)."""
    from src.ingestion.service import ingest_all

    return ingest_all()


def _default_evaluate() -> Tuple[Dict[str, float], bool, List[str]]:
    """Generate QA, score with Ragas, and compare against the drift baseline."""
    from src.evaluation import METRICS, RAGEvaluator, SyntheticDataGenerator
    from src.evaluation.drift import check_drift, load_baseline

    data_dir = cfg("paths", "data_dir", default="data")
    files = glob.glob(str(ROOT_DIR / data_dir / "*.md"))
    full_text = ""
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            full_text += f.read() + "\n"

    num_q = cfg("evaluation", "num_questions", default=5)
    qa_pairs = SyntheticDataGenerator().generate_qa_pairs(full_text, num_questions=num_q)
    questions = [p["question"] for p in qa_pairs]
    answers = [p["answer"] for p in qa_pairs]

    result = RAGEvaluator().run_evaluation(
        questions=questions, contexts=answers, answers=answers
    )
    df = result.to_pandas()
    scores = {m: float(df[m].mean()) for m in METRICS if m in df.columns}

    passed, reasons = check_drift(scores, load_baseline())
    return scores, passed, reasons


def run_auto_reingest(
    *,
    force: bool = False,
    data_dir: Optional[str] = None,
    ingest_fn: Callable[[], int] = _default_ingest,
    evaluate_fn: Callable[[], Tuple[Dict[str, float], bool, List[str]]] = _default_evaluate,
) -> AutoReport:
    """Detect changes, and if any exist (or ``force``), re-ingest then evaluate.

    ``ingest_fn`` and ``evaluate_fn`` are injectable so the orchestration can be
    tested without Ollama/Qdrant.
    """
    from src.observability.tracing import Tracer

    with Tracer("auto_reingest") as tracer:
        changes = detect_changes(data_dir=data_dir)
        tracer.update(changes=changes.to_dict())

        if not changes.has_work and not force:
            logger.info("No documentation changes detected — skipping eval.")
            return AutoReport(status="no_changes", changes=changes.to_dict())

        logger.info(
            "Changes detected (added=%d changed=%d removed=%d) — re-ingesting.",
            len(changes.added), len(changes.changed), len(changes.removed),
        )

        try:
            ingested = ingest_fn()
        except Exception as exc:  # noqa: BLE001
            logger.error("Re-ingestion failed: %s", exc)
            return AutoReport(
                status="ingest_failed", changes=changes.to_dict(), error=str(exc)
            )

        scores, passed, reasons = evaluate_fn()
        tracer.update(
            ingested_chunks=ingested,
            drift_passed=passed,
            grounded=passed,  # surface in /metrics grounding column
        )
        if not passed:
            for r in reasons:
                logger.error("DRIFT DETECTED: %s", r)

        return AutoReport(
            status="evaluated",
            changes=changes.to_dict(),
            ingested_chunks=ingested,
            scores=scores,
            drift_passed=passed,
            drift_reasons=reasons,
        )


def watch(
    interval: float = 30.0,
    max_iterations: Optional[int] = None,
    on_report: Optional[Callable[[AutoReport], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll the data dir every ``interval`` seconds and react to changes.

    Unlike ``--once``, this never exits on drift — it logs and keeps watching, so
    a long-running local watcher isn't killed by a single bad edit.
    ``max_iterations`` and ``sleep`` are injectable for tests.
    """
    logger.info("Watching for documentation changes every %.0fs (Ctrl-C to stop).", interval)
    iterations = 0
    while True:
        report = run_auto_reingest()
        if report.status != "no_changes":
            logger.info("Auto-reingest report: %s", report.to_dict())
        if on_report is not None:
            on_report(report)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return
        sleep(interval)
