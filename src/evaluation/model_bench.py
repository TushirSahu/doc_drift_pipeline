"""
Multi-LLM benchmark: score every candidate model and crown a champion.

Why: DocDrift can serve answers from any number of LLMs (HF-router models, a
local Ollama model, OpenAI, ...). Rather than pick one by gut feel, we run the
*same* synthetic QA set through each model listed in ``models.registry`` and let
Ragas score them. The best model on ``models.primary_metric`` is written to
``metrics/champion.json`` and becomes the model the serving path uses — "answer
based on the scores", made concrete.

Design notes:
- The Ragas *judge* is held fixed across all candidates (see RAGEvaluator); only
  the answer-*generating* model varies. That keeps scores comparable.
- ``benchmark_models`` takes an injectable ``evaluate_fn`` so tests can exercise
  champion selection and I/O without spinning up any LLM/vector backend.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from src.core import llm
from src.core.blob_store import write_metrics_json
from src.core.llm import ModelSpec
from src.core.settings import cfg
from src.evaluation.drift import METRICS
from src.evaluation.export import export_json, metrics_dir

logger = logging.getLogger(__name__)

# A per-model scorecard maps metric name -> score, or {"error": "..."} on failure.
Scores = Dict[str, float]
EvaluateFn = Callable[[ModelSpec], Scores]


def _default_evaluate_fn(questions: List[str], answers: List[str]) -> EvaluateFn:
    """Real scorer: run the RAG eval with `spec` as the answer generator."""

    def _run(spec: ModelSpec) -> Scores:
        # Imported lazily: RAGEvaluator pulls in Ragas/datasets/vectorstore, which
        # aren't installed in every sandbox. Keeping the import here lets the rest
        # of this module (and its tests) stay importable without them.
        from src.evaluation.evaluator import RAGEvaluator

        df = (
            RAGEvaluator(gen_spec=spec)
            .run_evaluation(questions, answers, answers, export=False)
            .to_pandas()
        )
        return {m: float(df[m].mean()) for m in METRICS if m in df.columns}

    return _run


def _mean(scores: Scores) -> float:
    vals = [v for k, v in scores.items() if k in METRICS and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def select_champion(
    results: Dict[str, Scores],
    primary_metric: str,
) -> Optional[str]:
    """Return the name of the best model.

    Ranks by ``primary_metric``; ties (or models missing that metric) fall back to
    the mean across all metrics. Models that errored are ignored. Returns None if
    nothing succeeded.
    """
    ranked = []
    for name, scores in results.items():
        if "error" in scores:
            continue
        primary = scores.get(primary_metric)
        primary = float(primary) if isinstance(primary, (int, float)) else -1.0
        ranked.append((primary, _mean(scores), name))
    if not ranked:
        return None
    ranked.sort(reverse=True)  # highest primary, then highest mean
    return ranked[0][2]


def _write_champion(spec: ModelSpec, primary_metric: str, score: Scores) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": spec.name,
        "primary_metric": primary_metric,
        "score": score.get(primary_metric),
        "spec": asdict(spec),
    }
    write_metrics_json("champion.json", payload)
    logger.info("Champion → %s (%s=%s)", spec.name, primary_metric, payload["score"])


def benchmark_models(
    specs: List[ModelSpec],
    questions: List[str],
    answers: List[str],
    primary_metric: str | None = None,
    evaluate_fn: EvaluateFn | None = None,
) -> dict:
    """Score every candidate model and record the champion.

    Writes ``metrics/model_scores.json`` (all scorecards) and, if any model
    succeeded, ``metrics/champion.json`` (the serving model). Returns a summary
    dict: ``{"models": {...}, "champion": name|None, "primary_metric": ...}``.
    """
    primary_metric = primary_metric or cfg("models", "primary_metric", default="answer_correctness")
    evaluate_fn = evaluate_fn or _default_evaluate_fn(questions, answers)
    by_name = {s.name: s for s in specs}

    results: Dict[str, Scores] = {}
    for spec in specs:
        logger.info("Benchmarking model '%s' (%s:%s)...", spec.name, spec.provider, spec.model)
        try:
            results[spec.name] = evaluate_fn(spec)
        except Exception as e:  # noqa: BLE001 - one bad model shouldn't sink the run
            logger.error("Model '%s' failed: %s", spec.name, e)
            results[spec.name] = {"error": str(e)}

    champion = select_champion(results, primary_metric)
    export_json(
        {"primary_metric": primary_metric, "champion": champion, "models": results},
        "model_scores.json",
    )
    if champion is not None:
        _write_champion(by_name[champion], primary_metric, results[champion])

    return {"models": results, "champion": champion, "primary_metric": primary_metric}
