import argparse
import glob
import logging
import sys

from src.core.logging import configure_logging
from src.core.settings import ROOT_DIR, cfg
from src.evaluation import METRICS, RAGEvaluator, SyntheticDataGenerator, enforce_drift_or_exit
from src.ingestion import ingest_all

configure_logging()
logger = logging.getLogger(__name__)


def _collect_scores(df) -> dict:
    return {m: float(df[m].mean()) for m in METRICS if m in df.columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocDrift evaluation pipeline")
    parser.add_argument("--set-baseline", action="store_true",
                        help="Save current metrics as the drift baseline")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip the ingestion step")
    parser.add_argument("--compare-retrievers", action="store_true",
                        help="Run top_k=2 vs top_k=5 vs MMR side-by-side")
    parser.add_argument("--compare-agentic", action="store_true",
                        help="Compare naive RAG vs agentic RAG on same questions")
    parser.add_argument("--include-regressions", action="store_true",
                        help="Fold human-feedback regression cases into the QA set")
    parser.add_argument("--compare-providers", metavar="SPECS",
                        help='Eval across LLMs, e.g. "ollama=llama3.2:3b,openai=gpt-4o-mini"')
    parser.add_argument("--compare-models", action="store_true",
                        help="Benchmark every model in config models.registry, pick a champion")
    parser.add_argument("--dashboard", action="store_true",
                        help="(Re)build metrics/dashboard.html from the last model benchmark")
    args = parser.parse_args(argv)

    logger.info("Starting DocDrift pipeline...")

    # Rebuilding the dashboard from an existing benchmark needs no ingest/eval.
    if args.dashboard and not args.compare_models:
        from src.evaluation.dashboard import build_dashboard

        path = build_dashboard()
        if path is None:
            logger.error("No metrics/model_scores.json yet — run --compare-models first.")
            return 1
        print(f"Dashboard → {path}")
        return 0

    if not args.skip_ingest:
        total = ingest_all()
        logger.info("Ingested %d chunks", total)

    data_dir = cfg("paths", "data_dir", default="data")
    markdown_files = glob.glob(str(ROOT_DIR / data_dir / "*.md"))
    if not markdown_files:
        logger.error("No markdown files found in %s", data_dir)
        return 1

    full_text = ""
    for file_path in markdown_files:
        with open(file_path, encoding="utf-8") as f:
            full_text += f.read() + "\n"

    generator = SyntheticDataGenerator()
    num_q = cfg("evaluation", "num_questions", default=5)
    qa_pairs = generator.generate_qa_pairs(full_text, num_questions=num_q)
    logger.info("Generated %d QA pairs", len(qa_pairs))

    if not qa_pairs:
        logger.error("No QA pairs generated")
        return 1

    if args.include_regressions:
        from src.evaluation import regression_qa_pairs

        regressions = regression_qa_pairs()
        if regressions:
            logger.info("Adding %d regression case(s) from human feedback", len(regressions))
            qa_pairs.extend(regressions)

    questions = [p["question"] for p in qa_pairs]
    answers = [p["answer"] for p in qa_pairs]

    if args.compare_providers:
        import os
        from src.evaluation.export import export_json

        comparison: dict = {}
        for spec in [s.strip() for s in args.compare_providers.split(",") if s.strip()]:
            prov, _, model = spec.partition("=")
            os.environ["LLM_PROVIDER"] = prov.strip()
            os.environ["LLM_MODEL"] = model.strip() if model.strip() else ""
            logger.info("Evaluating %s ...", spec)
            try:
                df = RAGEvaluator().run_evaluation(
                    questions, answers, answers, export=False
                ).to_pandas()
                comparison[spec] = {m: float(df[m].mean()) for m in METRICS if m in df.columns}
            except Exception as e:  # noqa: BLE001
                logger.error("Provider %s failed: %s", spec, e)
                comparison[spec] = {"error": str(e)}
        export_json({"comparison": comparison}, "provider_comparison.json")
        print("\n" + "=" * 50 + "\nPROVIDER COMPARISON\n" + "=" * 50)
        for spec, scores in comparison.items():
            print(f"\n--- {spec} ---")
            for k, v in scores.items():
                print(f"  {k:<22}: {v * 100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")
        print("\nFull results → metrics/provider_comparison.json")
        return 0

    if args.compare_models:
        from src.core import llm
        from src.evaluation import benchmark_models

        specs = llm.registry()
        if not specs:
            logger.error("models.registry is empty — add candidate models to config.yaml.")
            return 1
        summary = benchmark_models(specs, questions, answers)
        print("\n" + "=" * 60)
        print("MULTI-LLM BENCHMARK  (primary metric: %s)" % summary["primary_metric"])
        print("=" * 60)
        for name, scores in summary["models"].items():
            crown = "  👑" if name == summary["champion"] else ""
            print(f"\n--- {name}{crown} ---")
            if "error" in scores:
                print(f"  ERROR: {scores['error']}")
                continue
            for metric, score in scores.items():
                print(f"  {metric.replace('_', ' ').title():<25}: {score * 100:.2f}%")
        print(f"\nChampion → {summary['champion']}  (serving path now uses it)")
        print("Scores → metrics/model_scores.json | Champion → metrics/champion.json")

        if args.dashboard:
            from src.evaluation.dashboard import build_dashboard

            path = build_dashboard()
            if path:
                print(f"Dashboard → {path}")
        return 0

    evaluator = RAGEvaluator()

    if args.compare_agentic:
        comparison = evaluator.compare_naive_vs_agentic(questions, answers, answers)
        print("\n" + "=" * 50)
        print("NAIVE vs AGENTIC RAG")
        print("=" * 50)
        for mode, scores in comparison.items():
            print(f"\n--- {mode} ---")
            for metric, score in scores.items():
                print(f"  {metric.replace('_', ' ').title():<25}: {score * 100:.2f}%")
        print("\nFull results → metrics/naive_vs_agentic.json")
        return 0

    if args.compare_retrievers:
        comparison = evaluator.compare_retrievers(questions, answers, answers)
        print("\n" + "=" * 50)
        print("RETRIEVER COMPARISON")
        print("=" * 50)
        for config_name, scores in comparison.items():
            print(f"\n--- {config_name} ---")
            for metric, score in scores.items():
                print(f"  {metric.replace('_', ' ').title():<25}: {score * 100:.2f}%")
        print("\nFull results → metrics/retriever_comparison.json")
        return 0

    results = evaluator.run_evaluation(
        questions=questions, contexts=answers, answers=answers,
    )

    df = results.to_pandas()
    scores = _collect_scores(df)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores.items():
        print(f"{metric.replace('_', ' ').title():<25}: {score * 100:.2f}%")
    print("\nDetailed results → metrics/latest_eval.csv")

    enforce_drift_or_exit(scores, set_baseline=args.set_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
