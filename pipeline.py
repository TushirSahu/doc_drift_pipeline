import argparse
import glob
import logging
import sys

from src.core.settings import ROOT_DIR, cfg
from src.evaluation import METRICS, RAGEvaluator, SyntheticDataGenerator, enforce_drift_or_exit
from src.ingestion import ingest_all

logging.basicConfig(level=logging.INFO)
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
    args = parser.parse_args(argv)

    logger.info("Starting DocDrift pipeline...")

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

    questions = [p["question"] for p in qa_pairs]
    answers = [p["answer"] for p in qa_pairs]
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
