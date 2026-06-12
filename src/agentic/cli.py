"""
Interactive CLI for agentic RAG.

Usage:
  python -m src.agentic.cli "What auth method does v2 use?"
  python -m src.agentic.cli "How many minutes is the admin session?" --json
"""
import argparse
import json
import logging
import sys

from src.agentic.controller import AgenticController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG — ask questions about your docs")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    args = parser.parse_args(argv)

    controller = AgenticController()
    result = controller.run(args.question)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n" + "=" * 50)
    print("AGENT TRACE")
    print("=" * 50)
    for call in result.get("tool_calls", []):
        print(f"  Step {call['step']}: {call['tool']}({call['args'][:60]})")
        print(f"    → {call['result_preview'][:120]}...")

    if result.get("retrieved_contexts"):
        print("\n" + "=" * 50)
        print("RETRIEVED CONTEXT")
        print("=" * 50)
        for i, ctx in enumerate(result["retrieved_contexts"], 1):
            print(f"\n--- Search {i} ---")
            print(ctx[:500])

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(result["answer"])
    print(f"\n(completed in {result['steps']} step(s), {len(result.get('tool_calls', []))} tool call(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
