"""
CLI for automated re-ingestion + evaluation.

Usage:
  # One-shot: re-ingest changed docs, re-eval, exit 1 if drift detected.
  # Ideal for cron or a scheduled CI job.
  python -m src.automation.cli --once

  # Force a full eval even if nothing changed.
  python -m src.automation.cli --once --force

  # Watch the data dir locally and react to edits (never exits on drift).
  python -m src.automation.cli --watch --interval 60

Cron example (every 15 min):
  */15 * * * * cd /app && python -m src.automation.cli --once >> logs/auto.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.automation.reingest import run_auto_reingest, watch
from src.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _print_report(report) -> None:
    print("\n" + "=" * 50)
    print("AUTO RE-INGEST REPORT")
    print("=" * 50)
    print(f"Status        : {report.status}")
    c = report.changes
    print(f"Changes       : +{len(c['added'])} added, "
          f"~{len(c['changed'])} changed, -{len(c['removed'])} removed")
    if report.status == "evaluated":
        print(f"Ingested      : {report.ingested_chunks} chunks")
        for metric, score in report.scores.items():
            print(f"  {metric.replace('_', ' ').title():<22}: {score * 100:.2f}%")
        print(f"Drift passed  : {report.drift_passed}")
        for reason in report.drift_reasons:
            print(f"    - {reason}")
    if report.error:
        print(f"Error         : {report.error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated documentation re-ingestion + drift evaluation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run a single check and exit")
    mode.add_argument("--watch", action="store_true", help="Poll continuously")
    parser.add_argument("--force", action="store_true",
                        help="Evaluate even if no docs changed")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between polls in --watch mode")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)

    if args.watch:
        watch(interval=args.interval)
        return 0

    report = run_auto_reingest(force=args.force)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)

    # Gate semantics: fail the process if drift was detected or ingestion failed.
    if report.status == "ingest_failed":
        return 2
    if report.drift_passed is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
