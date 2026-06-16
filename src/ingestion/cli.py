import argparse
import logging
import sys

from src.core.logging import configure_logging
from src.ingestion.service import ingest_all, ingest_file
from src.ingestion.vectorstore import get_vectorstore

configure_logging()
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest markdown docs into Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single markdown file")
    group.add_argument("--all", action="store_true", help="Ingest all markdown under data/")
    args = parser.parse_args(argv)

    if args.all:
        total = ingest_all()
        logger.info("Total chunks ingested: %d", total)
        return 0

    chunks = ingest_file(get_vectorstore(), args.file)
    logger.info("Ingested %d chunks from %s", chunks, args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
