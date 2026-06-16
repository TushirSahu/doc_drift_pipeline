import glob
import logging
from pathlib import Path

from src.core.identity import doc_id_for
from src.core.settings import ROOT_DIR, cfg
from src.ingestion.vectorstore import CloudVectorStoreManager, get_vectorstore

logger = logging.getLogger(__name__)


def ingest_file(db: CloudVectorStoreManager, file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return 0

    text = path.read_text(encoding="utf-8")
    version = path.stem.split("_")[-1] if "_" in path.stem else "v1"

    return db.add_documents(
        doc_id=doc_id_for(path),
        text=text,
        metadata={"source": str(path), "version": version},
        skip_if_unchanged=True,
    )


def ingest_all(data_dir: str | None = None) -> int:
    data_dir = data_dir or cfg("paths", "data_dir", default="data")
    pattern = str(ROOT_DIR / data_dir / "**" / "*.md")
    files = glob.glob(pattern, recursive=True)
    if not files:
        logger.warning("No markdown files found in %s", data_dir)
        return 0

    db = get_vectorstore()
    total = 0
    for file_path in files:
        chunks = ingest_file(db, file_path)
        total += chunks
        logger.info("Ingested %s (%d chunks)", file_path, chunks)
    return total
