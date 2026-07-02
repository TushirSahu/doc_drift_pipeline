import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.identity import content_hash
from src.core.settings import ROOT_DIR
from src.ingestion.chunking import chunk_text
from src.ingestion.embedder import LocalEmbedder
from src.retrieval.engine import RetrievalEngine

load_dotenv()
logger = logging.getLogger(__name__)

INGEST_STATE_FILE = ROOT_DIR / "metrics" / "ingest_state.json"


class CloudVectorStoreManager:
    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "default_collection"
        )
        self.client = self._make_client()
        self.embedder = LocalEmbedder()
        self._corpus_cache: List[str] = []
        self._retriever = RetrievalEngine(self)
        self._ensure_collection_exists()

    @staticmethod
    def _make_client() -> QdrantClient:
        """Pick a Qdrant backend from env — remote, on-disk, or in-memory.

        - ``QDRANT_URL`` set  -> connect to a remote/cloud server (production).
        - ``QDRANT_PATH=:memory:`` -> ephemeral in-memory store (great for tests).
        - otherwise           -> embedded on-disk store, **no server needed**.
          Defaults to ``<repo>/qdrant_storage`` (override with ``QDRANT_PATH``).

        Embedded mode lets the whole project run with zero external infra, which
        is ideal for local dev and demos. (One process may open an on-disk store
        at a time; use a real server when running the API and pipeline together.)
        """
        url = (os.getenv("QDRANT_URL") or "").strip()
        path = (os.getenv("QDRANT_PATH") or "").strip()

        if url:
            return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        if path == ":memory:":
            logger.info("Using in-memory Qdrant (data is not persisted).")
            return QdrantClient(location=":memory:")

        store = path or str(ROOT_DIR / "qdrant_storage")
        logger.info("Using embedded Qdrant at %s (no server required).", store)
        return QdrantClient(path=store)

    def _ensure_collection_exists(self) -> None:
        want = self.embedder.dimensions
        names = {col.name for col in self.client.get_collections().collections}

        if self.collection_name in names:
            current = None
            try:
                current = self.client.get_collection(
                    self.collection_name
                ).config.params.vectors.size
            except Exception:  # noqa: BLE001 - introspection best-effort
                pass
            if current == want:
                return
            logger.warning(
                "Collection '%s' dim %s != embedder dim %s — recreating (drops old vectors).",
                self.collection_name, current, want,
            )
        else:
            logger.info("Creating collection '%s' (dim %s)", self.collection_name, want)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=want, distance=Distance.COSINE),
        )

    def _load_ingest_state(self) -> Dict[str, str]:
        if not INGEST_STATE_FILE.exists():
            return {}
        with open(INGEST_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)

    def _save_ingest_state(self, state: Dict[str, str]) -> None:
        INGEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INGEST_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def should_skip_ingest(self, doc_id: str, text: str) -> bool:
        return self._load_ingest_state().get(doc_id) == content_hash(text)

    def get_corpus_texts(self) -> List[str]:
        """Scroll all chunk texts from Qdrant (used by BM25 hybrid search)."""
        if self._corpus_cache:
            return self._corpus_cache
        texts: List[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for rec in records:
                if rec.payload and "text" in rec.payload:
                    texts.append(rec.payload["text"])
            if offset is None:
                break
        self._corpus_cache = texts
        return texts

    def list_documents(self) -> List[str]:
        """Distinct doc_ids actually present in the collection.

        Source of truth for "what's searchable" — works no matter where
        ingestion ran (unlike the local ingest-state file).
        """
        docs: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for rec in records:
                if rec.payload and rec.payload.get("doc_id"):
                    docs.add(rec.payload["doc_id"])
            if offset is None:
                break
        return sorted(docs)

    def add_documents(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_unchanged: bool = True,
    ) -> int:
        if skip_if_unchanged and self.should_skip_ingest(doc_id, text):
            logger.info("Skipping unchanged document '%s'", doc_id)
            return 0

        chunks = chunk_text(text)
        embeddings = self.embedder.get_embeddings_batch(chunks)
        now = datetime.now(timezone.utc).isoformat()
        points = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = dict(metadata or {})
            payload.update({
                "text": chunk,
                "doc_id": doc_id,
                "chunk_index": idx,
                "ingested_at": now,
            })
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %d chunks for '%s'", len(points), doc_id)

        state = self._load_ingest_state()
        state[doc_id] = content_hash(text)
        self._save_ingest_state(state)
        self._corpus_cache = []
        return len(chunks)

    def query_similarity(
        self,
        query: str,
        limit: int | None = None,
        use_mmr: bool | None = None,
        use_hybrid: bool | None = None,
        rerank: bool | None = None,
        model_name: str | None = None,
    ) -> List[str]:
        """Delegate to RetrievalEngine — keeps vectorstore focused on storage."""
        return self._retriever.retrieve(
            query,
            limit=limit,
            use_mmr=use_mmr,
            use_hybrid=use_hybrid,
            rerank=rerank,
            model_name=model_name,
        )


# Process-wide cache of vector store managers, keyed by collection name.
# Each manager opens a Qdrant client and builds an embedder, so constructing a
# fresh one per request/tool-call (as the code used to) wastes connections.
# Reuse one instead.
_MANAGERS: Dict[str, "CloudVectorStoreManager"] = {}


def get_vectorstore(collection_name: str | None = None) -> "CloudVectorStoreManager":
    """Return a shared CloudVectorStoreManager for the given collection."""
    key = collection_name or os.getenv("QDRANT_COLLECTION", "default_collection")
    manager = _MANAGERS.get(key)
    if manager is None:
        manager = CloudVectorStoreManager(collection_name=key)
        _MANAGERS[key] = manager
    return manager
