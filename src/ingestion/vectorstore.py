import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.settings import ROOT_DIR, cfg
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
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.embedder = LocalEmbedder()
        self._corpus_cache: List[str] = []
        self._retriever = RetrievalEngine(self)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        if not exists:
            logger.info("Creating collection '%s'", self.collection_name)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )

    def _file_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

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
        return self._load_ingest_state().get(doc_id) == self._file_hash(text)

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
        state[doc_id] = self._file_hash(text)
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
