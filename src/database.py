import os
import uuid
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from .embedder import LocalEmbedder

load_dotenv()
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudVectorStoreManager:
    def __init__(self, collection_name: str = "default_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embedder = LocalEmbedder()
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        if not exists:
            logger.info(f"Creating collection '{self.collection_name}' in Qdrant")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedder.dimensions, 
                distance=Distance.COSINE)
            )

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def add_documents(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        chunks = self.chunk_text(text)
        embeddings = self.embedder.get_embeddings_batch(chunks)
        points = []
        for idx, (chunk,embedding) in enumerate(zip(chunks, embeddings)):
            payload = metadata or {}
            payload["text"] = chunk
            points.append(PointStruct(
                id  = str(uuid.uuid4()),
                vector = embedding,
                payload = payload
            ))
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points for document ID '{doc_id}' into collection '{self.collection_name}'")
        return len(chunks)

    def query_similarity(self,query:str, limit:int =3):
        query_embedding = self.embedder.get_embeddings(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        return [hit.payload['text'] for hit in search_result if 'text' in hit.payload]        