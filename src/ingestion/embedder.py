import logging
from typing import List

from src.core import llm
from src.core.cache import embedding_cache, make_key
from src.core.resilience import retry
from src.core.settings import cfg

logger = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or cfg("models", "embed", default="nomic-embed-text")
        # Dimension depends on the embedding model (768 for nomic, 1536 for
        # OpenAI text-embedding-3-small) — configurable so the provider can swap.
        self.dimensions = cfg("models", "embed_dim", default=768)

    @retry(attempts=3, base_delay=0.5)
    def _embed_remote(self, text: str) -> List[float]:
        try:
            return llm.embed(text, model=self.model_name)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            raise

    def get_embeddings(self, text: str) -> List[float]:
        # Embeddings are deterministic per (model, text) → safe to cache.
        key = make_key("embed", self.model_name, text)
        return embedding_cache.get_or_compute(key, lambda: self._embed_remote(text))

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embeddings(text) for text in texts]
