"""
RetrievalEngine — orchestrates the retrieval pipeline.

Flow:
  1. Dense vector search (Qdrant) → fetch N candidates
  2. Optional: hybrid fuse with BM25 keyword scores
  3. Optional: MMR to deduplicate similar chunks
  4. Optional: LLM rerank to pick the best top_k
"""
import logging
from typing import List

from src.core.cache import make_key, retrieval_cache
from src.core.settings import cfg
from src.retrieval.hybrid import bm25_search, hybrid_fuse
from src.retrieval.mmr import mmr_select
from src.retrieval.reranker import rerank as rerank_candidates

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, vectorstore):
        self.vs = vectorstore

    def retrieve(
        self,
        query: str,
        limit: int | None = None,
        use_mmr: bool | None = None,
        use_hybrid: bool | None = None,
        rerank: bool | None = None,
        model_name: str | None = None,
    ) -> List[str]:
        limit = limit or cfg("retrieval", "top_k", default=2)
        use_mmr = use_mmr if use_mmr is not None else cfg("retrieval", "use_mmr", default=False)
        use_hybrid = use_hybrid if use_hybrid is not None else cfg("retrieval", "use_hybrid", default=False)
        rerank = rerank if rerank is not None else cfg("retrieval", "rerank", default=False)
        multi_query = cfg("retrieval", "multi_query", default=False)
        model_name = model_name or cfg("models", "llm", default="llama3.2:3b")

        # Identical (query, params) → identical result until docs change.
        cache_key = make_key(
            "retrieve", self.vs.collection_name, query, limit,
            use_mmr, use_hybrid, rerank, multi_query, model_name,
        )
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            return cached

        if multi_query:
            # Retrieve for each rephrasing, then merge unique chunks.
            from src.retrieval.rewrite import expand_query, merge_unique

            per_query = [
                self._retrieve_uncached(q, limit, use_mmr, use_hybrid, rerank, model_name)
                for q in expand_query(query, model=model_name)
            ]
            result = merge_unique(per_query, limit)
        else:
            result = self._retrieve_uncached(
                query, limit, use_mmr, use_hybrid, rerank, model_name
            )
        retrieval_cache.set(cache_key, result)
        return result

    def _retrieve_uncached(
        self, query, limit, use_mmr, use_hybrid, rerank, model_name
    ) -> List[str]:
        fetch_limit = limit
        if use_mmr or rerank or use_hybrid:
            fetch_limit = max(limit, cfg("retrieval", "rerank_candidates", default=10))

        query_embedding = self.vs.embedder.get_embeddings(query)
        hits = self.vs.client.query_points(
            collection_name=self.vs.collection_name,
            query=query_embedding,
            limit=fetch_limit,
        ).points

        if not hits:
            return []

        dense_scored = [
            (hit.payload["text"], hit.score)
            for hit in hits
            if hit.payload and "text" in hit.payload
        ]

        if use_hybrid:
            corpus = self.vs.get_corpus_texts()
            sparse_scored = bm25_search(query, corpus, limit=fetch_limit)
            if sparse_scored:
                candidates = hybrid_fuse(dense_scored, sparse_scored, limit=fetch_limit)
            else:
                logger.warning("BM25 unavailable; using dense only")
                candidates = [text for text, _ in dense_scored]
        else:
            candidates = [text for text, _ in dense_scored]

        if use_mmr:
            candidates_with_emb = [
                (text, self.vs.embedder.get_embeddings(text)) for text in candidates
            ]
            candidates = mmr_select(query_embedding, candidates_with_emb, limit)

        if rerank and len(candidates) > limit:
            candidates = rerank_candidates(query, candidates, limit, model_name=model_name)

        return candidates[:limit]
