"""
Separates retrieval *algorithms* from the vector store so you can
mix-and-match: dense search → MMR → hybrid → rerank.
"""
from src.retrieval.engine import RetrievalEngine
from src.retrieval.mmr import mmr_select
from src.retrieval.hybrid import hybrid_fuse
from src.retrieval.reranker import cross_encoder_rerank, llm_rerank, rerank
from src.retrieval.rewrite import expand_query, merge_unique

__all__ = [
    "RetrievalEngine",
    "mmr_select",
    "hybrid_fuse",
    "rerank",
    "llm_rerank",
    "cross_encoder_rerank",
    "expand_query",
    "merge_unique",
]
