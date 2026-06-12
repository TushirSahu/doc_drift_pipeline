"""
Separates retrieval *algorithms* from the vector store so you can
mix-and-match: dense search → MMR → hybrid → rerank.
"""
from src.retrieval.engine import RetrievalEngine
from src.retrieval.mmr import mmr_select
from src.retrieval.hybrid import hybrid_fuse
from src.retrieval.reranker import llm_rerank

__all__ = ["RetrievalEngine", "mmr_select", "hybrid_fuse", "llm_rerank"]
