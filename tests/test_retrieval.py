from src.retrieval.hybrid import bm25_search, hybrid_fuse
from src.retrieval.mmr import cosine_similarity, mmr_select


def test_cosine_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6


def test_mmr_prefers_diverse_chunks():
    query_emb = [1.0, 0.0, 0.0]
    candidates = [
        ("most relevant", [1.0, 0.0, 0.0]),
        ("near-duplicate", [0.99, 0.01, 0.0]),
        ("different topic", [0.0, 1.0, 0.0]),
    ]
    # Low lambda favours diversity on the 2nd pick
    selected = mmr_select(query_emb, candidates, limit=2, lambda_param=0.3)
    assert len(selected) == 2
    assert selected[0] == "most relevant"
    assert selected[1] == "different topic"


def test_hybrid_fuse_combines_dense_and_sparse():
    dense = [("doc_a", 0.9), ("doc_b", 0.5)]
    sparse = [("doc_b", 0.8), ("doc_c", 0.6)]
    result = hybrid_fuse(dense, sparse, alpha=0.5, limit=3)
    assert len(result) <= 3
    assert "doc_b" in result


def test_bm25_finds_exact_keyword():
    corpus = [
        "Auth uses OAuth2 and JWT tokens",
        "Payment processing via Stripe",
    ]
    results = bm25_search("OAuth2", corpus, limit=1)
    assert len(results) == 1
    assert "OAuth2" in results[0][0]
