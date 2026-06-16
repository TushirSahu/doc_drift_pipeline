from src.retrieval.reranker import (
    _rank_by_scores,
    cross_encoder_rerank,
    rerank,
)


def test_rank_by_scores_orders_and_truncates():
    cands = ["a", "b", "c", "d"]
    scores = [0.1, 0.9, 0.5, 0.2]
    assert _rank_by_scores(cands, scores, 2) == ["b", "c"]


def test_cross_encoder_uses_injected_scores():
    cands = ["irrelevant", "perfect match", "partial"]
    # Higher score = more relevant; "perfect match" should win.
    def fake_score(pairs):
        return [0.1, 0.95, 0.4]

    out = cross_encoder_rerank("q", cands, limit=2, score_fn=fake_score)
    assert out == ["perfect match", "partial"]


def test_cross_encoder_noop_when_within_limit():
    cands = ["a", "b"]
    assert cross_encoder_rerank("q", cands, limit=2, score_fn=lambda p: [1, 1]) == cands


def test_cross_encoder_falls_back_on_error():
    cands = ["a", "b", "c"]

    def boom(pairs):
        raise RuntimeError("model not installed")

    # Should not raise — returns original order truncated to limit.
    assert cross_encoder_rerank("q", cands, limit=2, score_fn=boom) == ["a", "b"]


def test_rerank_dispatches_to_llm(monkeypatch):
    import src.retrieval.reranker as mod
    called = {}

    def fake_llm(question, candidates, model_name, limit):
        called["llm"] = True
        return candidates[:limit]

    monkeypatch.setattr(mod, "llm_rerank", fake_llm)
    rerank("q", ["a", "b", "c"], limit=2, strategy="llm")
    assert called.get("llm") is True


def test_rerank_dispatches_to_cross_encoder(monkeypatch):
    import src.retrieval.reranker as mod
    called = {}

    def fake_ce(question, candidates, limit, model_name=None):
        called["ce"] = True
        return candidates[:limit]

    monkeypatch.setattr(mod, "cross_encoder_rerank", fake_ce)
    rerank("q", ["a", "b", "c"], limit=2, strategy="cross_encoder")
    assert called.get("ce") is True
