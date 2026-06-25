from src.retrieval import rewrite


def test_merge_unique_dedupes_keeps_order():
    lists = [["a", "b"], ["b", "c"], ["c", "d"]]
    assert rewrite.merge_unique(lists, limit=3) == ["a", "b", "c"]


def test_merge_unique_respects_limit():
    assert rewrite.merge_unique([["a", "b", "c", "d"]], limit=2) == ["a", "b"]


def test_expand_query_includes_original_and_variants(monkeypatch):
    monkeypatch.setattr(
        rewrite.llm, "chat",
        lambda messages, model=None: "How do I pay?\nProcess a payment\nmake a charge",
    )
    out = rewrite.expand_query("How to process a payment?", n=3)
    assert out[0] == "How to process a payment?"     # original always first
    assert "Process a payment" in out
    assert len(out) <= 4                             # original + up to n


def test_expand_query_falls_back_on_error(monkeypatch):
    def boom(messages, model=None):
        raise RuntimeError("no model")

    monkeypatch.setattr(rewrite.llm, "chat", boom)
    assert rewrite.expand_query("hello", n=3) == ["hello"]
