"""Offline lexical eval judge (no live LLM)."""
import pytest

pytest.importorskip("pandas")

from src.evaluation.offline_judge import LexicalResult, _row_scores, lexical_frame

METRICS = {"faithfulness", "answer_relevancy", "answer_correctness"}


def test_row_scores_are_bounded_and_named():
    s = _row_scores(
        "what auth does the service use",
        "OAuth2 and JWT tokens secure the service.",
        ["OAuth2 and JWT tokens secure the auth service."],
        "OAuth2 and JWT tokens",
    )
    assert set(s) == METRICS
    assert all(0.0 <= v <= 1.0 for v in s.values())
    assert s["faithfulness"] > 0          # answer overlaps its context


def test_unsupported_answer_scores_low():
    s = _row_scores("q", "Totally unrelated pineapple content.",
                    ["OAuth2 and JWT secure the service."], "OAuth2")
    assert s["faithfulness"] == 0.0


def test_lexical_frame_and_result_shape():
    data = {
        "user_input": ["q1"],
        "response": ["OAuth2 secures auth"],
        "retrieved_contexts": [["OAuth2 secures the auth service"]],
        "reference": ["OAuth2 auth"],
    }
    df = lexical_frame(data)
    assert set(df.columns) == METRICS
    assert len(df) == 1
    # LexicalResult mimics a Ragas result so callers need no changes.
    assert LexicalResult(df).to_pandas() is df
