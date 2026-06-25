from src.evaluation.feedback import (
    JsonlFeedbackStore,
    PostgresFeedbackStore,
    get_store,
    load_regression_cases,
    record_feedback,
    regression_qa_pairs,
)


def test_get_store_selects_backend(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert isinstance(get_store(), JsonlFeedbackStore)
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    assert isinstance(get_store(), PostgresFeedbackStore)  # lazy: doesn't connect


def _paths(tmp_path):
    return tmp_path / "feedback.jsonl", tmp_path / "regression.jsonl"


def test_upvote_is_recorded_not_promoted(tmp_path):
    fb, reg = _paths(tmp_path)
    entry = record_feedback(
        question="What auth does v2 use?", answer="OAuth2", rating="up",
        fb_path=fb, reg_path=reg,
    )
    assert entry["rating"] == "up"
    assert entry["promoted_to_regression"] is False
    assert fb.exists()
    assert not reg.exists()  # up-votes don't create regression cases


def test_downvote_is_promoted_to_regression(tmp_path):
    fb, reg = _paths(tmp_path)
    entry = record_feedback(
        question="How long is the admin session?", answer="5 minutes (wrong)",
        rating="down", correct_answer="12 hours", fb_path=fb, reg_path=reg,
    )
    assert entry["promoted_to_regression"] is True
    cases = load_regression_cases(reg)
    assert len(cases) == 1
    assert cases[0]["reference"] == "12 hours"


def test_invalid_rating_rejected(tmp_path):
    fb, reg = _paths(tmp_path)
    try:
        record_feedback(question="q", answer="a", rating="meh", fb_path=fb, reg_path=reg)
        assert False, "should have raised"
    except ValueError:
        pass


def test_regression_qa_pairs_only_includes_referenced(tmp_path):
    fb, reg = _paths(tmp_path)
    record_feedback(question="q1", answer="bad", rating="down",
                    correct_answer="good answer", fb_path=fb, reg_path=reg)
    record_feedback(question="q2", answer="bad2", rating="down",
                    fb_path=fb, reg_path=reg)  # no correction -> watchlist only
    pairs = regression_qa_pairs(reg)
    assert pairs == [{"question": "q1", "answer": "good answer"}]


def test_regression_cases_dedup_by_question(tmp_path):
    fb, reg = _paths(tmp_path)
    record_feedback(question="dupe", answer="v1", rating="down",
                    correct_answer="first", fb_path=fb, reg_path=reg)
    record_feedback(question="dupe", answer="v2", rating="down",
                    correct_answer="second", fb_path=fb, reg_path=reg)
    cases = load_regression_cases(reg)
    assert len(cases) == 1
    assert cases[0]["reference"] == "second"  # last wins
