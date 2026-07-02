import pytest

from src.core.resilience import retry


def test_retry_succeeds_first_try():
    calls = {"n": 0}

    @retry(attempts=3, sleep=lambda s: None)
    def ok():
        calls["n"] += 1
        return "done"

    assert ok() == "done"
    assert calls["n"] == 1


def test_retry_recovers_after_failures():
    calls = {"n": 0}

    @retry(attempts=3, sleep=lambda s: None)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts():
    calls = {"n": 0}

    @retry(attempts=2, sleep=lambda s: None)
    def always_fails():
        calls["n"] += 1
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        always_fails()
    assert calls["n"] == 2


def test_retry_only_catches_listed_exceptions():
    @retry(attempts=3, exceptions=(ConnectionError,), sleep=lambda s: None)
    def raises_value():
        raise ValueError("not retried")

    with pytest.raises(ValueError):
        raises_value()


def test_invalid_attempts_rejected():
    with pytest.raises(ValueError):
        retry(attempts=0)
