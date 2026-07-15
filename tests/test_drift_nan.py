"""Drift gate stays sane when a weak judge NaNs a metric."""
from src.evaluation.drift import check_drift, finite_only


def test_finite_only_drops_nan_and_inf():
    d = {"a": 0.9, "b": float("nan"), "c": float("inf"), "d": 0.5}
    assert finite_only(d) == {"a": 0.9, "d": 0.5}


def test_check_drift_skips_a_nan_metric():
    # A NaN metric is dropped, not silently compared as pass or fail.
    passed, reasons = check_drift({"faithfulness": float("nan")},
                                  {"faithfulness": 0.9})
    assert passed
    assert reasons == []


def test_check_drift_still_catches_a_real_regression():
    passed, reasons = check_drift({"faithfulness": 0.5}, {"faithfulness": 0.9})
    assert not passed
    assert reasons
