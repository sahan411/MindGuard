import pandas as pd

from scripts.evaluate_baselines import _keyword_baseline_predict, _prf


def test_keyword_baseline_predict_detects_crisis_phrase() -> None:
    assert _keyword_baseline_predict("I want to die") == 1
    assert _keyword_baseline_predict("I am okay today") == 0


def test_prf_returns_expected_keys() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    metrics = _prf(y_true, y_pred)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    assert 0 <= metrics["f1"] <= 1
