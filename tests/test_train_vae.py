from __future__ import annotations

import numpy as np
import pytest

from scripts.train_vae import _binary_metrics, _estimate_threshold, _is_positive_label


def test_estimate_threshold_returns_percentile_value() -> None:
    errors = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold = _estimate_threshold(errors, 80.0)
    assert threshold >= 0.4


def test_estimate_threshold_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        _estimate_threshold([], 95.0)


def test_binary_metrics_basic_case() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = _binary_metrics(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_positive_label_parser() -> None:
    positives = {"1", "true", "crisis"}
    assert _is_positive_label("crisis", positives) is True
    assert _is_positive_label("0", positives) is False
