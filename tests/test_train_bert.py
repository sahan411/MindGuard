import pytest

from scripts.train_bert import _labels_to_multihot


def test_labels_to_multihot_valid_case() -> None:
    result = _labels_to_multihot([0, 2, 5], 6)
    assert result == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_labels_to_multihot_rejects_invalid_label() -> None:
    with pytest.raises(ValueError):
        _labels_to_multihot([999], 28)
