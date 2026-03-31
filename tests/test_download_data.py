from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_data import (
    _ensure_required_columns,
    _validate_amod,
    _validate_go_emotions,
)


def test_ensure_required_columns_raises_for_missing() -> None:
    df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError):
        _ensure_required_columns(df, ["text", "labels"], "go_emotions")


def test_validate_go_emotions_accepts_valid_labels() -> None:
    df = pd.DataFrame(
        {
            "text": ["sample"],
            "labels": [[1, 5]],
            "comment_id": ["abc"],
        }
    )
    _validate_go_emotions(df)


def test_validate_go_emotions_rejects_invalid_labels() -> None:
    df = pd.DataFrame(
        {
            "text": ["sample"],
            "labels": [[99]],
            "comment_id": ["abc"],
        }
    )
    with pytest.raises(ValueError):
        _validate_go_emotions(df)


def test_validate_amod_rejects_empty_response() -> None:
    df = pd.DataFrame(
        {
            "Context": ["How can I cope?"],
            "Response": ["  "],
        }
    )
    with pytest.raises(ValueError):
        _validate_amod(df)
