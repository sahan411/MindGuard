"""Unit tests for app.services.nlp_preprocessor.normalize_text and prepare_text."""

import pytest
from app.services.nlp_preprocessor import normalize_text, prepare_text

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_normalizes_whitespace_and_case() -> None:
    assert normalize_text("  Hello   WORLD ") == "hello world"


def test_normalize_text_strips_noise_and_non_ascii() -> None:
    assert normalize_text("Hi🙂   THERE!!!\n\tCaf\u00e9 #1") == "hi there!!! cafe 1"


def test_collapses_tabs_and_newlines() -> None:
    assert normalize_text("I\tfeel\nvery sad today") == "i feel very sad today"


def test_already_clean_text_unchanged() -> None:
    assert normalize_text("i feel okay") == "i feel okay"


def test_multiple_internal_spaces_collapsed() -> None:
    assert normalize_text("i    cannot   cope") == "i cannot cope"


# ---------------------------------------------------------------------------
# Non-ASCII / noise stripping
# ---------------------------------------------------------------------------


def test_strips_emoji() -> None:
    result = normalize_text("I feel sad 😢")
    assert "😢" not in result
    assert "i feel sad" in result


def test_strips_unicode_accented_chars() -> None:
    result = normalize_text("caf\u00e9 latte vibes")
    assert "\u00e9" not in result


def test_strips_rtl_control_characters() -> None:
    result = normalize_text("hello\u200fworld")
    assert "\u200f" not in result


def test_nfkc_normalisation_collapses_ligatures() -> None:
    result = normalize_text("\ufb01ne")
    assert "\ufb01" not in result


# ---------------------------------------------------------------------------
# Edge cases -> ValueError
# ---------------------------------------------------------------------------


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_text("")


def test_whitespace_only_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_text("     ")


def test_too_short_after_normalisation_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        normalize_text("ok")


def test_non_ascii_only_raises() -> None:
    with pytest.raises(ValueError):
        normalize_text("😢😢😢")


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def test_non_string_raises_type_error() -> None:
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


def test_integer_raises_type_error() -> None:
    with pytest.raises(TypeError):
        normalize_text(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_input_same_output() -> None:
    text = "  I feel  really  low   today  "
    assert normalize_text(text) == normalize_text(text)


def test_repeated_calls_stable() -> None:
    text = "panic attacks every night"
    results = [normalize_text(text) for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# prepare_text tests
# ---------------------------------------------------------------------------


def test_prepare_text_for_empty_input_uses_safe_placeholder() -> None:
    payload = prepare_text("   \n\t   ")
    assert payload["normalized_text"] == ""
    assert payload["model_text"] == "no content provided"
    assert payload["is_empty"] is True
    assert payload["token_count"] == 0


def test_prepare_text_short_text_flag() -> None:
    payload = prepare_text("low mood")
    assert payload["is_empty"] is False
    assert payload["is_short"] is True
    assert payload["token_count"] == 2
