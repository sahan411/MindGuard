import pytest

from app.services.nlp_preprocessor import normalize_text, prepare_text


def test_normalize_text() -> None:
    assert normalize_text("  Hello   WORLD ") == "hello world"


def test_normalize_text_strips_noise_and_non_ascii() -> None:
    assert normalize_text("Hi🙂   THERE!!!\n\tCaf\u00e9 #1") == "hi there!!! cafe 1"


def test_normalize_text_raises_for_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


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
