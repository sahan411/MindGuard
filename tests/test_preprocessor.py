from app.services.nlp_preprocessor import normalize_text


def test_normalize_text() -> None:
    assert normalize_text("  Hello   WORLD ") == "hello world"
