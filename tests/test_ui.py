from __future__ import annotations

from app.core.constants import CRISIS_RESOURCE_TEXT
from app.ui.gradio_app import _format_emotions, analyze_text


def test_format_emotions_outputs_label_and_confidence() -> None:
    rows = _format_emotions(
        [
            {"label": "sadness", "confidence": 0.8731},
            {"label": "fear", "confidence": 0.6111},
        ]
    )
    assert rows == [["sadness", "0.873"], ["fear", "0.611"]]


def test_analyze_text_handles_empty_input() -> None:
    response, emotions, top_emotion, crisis_flag, guidance = analyze_text("  ")

    assert "Please enter your message" in response
    assert emotions == []
    assert top_emotion == "N/A"
    assert crisis_flag == "No"
    assert guidance == CRISIS_RESOURCE_TEXT


def test_analyze_text_returns_expected_contract() -> None:
    response, emotions, top_emotion, crisis_flag, guidance = analyze_text(
        "I feel low and anxious"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(emotions, list)
    assert len(emotions) >= 1
    assert isinstance(top_emotion, str)
    assert crisis_flag in {"Yes", "No"}
    if crisis_flag == "Yes":
        assert guidance == CRISIS_RESOURCE_TEXT
