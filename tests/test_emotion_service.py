from __future__ import annotations

from typing import Dict

import app.services.emotion_service as emotion_service_module
from app.services.emotion_service import EmotionService


def test_emotion_service_predict_has_top_emotion() -> None:
    service = EmotionService()
    result = service.predict("I feel low")
    assert "top_emotion" in result


def test_emotion_service_returns_sorted_emotions() -> None:
    service = EmotionService()
    result = service.predict("I feel anxious and sad")

    emotions = result["emotions"]
    assert isinstance(emotions, list)
    assert len(emotions) > 0
    assert float(emotions[0]["confidence"]) >= float(emotions[-1]["confidence"])


def test_emotion_service_uses_fallback_when_model_init_fails(monkeypatch) -> None:
    class FailingModel:
        def __init__(self) -> None:
            raise RuntimeError("model init failed")

    monkeypatch.setattr(emotion_service_module, "BertEmotionClassifier", FailingModel)

    service = EmotionService()
    result = service.predict("any text")

    assert result["top_emotion"] == "neutral"
    assert len(result["emotions"]) >= 1


def test_emotion_service_uses_fallback_when_model_predict_fails(monkeypatch) -> None:
    class BrokenPredictModel:
        def predict(self, text: str) -> Dict[str, object]:
            _ = text
            raise RuntimeError("predict failed")

    monkeypatch.setattr(
        emotion_service_module,
        "BertEmotionClassifier",
        lambda: BrokenPredictModel(),
    )

    service = EmotionService()
    result = service.predict("any text")

    assert result["top_emotion"] == "neutral"
    assert len(result["emotions"]) >= 1
