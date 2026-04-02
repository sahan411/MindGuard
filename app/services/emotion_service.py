from __future__ import annotations

from typing import Dict, List

from app.core.config import settings
from app.core.constants import EMOTION_TOP_K_DEFAULT
from app.models.bert_classifier import BertEmotionClassifier
from app.services.nlp_preprocessor import prepare_text


class EmotionService:
    def __init__(self) -> None:
        self.model = None
        self.model_error: str | None = None

        try:
            self.model = BertEmotionClassifier()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.model_error = str(exc)
            self.model = None

    def predict(self, text: str) -> Dict[str, object]:
        payload = prepare_text(text)

        # Keep API behavior stable even when model loading fails at startup.
        if self.model is None:
            return self._fallback_prediction()

        try:
            raw_output = self.model.predict(payload["model_text"])
        except Exception:  # pragma: no cover - defensive guard
            return self._fallback_prediction()

        return self._normalize_prediction(raw_output)

    def _normalize_prediction(self, raw_output: Dict[str, object]) -> Dict[str, object]:
        default = self._fallback_prediction()
        emotions_raw = raw_output.get("emotions")

        if not isinstance(emotions_raw, list) or not emotions_raw:
            return default

        normalized: List[Dict[str, object]] = []
        for entry in emotions_raw:
            if not isinstance(entry, dict):
                continue

            label = entry.get("label")
            confidence = entry.get("confidence")
            if not isinstance(label, str):
                continue

            try:
                score = float(confidence)
            except (TypeError, ValueError):
                continue

            normalized.append(
                {
                    "label": label,
                    # Clamp to [0, 1] because downstream contracts assume valid probability bounds.
                    "confidence": min(max(score, 0.0), 1.0),
                }
            )

        if not normalized:
            return default

        normalized.sort(key=lambda item: float(item["confidence"]), reverse=True)
        top_k = settings.emotion_top_k or EMOTION_TOP_K_DEFAULT
        limited = normalized[:top_k]
        top_emotion = raw_output.get("top_emotion")
        # Prefer model-provided top label when valid; otherwise derive from ranked scores.
        if not isinstance(top_emotion, str) or top_emotion == "":
            top_emotion = str(limited[0]["label"])

        return {"emotions": limited, "top_emotion": top_emotion}

    def _fallback_prediction(self) -> Dict[str, object]:
        return {
            "emotions": [
                {"label": "neutral", "confidence": 0.51},
                {"label": "sadness", "confidence": 0.27},
                {"label": "fear", "confidence": 0.22},
            ],
            "top_emotion": "neutral",
        }
