from __future__ import annotations

from typing import Dict, List

from app.core.config import settings
from app.core.constants import EMOTION_TOP_K_DEFAULT


class BertEmotionClassifier:
    """BERT classifier wrapper with a safe local fallback prediction path."""

    def __init__(self) -> None:
        self.model_loaded = False
        self.model_error: str | None = None

        # Placeholder: real model artifact loading is added in the training-integration phase.
        self.model_loaded = False

    def predict(self, text: str) -> Dict[str, object]:
        """Return top emotion plus sorted emotion scores for the given text."""
        return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Dict[str, object]:
        lowered = text.lower()
        scores: Dict[str, float] = {
            "sadness": 0.18,
            "neutral": 0.20,
            "fear": 0.14,
            "anger": 0.14,
            "joy": 0.14,
            "grief": 0.10,
            "optimism": 0.10,
        }

        sadness_markers = {"sad", "down", "low", "hopeless", "cry", "tired"}
        fear_markers = {"anxious", "afraid", "scared", "panic", "worry"}
        anger_markers = {"angry", "mad", "hate", "furious"}
        joy_markers = {"happy", "great", "good", "excited", "grateful"}

        if any(token in lowered for token in sadness_markers):
            scores["sadness"] += 0.52
            scores["neutral"] -= 0.10
        if any(token in lowered for token in fear_markers):
            scores["fear"] += 0.48
            scores["neutral"] -= 0.08
        if any(token in lowered for token in anger_markers):
            scores["anger"] += 0.48
            scores["neutral"] -= 0.08
        if any(token in lowered for token in joy_markers):
            scores["joy"] += 0.55
            scores["neutral"] -= 0.10

        clipped_scores = {
            label: min(max(value, 0.0), 1.0) for label, value in scores.items()
        }
        sorted_scores: List[Dict[str, object]] = [
            {"label": label, "confidence": confidence}
            for label, confidence in sorted(
                clipped_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]

        top_k = settings.emotion_top_k or EMOTION_TOP_K_DEFAULT
        top_emotions = sorted_scores[:top_k]

        return {
            "emotions": top_emotions,
            "top_emotion": top_emotions[0]["label"] if top_emotions else "neutral",
        }
