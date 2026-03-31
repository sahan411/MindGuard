from __future__ import annotations

from typing import Dict

from app.core.config import settings
from app.core.constants import CRISIS_THRESHOLD
from app.models.vae_detector import VAEDetector
from app.services.nlp_preprocessor import prepare_text


class CrisisService:
    """Crisis prediction service with VAE and keyword-based safety fallback."""

    def __init__(self) -> None:
        self.model = None
        self.threshold = settings.default_crisis_threshold or CRISIS_THRESHOLD

        try:
            self.model = VAEDetector()
        except Exception:  # pragma: no cover - defensive guard
            self.model = None

    def predict(self, text: str) -> Dict[str, object]:
        payload = prepare_text(text)
        model_text = payload["model_text"]
        keyword_match = self._keyword_baseline(model_text)

        method = "vae"
        score = 0.0
        vae_detected = False

        if self.model is not None:
            try:
                score = float(self.model.score(model_text))
                vae_detected = score >= self.threshold
            except Exception:  # pragma: no cover - defensive guard
                method = "fallback"
        else:
            method = "fallback"

        crisis_detected = vae_detected or keyword_match

        return {
            "crisis_detected": crisis_detected,
            "reconstruction_error": score,
            "threshold": self.threshold,
            "method": method,
            "keyword_match": keyword_match,
            "crisis_guidance_required": crisis_detected,
        }

    @staticmethod
    def _keyword_baseline(text: str) -> bool:
        crisis_markers = {
            "suicide",
            "kill myself",
            "end my life",
            "self harm",
            "can't go on",
            "cannot go on",
            "want to die",
            "no reason to live",
        }

        lowered = text.lower()
        return any(marker in lowered for marker in crisis_markers)
