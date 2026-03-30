from app.core.constants import CRISIS_THRESHOLD


class VAEDetector:
    """Placeholder VAE detector wrapper."""

    def score(self, text: str) -> float:
        _ = text
        return 0.842

    def is_crisis(self, text: str) -> bool:
        return self.score(text) > CRISIS_THRESHOLD
