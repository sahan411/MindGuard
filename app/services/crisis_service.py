from app.models.vae_detector import VAEDetector


class CrisisService:
    def __init__(self) -> None:
        self.model = VAEDetector()

    def predict(self, text: str) -> dict:
        score = self.model.score(text)
        return {"crisis_detected": score > 0.65, "reconstruction_error": score}
