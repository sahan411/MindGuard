from app.models.vae_detector import VAEDetector
from app.services.nlp_preprocessor import prepare_text


class CrisisService:
    def __init__(self) -> None:
        self.model = VAEDetector()

    def predict(self, text: str) -> dict:
        payload = prepare_text(text)
        score = self.model.score(payload["model_text"])
        return {"crisis_detected": score > 0.65, "reconstruction_error": score}
