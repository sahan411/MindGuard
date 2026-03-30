from app.models.bert_classifier import BertEmotionClassifier
from app.services.nlp_preprocessor import prepare_text


class EmotionService:
    def __init__(self) -> None:
        self.model = BertEmotionClassifier()

    def predict(self, text: str) -> dict:
        payload = prepare_text(text)
        return self.model.predict(payload["model_text"])
