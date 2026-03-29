from app.models.bert_classifier import BertEmotionClassifier


class EmotionService:
    def __init__(self) -> None:
        self.model = BertEmotionClassifier()

    def predict(self, text: str) -> dict:
        return self.model.predict(text)
