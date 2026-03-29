from app.services.emotion_service import EmotionService


def test_emotion_service_predict_has_top_emotion() -> None:
    service = EmotionService()
    result = service.predict("I feel low")
    assert "top_emotion" in result
