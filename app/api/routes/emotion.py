from fastapi import APIRouter, HTTPException

from app.api.schemas import EmotionResponse, EmotionScore, TextRequest
from app.services.emotion_service import EmotionService

router = APIRouter()
emotion_service = EmotionService()


@router.post("/emotion", response_model=EmotionResponse)
def predict_emotion(payload: TextRequest) -> EmotionResponse:
    try:
        prediction = emotion_service.predict(payload.text)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="emotion_prediction_failed") from exc

    emotions_raw = prediction.get("emotions", [])
    emotions = [
        EmotionScore(
            label=str(item.get("label", "neutral")),
            confidence=float(item.get("confidence", 0.0)),
        )
        for item in emotions_raw
        if isinstance(item, dict)
    ]

    if not emotions:
        emotions = [EmotionScore(label="neutral", confidence=0.0)]

    top_emotion = prediction.get("top_emotion")
    if not isinstance(top_emotion, str) or top_emotion == "":
        top_emotion = emotions[0].label

    return EmotionResponse(emotions=emotions, top_emotion=top_emotion)
