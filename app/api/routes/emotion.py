from fastapi import APIRouter

from app.api.schemas import EmotionResponse, EmotionScore, TextRequest

router = APIRouter()


@router.post("/emotion", response_model=EmotionResponse)
def predict_emotion(payload: TextRequest) -> EmotionResponse:
    _ = payload
    return EmotionResponse(
        emotions=[
            EmotionScore(label="sadness", confidence=0.87),
            EmotionScore(label="fear", confidence=0.61),
            EmotionScore(label="grief", confidence=0.44),
        ],
        top_emotion="sadness",
    )
