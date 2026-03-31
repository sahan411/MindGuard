from fastapi import APIRouter, HTTPException

from app.api.schemas import CrisisResponse, TextRequest
from app.services.crisis_service import CrisisService

router = APIRouter()
crisis_service = CrisisService()


@router.post("/crisis", response_model=CrisisResponse)
def predict_crisis(payload: TextRequest) -> CrisisResponse:
    try:
        prediction = crisis_service.predict(payload.text)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="crisis_prediction_failed") from exc

    method = str(prediction.get("method", "fallback"))
    if method not in {"vae", "keyword", "fallback"}:
        method = "fallback"

    return CrisisResponse(
        crisis_detected=bool(prediction.get("crisis_detected", False)),
        reconstruction_error=float(prediction.get("reconstruction_error", 0.0)),
        threshold=float(prediction.get("threshold", crisis_service.threshold)),
        method=method,
        keyword_match=bool(prediction.get("keyword_match", False)),
        crisis_guidance_required=bool(
            prediction.get("crisis_guidance_required", False)
        ),
    )
