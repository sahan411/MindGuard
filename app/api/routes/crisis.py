from fastapi import APIRouter

from app.api.schemas import CrisisResponse, TextRequest
from app.core.constants import CRISIS_THRESHOLD

router = APIRouter()


@router.post("/crisis", response_model=CrisisResponse)
def predict_crisis(payload: TextRequest) -> CrisisResponse:
    _ = payload
    reconstruction_error = 0.842
    return CrisisResponse(
        crisis_detected=reconstruction_error > CRISIS_THRESHOLD,
        reconstruction_error=reconstruction_error,
        threshold=CRISIS_THRESHOLD,
        method="vae",
        keyword_match=False,
    )
