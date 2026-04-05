from fastapi import APIRouter

from app.api.schemas import GenerateRequest, GenerateResponse
from app.core.constants import CRISIS_RESOURCE_SL
from app.services.prompt_builder import build_response

router = APIRouter()


@router.post("/response", response_model=GenerateResponse)
def generate_response(payload: GenerateRequest) -> GenerateResponse:
    """Generate an empathetic response conditioned on emotion and crisis state."""
    text = build_response(
        payload.text, payload.top_emotion, payload.crisis, payload.strategy
    )
    return GenerateResponse(
        response=text,
        crisis_resource=CRISIS_RESOURCE_SL if payload.crisis else "",
        strategy_used=payload.strategy,
    )
