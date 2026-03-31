from fastapi import APIRouter

from app.api.schemas import GenerateRequest, GenerateResponse
from app.core.constants import CRISIS_RESOURCE_TEXT
from app.services.prompt_builder import build_response

router = APIRouter()


@router.post("/response", response_model=GenerateResponse)
def generate_response(payload: GenerateRequest) -> GenerateResponse:
    text = build_response(
        payload.text, payload.top_emotion, payload.crisis, payload.strategy
    )
    return GenerateResponse(
        response=text,
        crisis_resource=CRISIS_RESOURCE_TEXT if payload.crisis else "",
        strategy_used=payload.strategy,
    )
