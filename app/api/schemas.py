from typing import Annotated, List, Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Input payload for routes that only require user text."""

    text: Annotated[
        str,
        Field(min_length=1, max_length=4000, description="User input text."),
    ]


class EmotionScore(BaseModel):
    """Single emotion probability output."""

    label: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class EmotionResponse(BaseModel):
    """Emotion prediction payload returned by the API."""

    emotions: Annotated[List[EmotionScore], Field(min_length=1, max_length=10)]
    top_emotion: Annotated[str, Field(min_length=1, max_length=64)]


class CrisisResponse(BaseModel):
    """Crisis prediction payload returned by the API."""

    crisis_detected: bool
    reconstruction_error: Annotated[float, Field(ge=0.0)]
    threshold: Annotated[float, Field(gt=0.0)]
    method: Literal["vae", "keyword", "fallback"]
    keyword_match: bool


class GenerateRequest(BaseModel):
    """Input payload for empathetic response generation."""

    text: Annotated[
        str,
        Field(min_length=1, max_length=4000, description="User input text."),
    ]
    top_emotion: Annotated[str, Field(min_length=1, max_length=64)]
    crisis: bool
    strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "few_shot"


class GenerateResponse(BaseModel):
    """Generated response payload with safety metadata."""

    response: str
    crisis_resource: str = ""
    strategy_used: Literal["zero_shot", "few_shot", "chain_of_thought"]
