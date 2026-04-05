"""Pydantic request/response schemas for the MindGuard API."""

from typing import Annotated, List, Literal
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Payload for single-text prediction endpoints."""

    text: Annotated[
        str,
        Field(min_length=1, max_length=2000, description="User input text to analyse."),
    ]


class EmotionScore(BaseModel):
    """A single emotion label and its confidence score."""

    label: str = Field(..., description="Emotion label (e.g. 'sadness').")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in [0, 1]."
    )


class EmotionResponse(BaseModel):
    """Response from the emotion classification endpoint."""

    emotions: Annotated[
        List[EmotionScore],
        Field(
            min_length=1,
            max_length=10,
            description="Top-k emotions sorted by descending confidence.",
        ),
    ]
    top_emotion: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Label of the highest-confidence emotion.",
        ),
    ]


class CrisisResponse(BaseModel):
    """Response from the crisis detection endpoint."""

    crisis_detected: bool = Field(
        ..., description="True if reconstruction error exceeds the threshold."
    )
    reconstruction_error: Annotated[
        float, Field(ge=0.0, description="VAE reconstruction error for the input.")
    ]
    threshold: Annotated[
        float, Field(gt=0.0, description="Threshold used for this prediction.")
    ]
    method: Literal["vae", "keyword", "fallback"] = Field(
        ..., description="Detection method used."
    )
    keyword_match: bool = Field(
        ..., description="Whether any crisis keyword was matched."
    )
    crisis_guidance_required: bool = Field(
        default=False, description="Whether to show crisis guidance."
    )


class GenerateRequest(BaseModel):
    """Payload for the empathetic response generation endpoint."""

    text: Annotated[
        str,
        Field(min_length=1, max_length=2000, description="Original user message."),
    ]
    top_emotion: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Dominant emotion label from the emotion endpoint.",
        ),
    ]
    crisis: bool = Field(..., description="Crisis flag from the crisis endpoint.")
    strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = Field(
        default="few_shot",
        description="Prompting strategy to use when generating the response.",
    )


class GenerateResponse(BaseModel):
    """Response from the empathetic response generation endpoint."""

    response: str = Field(..., description="Generated empathetic text.")
    crisis_resource: str = Field(
        default="",
        description="Hotline / resource string; empty when no crisis detected.",
    )
    strategy_used: Literal["zero_shot", "few_shot", "chain_of_thought"] = Field(
        ..., description="Prompting strategy that was applied."
    )
