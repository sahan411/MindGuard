from typing import List

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class EmotionScore(BaseModel):
    label: str
    confidence: float


class EmotionResponse(BaseModel):
    emotions: List[EmotionScore]
    top_emotion: str


class CrisisResponse(BaseModel):
    crisis_detected: bool
    reconstruction_error: float
    threshold: float
    method: str
    keyword_match: bool


class GenerateRequest(BaseModel):
    text: str
    top_emotion: str
    crisis: bool
    strategy: str = "few_shot"


class GenerateResponse(BaseModel):
    response: str
    crisis_resource: str
    strategy_used: str
