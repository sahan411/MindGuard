from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration loaded from environment variables."""

    environment: Literal["dev", "test", "prod"] = "dev"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: int = Field(default=30, ge=1, le=180)

    default_response_strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "few_shot"
    default_crisis_threshold: float = Field(default=0.65, gt=0.0)
    emotion_top_k: int = Field(default=3, ge=1, le=10)

    safety_disclaimer_text: str = (
        "MindGuard is a non-clinical academic prototype and does not provide diagnosis."
    )
    crisis_resource_text: str = "Lifeline Sri Lanka: 1926 | CCCline: 1333"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
