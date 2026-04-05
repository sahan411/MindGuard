"""Project-wide constants used across API, services, and UI layers."""

from typing import Final

# ---------------------------------------------------------------------------
# Crisis detection
# ---------------------------------------------------------------------------

#: VAE reconstruction-error threshold; exceed -> crisis flagged.
#: Set conservatively (recall-oriented) to minimise missed cases.
CRISIS_THRESHOLD_DEFAULT: Final[float] = 0.65
CRISIS_THRESHOLD: Final[float] = CRISIS_THRESHOLD_DEFAULT

#: Crisis hotline displayed to the user when crisis_detected is True.
CRISIS_RESOURCE_TEXT: Final[str] = "Lifeline Sri Lanka: 1926 | CCCline: 1333"
CRISIS_RESOURCE_SL: Final[str] = CRISIS_RESOURCE_TEXT
CRISIS_RESOURCE_INTERNATIONAL: Final[str] = "Crisis Text Line: text HOME to 741741"

CRISIS_GUIDANCE_SUFFIX: Final[str] = (
    " If you may be in immediate danger, contact local emergency services immediately."
)

# ---------------------------------------------------------------------------
# Emotion classification
# ---------------------------------------------------------------------------

#: Number of top-k emotions returned per prediction.
TOP_K_EMOTIONS: Final[int] = 3
EMOTION_TOP_K_DEFAULT: Final[int] = TOP_K_EMOTIONS

#: Minimum confidence score for an emotion to be included in the response.
EMOTION_MIN_CONFIDENCE: Final[float] = 0.10

# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

#: Default prompting strategy sent to the LLM.
DEFAULT_STRATEGY: Final[str] = "few_shot"

# ---------------------------------------------------------------------------
# Safety messaging
# ---------------------------------------------------------------------------

#: Disclaimer appended to all UI surfaces and API metadata.
SAFETY_DISCLAIMER: Final[str] = (
    "MindGuard is a research prototype only. "
    "It is not a clinical tool and must not be used for diagnosis "
    "or emergency intervention. "
    "If someone is in immediate danger, contact local emergency services."
)
SAFETY_DISCLAIMER_TEXT: Final[str] = SAFETY_DISCLAIMER
