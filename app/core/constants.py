"""Project-wide constants used across API, services, and UI layers."""

from typing import Final


CRISIS_THRESHOLD_DEFAULT: Final[float] = 0.65
EMOTION_TOP_K_DEFAULT: Final[int] = 3

SAFETY_DISCLAIMER_TEXT: Final[str] = (
	"MindGuard is a non-clinical academic prototype and does not provide diagnosis."
)
CRISIS_RESOURCE_TEXT: Final[str] = "Lifeline Sri Lanka: 1926 | CCCline: 1333"
CRISIS_GUIDANCE_SUFFIX: Final[str] = (
	" If you may be in immediate danger, contact local emergency services immediately."
)

# Backward-compatible aliases used by current route stubs.
CRISIS_THRESHOLD: Final[float] = CRISIS_THRESHOLD_DEFAULT
CRISIS_RESOURCE_SL: Final[str] = CRISIS_RESOURCE_TEXT
