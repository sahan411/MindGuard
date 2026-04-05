"""NLP pre-processing utilities shared across all MindGuard services."""

import re
import unicodedata
from typing import Any, Dict

# Minimum number of non-whitespace characters required after normalisation.
_MIN_CHARS: int = 3
_WHITESPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[^a-z0-9\s.,!?\-']")

def normalize_text(text: str) -> str:
    """Normalise text for downstream NLP processing."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}")

    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _NOISE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()

    if not text:
        raise ValueError("Text is empty after normalisation.")
    if len(text.replace(" ", "")) < _MIN_CHARS:
        raise ValueError(f"Text is too short after normalisation.")
    return text

def prepare_text(text: str) -> Dict[str, Any]:
    """Build a stable preprocessing payload consumed by downstream services."""
    try:
        normalized_text = normalize_text(text)
    except (TypeError, ValueError):
        normalized_text = ""
        
    token_count = len(normalized_text.split()) if normalized_text else 0
    is_empty = token_count == 0
    is_short = token_count < 3

    model_text = normalized_text if normalized_text else "no content provided"

    return {
        "raw_text": text,
        "normalized_text": normalized_text,
        "model_text": model_text,
        "token_count": token_count,
        "is_empty": is_empty,
        "is_short": is_short,
    }
