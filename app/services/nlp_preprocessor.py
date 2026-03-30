import re
import unicodedata
from typing import Any, Dict

_WHITESPACE_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"[^a-z0-9\s.,!?\-']")


def normalize_text(text: str) -> str:
    """Return deterministic normalized text for model inference."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = _NOISE_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def prepare_text(text: str) -> Dict[str, Any]:
    """Build a stable preprocessing payload consumed by downstream services."""
    normalized_text = normalize_text(text)
    token_count = len(normalized_text.split()) if normalized_text else 0
    is_empty = token_count == 0
    is_short = token_count < 3

    # Provide a safe placeholder so inference paths do not fail on empty input.
    model_text = normalized_text if normalized_text else "no content provided"

    return {
        "raw_text": text,
        "normalized_text": normalized_text,
        "model_text": model_text,
        "token_count": token_count,
        "is_empty": is_empty,
        "is_short": is_short,
    }
