def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase input text."""
    return " ".join(text.strip().lower().split())
