def build_response(text: str, top_emotion: str, crisis: bool, strategy: str) -> str:
    """Build a safe, empathetic response stub."""
    _ = strategy
    if crisis:
        return (
            "I hear how hard this feels right now. You are not alone. "
            "Please reach out to a trusted person or a professional support line today."
        )
    return f"Thank you for sharing. It sounds like you are feeling {top_emotion}."
