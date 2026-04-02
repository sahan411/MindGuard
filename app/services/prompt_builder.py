from __future__ import annotations

import time

from app.core.config import settings
from app.core.constants import CRISIS_GUIDANCE_SUFFIX


def _build_prompt(text: str, top_emotion: str, crisis: bool, strategy: str) -> str:
    strategy_guidance = {
        "zero_shot": "Respond empathetically and keep it concise.",
        "few_shot": "Respond empathetically using supportive and validating language.",
        "chain_of_thought": (
            "Think through emotional validation internally, but provide only the final "
            "supportive response to the user."
        ),
    }

    crisis_note = (
        "The user may be in crisis. Prioritize safety and encourage immediate professional help."
        if crisis
        else "No explicit crisis risk is detected."
    )

    return (
        "You are a supportive non-clinical assistant for an academic prototype. "
        "Do not provide diagnosis or medical instructions. "
        f"Detected emotion: {top_emotion}. "
        f"{crisis_note} "
        f"Strategy: {strategy_guidance.get(strategy, strategy_guidance['few_shot'])} "
        f"User text: {text}"
    )


def _safe_fallback_response(top_emotion: str, crisis: bool) -> str:
    if crisis:
        return (
            "I hear how hard this feels right now. You are not alone, and your safety matters. "
            "Please contact a trusted person or a professional support line right away."
            + CRISIS_GUIDANCE_SUFFIX
        )

    return (
        f"Thank you for sharing this. It sounds like you are feeling {top_emotion}. "
        "I am here to support you, and we can take this one step at a time."
    )


def _get_groq_client():
    from groq import Groq

    return Groq(api_key=settings.groq_api_key)


def _call_groq_with_retry(
    prompt: str, retries: int = 3, base_backoff_seconds: float = 1.0
) -> str | None:
    if not settings.groq_api_key:
        return None

    try:
        client = _get_groq_client()
    except Exception:
        return None

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=220,
                timeout=settings.groq_timeout_seconds,
            )
            content = completion.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            if attempt == retries - 1:
                break
            time.sleep(base_backoff_seconds * (attempt + 1))

    return None


def build_response(text: str, top_emotion: str, crisis: bool, strategy: str) -> str:
    """Build a safe response using Groq with retry, or local fallback on failure."""
    prompt = _build_prompt(
        text=text, top_emotion=top_emotion, crisis=crisis, strategy=strategy
    )
    generated = _call_groq_with_retry(prompt=prompt)
    if generated:
        if crisis and CRISIS_GUIDANCE_SUFFIX not in generated:
            return generated + CRISIS_GUIDANCE_SUFFIX
        return generated

    return _safe_fallback_response(top_emotion=top_emotion, crisis=crisis)
