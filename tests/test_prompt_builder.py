from __future__ import annotations

import app.services.prompt_builder as prompt_builder_module
from app.services.prompt_builder import build_response


def test_build_response_without_api_key_uses_safe_fallback(monkeypatch) -> None:
    monkeypatch.setattr(prompt_builder_module.settings, "groq_api_key", "")

    response = build_response(
        text="I am feeling low today",
        top_emotion="sadness",
        crisis=False,
        strategy="few_shot",
    )

    assert "sadness" in response.lower()


def test_build_response_crisis_always_contains_guidance(monkeypatch) -> None:
    monkeypatch.setattr(prompt_builder_module.settings, "groq_api_key", "")

    response = build_response(
        text="I cannot go on",
        top_emotion="grief",
        crisis=True,
        strategy="zero_shot",
    )

    assert "immediate danger" in response.lower()


def test_build_response_retries_then_returns_groq_content(monkeypatch) -> None:
    monkeypatch.setattr(prompt_builder_module.settings, "groq_api_key", "test-key")

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeMessage(content)

    class FakeCompletion:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            _ = kwargs
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary error")
            return FakeCompletion("Generated supportive response")

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(prompt_builder_module, "_get_groq_client", lambda: FakeClient())
    monkeypatch.setattr(prompt_builder_module.time, "sleep", lambda _: None)

    response = build_response(
        text="I feel anxious",
        top_emotion="fear",
        crisis=False,
        strategy="few_shot",
    )

    assert response == "Generated supportive response"
