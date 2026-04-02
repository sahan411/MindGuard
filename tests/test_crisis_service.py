from __future__ import annotations

import app.services.crisis_service as crisis_service_module
from app.services.crisis_service import CrisisService


def test_crisis_service_returns_required_keys() -> None:
    service = CrisisService()
    result = service.predict("I cannot do this anymore")
    assert "crisis_detected" in result
    assert "reconstruction_error" in result


def test_crisis_service_threshold_boundary_detects_crisis() -> None:
    class BoundaryModel:
        def score(self, text: str) -> float:
            _ = text
            return 0.65

    service = CrisisService()
    service.model = BoundaryModel()
    service.threshold = 0.65

    result = service.predict("I am struggling")
    assert result["crisis_detected"] is True
    assert result["method"] == "vae"


def test_crisis_service_keyword_match_forces_guidance() -> None:
    class LowScoreModel:
        def score(self, text: str) -> float:
            _ = text
            return 0.05

    service = CrisisService()
    service.model = LowScoreModel()

    result = service.predict("I want to die and cannot go on")
    assert result["keyword_match"] is True
    assert result["crisis_detected"] is True
    assert result["crisis_guidance_required"] is True


def test_crisis_service_fallback_when_model_unavailable(monkeypatch) -> None:
    class FailingModel:
        def __init__(self) -> None:
            raise RuntimeError("init failed")

    monkeypatch.setattr(crisis_service_module, "VAEDetector", FailingModel)
    service = CrisisService()

    result = service.predict("ordinary text")
    assert result["method"] == "fallback"
    assert result["reconstruction_error"] == 0.0
