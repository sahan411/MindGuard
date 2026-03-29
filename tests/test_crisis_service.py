from app.services.crisis_service import CrisisService


def test_crisis_service_returns_required_keys() -> None:
    service = CrisisService()
    result = service.predict("I cannot do this anymore")
    assert "crisis_detected" in result
    assert "reconstruction_error" in result
