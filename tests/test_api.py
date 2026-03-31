from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_emotion_endpoint() -> None:
    response = client.post("/predict/emotion", json={"text": "I feel sad"})
    assert response.status_code == 200
    body = response.json()
    assert "top_emotion" in body
    assert isinstance(body["emotions"], list)
    assert len(body["emotions"]) > 0


def test_predict_crisis_endpoint_contract() -> None:
    response = client.post("/predict/crisis", json={"text": "I cannot go on"})
    assert response.status_code == 200
    body = response.json()
    assert "crisis_detected" in body
    assert "reconstruction_error" in body
    assert "threshold" in body
    assert "method" in body
    assert "keyword_match" in body
    assert "crisis_guidance_required" in body


def test_generate_response_endpoint() -> None:
    response = client.post(
        "/generate/response",
        json={
            "text": "I feel low",
            "top_emotion": "sadness",
            "crisis": False,
            "strategy": "few_shot",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert body["strategy_used"] == "few_shot"


def test_generate_response_endpoint_crisis_resource() -> None:
    response = client.post(
        "/generate/response",
        json={
            "text": "I cannot go on",
            "top_emotion": "grief",
            "crisis": True,
            "strategy": "zero_shot",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crisis_resource"] != ""


def test_predict_endpoint_rejects_empty_text() -> None:
    response = client.post("/predict/emotion", json={"text": ""})
    assert response.status_code == 422


def test_generate_endpoint_rejects_invalid_strategy() -> None:
    response = client.post(
        "/generate/response",
        json={
            "text": "I feel low",
            "top_emotion": "sadness",
            "crisis": False,
            "strategy": "unsupported_strategy",
        },
    )
    assert response.status_code == 422
