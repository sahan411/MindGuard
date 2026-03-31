from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_emotion_endpoint() -> None:
    response = client.post("/predict/emotion", json={"text": "I feel sad"})
    assert response.status_code == 200
    assert "top_emotion" in response.json()


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
