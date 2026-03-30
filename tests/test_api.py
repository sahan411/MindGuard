from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_emotion_endpoint() -> None:
    response = client.post("/predict/emotion", json={"text": "I feel sad"})
    assert response.status_code == 200
    assert "top_emotion" in response.json()
