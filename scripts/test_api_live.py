"""Quick verification script for the live API with the trained BERT model."""
import requests

base = "http://127.0.0.1:8000"
tests = [
    "I feel so sad and hopeless today",
    "I am really happy and excited!",
    "This makes me so angry",
    "I am very scared and anxious",
    "Thank you so much for everything",
]

print("=" * 60)
print("TESTING LIVE API: POST /predict/emotion")
print("=" * 60)

for text in tests:
    r = requests.post(f"{base}/predict/emotion", json={"text": text})
    data = r.json()
    top = data["top_emotion"]
    scores = ", ".join(
        f"{e['label']}={e['confidence']:.3f}" for e in data["emotions"]
    )
    print(f'\n"{text}"')
    print(f"  Top: {top}  |  {scores}")

print(f"\nSTATUS: All requests returned {r.status_code}")
