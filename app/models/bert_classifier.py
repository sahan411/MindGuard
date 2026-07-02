from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.constants import EMOTION_TOP_K_DEFAULT

logger = logging.getLogger(__name__)

GO_EMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Path where the trained model is expected after Colab training.
_MODEL_DIR = Path("models/bert_emotion/final_best_model")


class BertEmotionClassifier:
    """BERT multi-label emotion classifier with keyword fallback.

    Attempts to load a fine-tuned DistilBERT/BERT model saved by train_bert.py.
    Falls back to a lightweight keyword heuristic when the artifact is absent
    so the API remains functional during development and demo without weights.
    """

    def __init__(self) -> None:
        self.model_loaded = False
        self.model_error: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._label_list: List[str] = GO_EMOTIONS_LABELS

        try:
            self._try_load_model()
        except Exception as exc:
            self.model_error = str(exc)
            self.model_loaded = False
            logger.warning("BERT model not loaded, using keyword fallback: %s", exc)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load the trained model from disk."""
        config_path = _MODEL_DIR / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {_MODEL_DIR}. "
                "Train the model using notebooks/train_bert_colab.ipynb first."
            )

        # Load label map if available, otherwise derive from config.json
        label_map_path = _MODEL_DIR / "label_map.json"
        if label_map_path.exists():
            raw = json.loads(label_map_path.read_text(encoding="utf-8"))
            if isinstance(raw.get("labels"), list):
                self._label_list = raw["labels"]

        import torch  # noqa: F401  (ensures torch is available before model load)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
        self._model.eval()

        # If we didn't get labels from label_map.json, read from the model config
        if not self._label_list and hasattr(self._model.config, "id2label"):
            id2label = self._model.config.id2label
            self._label_list = [id2label[i] for i in sorted(id2label.keys())]

        self.model_loaded = True
        logger.info("BERT emotion model loaded successfully from %s", _MODEL_DIR)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, object]:
        """Return top emotion plus sorted emotion scores for the given text."""
        if not self.model_loaded or self._model is None:
            return self._keyword_fallback(text)

        try:
            return self._model_predict(text)
        except Exception as exc:
            logger.warning("BERT inference failed, using fallback: %s", exc)
            return self._keyword_fallback(text)

    def _model_predict(self, text: str) -> Dict[str, object]:
        """Run actual BERT inference with manual tokenization."""
        import torch

        inputs = self._tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        )
        # DistilBERT does not accept token_type_ids — remove if present
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Apply sigmoid for multi-label probabilities
        probs = torch.sigmoid(outputs.logits[0]).tolist()

        # Build label → score mapping
        id2label = self._model.config.id2label
        scored = [
            {"label": id2label[i], "score": probs[i]}
            for i in range(len(probs))
        ]
        sorted_results = sorted(scored, key=lambda x: x["score"], reverse=True)

        top_k = settings.emotion_top_k or EMOTION_TOP_K_DEFAULT
        top_emotions: List[Dict[str, object]] = [
            {
                "label": item["label"],
                "confidence": round(min(max(float(item["score"]), 0.0), 1.0), 4),
            }
            for item in sorted_results[:top_k]
        ]

        return {
            "emotions": top_emotions,
            "top_emotion": top_emotions[0]["label"] if top_emotions else "neutral",
        }

    def _keyword_fallback(self, text: str) -> Dict[str, object]:
        lowered = text.lower()
        scores: Dict[str, float] = {
            "sadness": 0.18, "neutral": 0.20, "fear": 0.14,
            "anger": 0.14, "joy": 0.14, "grief": 0.10, "optimism": 0.10,
        }

        sadness_markers = {"sad", "down", "low", "hopeless", "cry", "tired", "depressed", "empty"}
        fear_markers    = {"anxious", "afraid", "scared", "panic", "worry", "dread", "terror"}
        anger_markers   = {"angry", "mad", "hate", "furious", "rage", "frustrated"}
        joy_markers     = {"happy", "great", "good", "excited", "grateful", "wonderful", "love"}
        grief_markers   = {"grief", "loss", "mourn", "bereaved", "miss", "gone"}

        if any(t in lowered for t in sadness_markers):
            scores["sadness"] += 0.52; scores["neutral"] -= 0.10
        if any(t in lowered for t in fear_markers):
            scores["fear"] += 0.48; scores["neutral"] -= 0.08
        if any(t in lowered for t in anger_markers):
            scores["anger"] += 0.48; scores["neutral"] -= 0.08
        if any(t in lowered for t in joy_markers):
            scores["joy"] += 0.55; scores["neutral"] -= 0.10
        if any(t in lowered for t in grief_markers):
            scores["grief"] += 0.45; scores["neutral"] -= 0.08

        clipped = {label: min(max(v, 0.0), 1.0) for label, v in scores.items()}
        sorted_scores: List[Dict[str, object]] = sorted(
            [{"label": lbl, "confidence": conf} for lbl, conf in clipped.items()],
            key=lambda x: float(x["confidence"]),
            reverse=True,
        )

        top_k = settings.emotion_top_k or EMOTION_TOP_K_DEFAULT
        top_emotions = sorted_scores[:top_k]
        return {
            "emotions": top_emotions,
            "top_emotion": str(top_emotions[0]["label"]) if top_emotions else "neutral",
        }