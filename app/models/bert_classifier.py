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

_DEFAULT_MODEL_DIR = "models/bert_emotion/final_best_model"


class BertEmotionClassifier:
    """BERT multi-label emotion classifier with keyword fallback.

    Attempts to load a fine-tuned DistilBERT/BERT model saved by train_bert.py.
    Falls back to a lightweight keyword heuristic when the artifact is absent
    so the API remains functional during development and demo without weights.
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_loaded = False
        self.model_error: Optional[str] = None
        self._pipeline = None
        self._labels: List[str] = GO_EMOTIONS_LABELS

        dir_path = Path(model_dir or _DEFAULT_MODEL_DIR)
        if dir_path.exists():
            self._try_load(dir_path)
        else:
            self.model_error = f"Model directory not found: {dir_path}"
            logger.warning("BERT model not found at %s — using keyword fallback.", dir_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self, dir_path: Path) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore

            self._pipeline = hf_pipeline(
                task="text-classification",
                model=str(dir_path),
                tokenizer=str(dir_path),
                top_k=None,          # return scores for all labels
                truncation=True,
                max_length=128,
            )

            # Refresh label list from saved label_map if present.
            label_map_path = dir_path / "label_map.json"
            if label_map_path.exists():
                data = json.loads(label_map_path.read_text(encoding="utf-8"))
                if isinstance(data.get("labels"), list):
                    self._labels = data["labels"]

            self.model_loaded = True
            logger.info("BERT emotion model loaded from %s", dir_path)
        except Exception as exc:
            self.model_error = str(exc)
            logger.warning("Failed to load BERT model: %s — using keyword fallback.", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, object]:
        """Return top-k emotions and the highest-confidence label."""
        if self.model_loaded and self._pipeline is not None:
            try:
                return self._bert_predict(text)
            except Exception as exc:
                logger.warning("BERT inference failed (%s) — using keyword fallback.", exc)

        return self._keyword_fallback(text)

    def _bert_predict(self, text: str) -> Dict[str, object]:
        raw = self._pipeline(text)
        # HF text-classification with top_k=None returns List[List[Dict]]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            scores_list = raw[0]
        elif isinstance(raw, list):
            scores_list = raw
        else:
            scores_list = []

        sorted_scores: List[Dict[str, object]] = sorted(
            [{"label": item["label"], "confidence": float(item["score"])}
             for item in scores_list if "label" in item and "score" in item],
            key=lambda x: float(x["confidence"]),
            reverse=True,
        )

        top_k = settings.emotion_top_k or EMOTION_TOP_K_DEFAULT
        top_emotions = sorted_scores[:top_k]
        top_emotion = str(top_emotions[0]["label"]) if top_emotions else "neutral"

        return {"emotions": top_emotions, "top_emotion": top_emotion}

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
