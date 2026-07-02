from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.constants import CRISIS_THRESHOLD

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR  = "models/vae_crisis_final"
_DEFAULT_SUMMARY    = "data/processed/vae_threshold_summary.json"

# VAE architecture — must match train_vae.py exactly.
_HIDDEN_DIM  = 256
_LATENT_DIM  = 32
_MAX_FEATURES = 4000


class TextVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class VAEDetector:
    """VAE-based crisis detector with keyword-baseline fallback.

    Loads the TF-IDF vectorizer vocabulary and the trained VAE state dict
    produced by scripts/train_vae.py. When the artifacts are absent, or
    loading fails for any reason, the detector falls back to a conservative
    keyword heuristic so the service stays functional for demo / development
    without weights.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        summary_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_loaded = False
        self.model_error:  Optional[str] = None
        self._model        = None
        self._vectorizer    = None
        self._device        = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._threshold     = CRISIS_THRESHOLD  # fallback default

        summary_file = Path(summary_path or _DEFAULT_SUMMARY)
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                self._threshold = float(data["threshold"]["value"])
            except Exception as exc:
                logger.warning("Could not read VAE threshold from summary: %s", exc)

        dir_path = Path(model_dir or _DEFAULT_MODEL_DIR)
        if dir_path.exists():
            self._try_load(dir_path)
        else:
            self.model_error = f"VAE model directory not found: {dir_path}"
            logger.warning("VAE model not found at %s — using keyword fallback.", dir_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self, dir_path: Path) -> None:
        try:
            vocab_path = dir_path / "tfidf_vocabulary.json"
            if not vocab_path.exists():
                raise FileNotFoundError(f"TF-IDF vocabulary not found: {vocab_path}")

            vocab: dict = json.loads(vocab_path.read_text(encoding="utf-8"))
            vectorizer = TfidfVectorizer(
                max_features=_MAX_FEATURES,
                vocabulary={token: int(idx) for token, idx in vocab.items()},
            )
            # Mark as fitted so transform() works without calling fit() on a corpus.
            vectorizer._validate_vocabulary()
            self._vectorizer = vectorizer

            state_dict_path = dir_path / "vae_state_dict.pt"
            if not state_dict_path.exists():
                raise FileNotFoundError(f"VAE state dict not found: {state_dict_path}")

            state = torch.load(str(state_dict_path), map_location="cpu", weights_only=True)

            # Prefer inferring dimensions from the saved weights themselves —
            # more robust than trusting hardcoded constants to stay in sync
            # with whatever train_vae.py was actually run with.
            enc_w = None
            mu_w = None
            for key, value in state.items():
                if key.endswith("encoder.0.weight"):
                    enc_w = value
                if key.endswith("mu_layer.weight"):
                    mu_w = value

            if enc_w is not None and mu_w is not None:
                hidden_dim = int(enc_w.shape[0])
                input_dim = int(enc_w.shape[1])
                latent_dim = int(mu_w.shape[0])
            else:
                # Fall back to the module-level constants if the state dict
                # doesn't expose the expected keys.
                hidden_dim, latent_dim = _HIDDEN_DIM, _LATENT_DIM
                input_dim = len(vocab)

            model_instance = TextVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            model_instance.load_state_dict(state)
            model_instance.to(self._device)
            model_instance.eval()

            self._model = model_instance
            self.model_loaded = True
            logger.info(
                "VAE crisis model loaded from %s (threshold=%.6f, device=%s)",
                dir_path, self._threshold, self._device,
            )

        except Exception as exc:
            self.model_error = str(exc)
            logger.warning("Failed to load VAE model: %s — using keyword fallback.", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Return reconstruction error for the input text.

        Higher error → more anomalous → more likely crisis.
        """
        if self.model_loaded and self._model is not None and self._vectorizer is not None:
            try:
                return self._vae_score(text)
            except Exception as exc:
                logger.warning("VAE inference failed (%s) — falling back.", exc)

        # Fallback: return a score below threshold for non-crisis keywords,
        # and above threshold for known crisis language.
        return self._keyword_score(text)

    def _vae_score(self, text: str) -> float:
        x = self._vectorizer.transform([text]).toarray().astype(np.float32)
        tensor = torch.tensor(x, device=self._device)
        with torch.no_grad():
            reconstructed, _, _ = self._model(tensor)
        error = float(torch.mean((reconstructed - tensor) ** 2).item())
        return error

    @staticmethod
    def _keyword_score(text: str) -> float:
        crisis_markers = {
            "suicide", "kill myself", "end my life", "self harm",
            "want to die", "no reason to live", "can't go on", "cannot go on",
        }
        lowered = text.lower()
        if any(m in lowered for m in crisis_markers):
            return 1.0   # guaranteed above any reasonable threshold
        return 0.0

    def is_crisis(self, text: str) -> bool:
        return self.score(text) >= self._threshold