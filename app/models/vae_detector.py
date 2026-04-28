from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.constants import CRISIS_THRESHOLD


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
    """VAE-based reconstruction-error detector.

    Attempts to load a trained VAE state dict and a TF-IDF vocabulary. If the
    expected artifacts are not present a FileNotFoundError is raised so callers
    can fall back to a safe alternative.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        stats_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        model_dir = Path(model_dir or "models/vae_crisis")
        state_path = model_dir / "vae_state_dict.pt"
        vocab_path = model_dir / "tfidf_vocabulary.json"

        if not state_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                f"VAE artifacts not found in {model_dir}. Expected files: {state_path.name}, {vocab_path.name}"
            )

        # load vocabulary
        with vocab_path.open("r", encoding="utf-8") as fh:
            vocab = json.load(fh)

        # sklearn's TfidfVectorizer accepts a vocabulary mapping token->index
        self.vectorizer = TfidfVectorizer(vocabulary=vocab)

        # load state dict and infer dimensions
        map_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(state_path, map_location=map_device)

        # infer dims from state dict keys
        enc_w = None
        mu_w = None
        for k, v in state.items():
            if k.endswith("encoder.0.weight"):
                enc_w = v
            if k.endswith("mu_layer.weight"):
                mu_w = v

        if enc_w is None or mu_w is None:
            raise RuntimeError("Saved VAE state dict missing expected keys")

        hidden_dim = int(enc_w.shape[0])
        input_dim = int(enc_w.shape[1])
        latent_dim = int(mu_w.shape[0])

        self.device = torch.device(map_device)
        self.model = TextVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # load threshold from stats if available, otherwise fall back
        threshold_value = None
        stats_path = Path(stats_path or "data/processed/vae_threshold_summary.json")
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as fh:
                    summary = json.load(fh)
                threshold_value = float(summary.get("threshold", {}).get("value", CRISIS_THRESHOLD))
            except Exception:
                threshold_value = CRISIS_THRESHOLD

        self.threshold = threshold_value if threshold_value is not None else CRISIS_THRESHOLD

    def _vectorize(self, texts: list[str]) -> np.ndarray:
        arr = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return arr

    def score(self, text: str) -> float:
        vec = self._vectorize([text])
        tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reconstructed, _, _ = self.model(tensor)
            per_row = torch.mean((reconstructed - tensor) ** 2, dim=1)
        return float(per_row.detach().cpu().numpy().item())

    def is_crisis(self, text: str) -> bool:
        return self.score(text) >= self.threshold
