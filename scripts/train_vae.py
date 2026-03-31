from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class VAETrainConfig:
    data_path: str
    text_column: str
    label_column: str
    positive_labels: str
    max_features: int
    hidden_dim: int
    latent_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    validation_ratio: float
    threshold_percentile: float
    seed: int
    model_output_dir: str
    stats_output_path: str
    mlflow_tracking_uri: str
    run_name: str


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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


def _parse_args() -> VAETrainConfig:
    parser = argparse.ArgumentParser(
        description="Train VAE crisis detector on non-crisis examples only."
    )
    parser.add_argument("--data-path", default="data/processed/crisis_labeled_text.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-labels", default="1,true,crisis,suicidal")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output-dir", default="models/vae_crisis")
    parser.add_argument(
        "--stats-output-path", default="data/processed/vae_threshold_summary.json"
    )
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--run-name", default="vae-crisis-detector")
    args = parser.parse_args()

    return VAETrainConfig(
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        positive_labels=args.positive_labels,
        max_features=args.max_features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        threshold_percentile=args.threshold_percentile,
        seed=args.seed,
        model_output_dir=args.model_output_dir,
        stats_output_path=args.stats_output_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        run_name=args.run_name,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported data format: {suffix}. Use .csv, .jsonl, or .parquet"
    )


def _is_positive_label(raw_label: object, positive_tokens: set[str]) -> bool:
    if isinstance(raw_label, bool):
        return raw_label
    text = str(raw_label).strip().lower()
    return text in positive_tokens


def _estimate_threshold(errors: Sequence[float], percentile: float) -> float:
    if len(errors) == 0:
        raise ValueError("Cannot estimate threshold from empty error sequence")
    return float(np.percentile(np.array(errors, dtype=np.float32), percentile))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _iterate_batches(data: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]


def _reconstruction_errors(
    model: TextVAE, data: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    errors: List[np.ndarray] = []
    with torch.no_grad():
        for batch in _iterate_batches(data, batch_size=512):
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            reconstructed, _, _ = model(tensor)
            per_row = torch.mean((reconstructed - tensor) ** 2, dim=1)
            errors.append(per_row.detach().cpu().numpy())
    return np.concatenate(errors, axis=0) if errors else np.array([], dtype=np.float32)


def main() -> None:
    config = _parse_args()
    _set_seed(config.seed)

    positive_tokens = {
        token.strip().lower() for token in config.positive_labels.split(",")
    }
    df = _load_dataframe(Path(config.data_path))

    if config.text_column not in df.columns or config.label_column not in df.columns:
        raise ValueError(
            f"Input data must contain '{config.text_column}' and '{config.label_column}' columns"
        )

    df = df[[config.text_column, config.label_column]].dropna().copy()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column] != ""].copy()
    df["is_crisis"] = df[config.label_column].apply(
        lambda value: _is_positive_label(value, positive_tokens)
    )

    non_crisis = df[df["is_crisis"] == 0].copy()
    crisis = df[df["is_crisis"] == 1].copy()
    if len(non_crisis) < 20:
        raise ValueError("Need at least 20 non-crisis samples for stable VAE training")

    shuffled = non_crisis.sample(frac=1.0, random_state=config.seed)
    split_index = int(len(shuffled) * (1.0 - config.validation_ratio))
    split_index = max(1, min(split_index, len(shuffled) - 1))
    train_nc = shuffled.iloc[:split_index]
    val_nc = shuffled.iloc[split_index:]

    vectorizer = TfidfVectorizer(max_features=config.max_features)
    x_train = (
        vectorizer.fit_transform(train_nc[config.text_column])
        .toarray()
        .astype(np.float32)
    )
    x_val = (
        vectorizer.transform(val_nc[config.text_column]).toarray().astype(np.float32)
    )
    x_full = vectorizer.transform(df[config.text_column]).toarray().astype(np.float32)

    input_dim = x_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextVAE(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for _ in range(config.epochs):
        for batch in _iterate_batches(x_train, config.batch_size):
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            reconstructed, mu, logvar = model(tensor)
            recon_loss = torch.mean((reconstructed - tensor) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val_errors = _reconstruction_errors(model, x_val, device)
    threshold = _estimate_threshold(val_errors, config.threshold_percentile)

    full_errors = _reconstruction_errors(model, x_full, device)
    predictions = (full_errors >= threshold).astype(int)
    truth = df["is_crisis"].to_numpy(dtype=int)
    metrics = _binary_metrics(truth, predictions)

    model_output_dir = Path(config.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_dir / "vae_state_dict.pt")

    vectorizer_path = model_output_dir / "tfidf_vocabulary.json"
    vocabulary_serializable = {
        token: int(index) for token, index in vectorizer.vocabulary_.items()
    }
    vectorizer_path.write_text(
        json.dumps(vocabulary_serializable, indent=2), encoding="utf-8"
    )

    errors_path = model_output_dir / "reconstruction_errors.csv"
    pd.DataFrame(
        {
            "text": df[config.text_column].to_list(),
            "is_crisis": truth.tolist(),
            "reconstruction_error": full_errors.tolist(),
            "predicted_crisis": predictions.tolist(),
        }
    ).to_csv(errors_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": config.data_path,
        "counts": {
            "total": int(len(df)),
            "non_crisis_total": int(len(non_crisis)),
            "crisis_total": int(len(crisis)),
            "non_crisis_train": int(len(train_nc)),
            "non_crisis_validation": int(len(val_nc)),
        },
        "threshold": {
            "percentile": config.threshold_percentile,
            "value": threshold,
            "validation_error_mean": (
                float(np.mean(val_errors)) if len(val_errors) else 0.0
            ),
            "validation_error_std": (
                float(np.std(val_errors)) if len(val_errors) else 0.0
            ),
        },
        "metrics": metrics,
        "artifacts": {
            "model_state": str((model_output_dir / "vae_state_dict.pt").as_posix()),
            "vectorizer_vocabulary": str(vectorizer_path.as_posix()),
            "reconstruction_errors": str(errors_path.as_posix()),
        },
    }

    stats_output_path = Path(config.stats_output_path)
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("mindguard-vae-crisis")
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_metrics(metrics)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_artifact(str(stats_output_path.as_posix()))
        mlflow.log_artifact(str((model_output_dir / "vae_state_dict.pt").as_posix()))
        mlflow.log_artifact(str(vectorizer_path.as_posix()))
        mlflow.log_artifact(str(errors_path.as_posix()))

    print(f"VAE summary written to: {stats_output_path}")


if __name__ == "__main__":
    main()
