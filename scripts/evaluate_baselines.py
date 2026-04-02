from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _keyword_baseline_predict(text: str) -> int:
    markers = {
        "suicide",
        "kill myself",
        "end my life",
        "self harm",
        "can't go on",
        "cannot go on",
        "want to die",
        "no reason to live",
    }
    lowered = text.lower()
    return int(any(marker in lowered for marker in markers))


def _prf(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation outputs for report baselines."
    )
    parser.add_argument(
        "--bert-summary",
        default="data/processed/bert_run_summary.json",
        help="Path to BERT run summary JSON.",
    )
    parser.add_argument(
        "--vae-summary",
        default="data/processed/vae_threshold_summary.json",
        help="Path to VAE threshold summary JSON.",
    )
    parser.add_argument(
        "--vae-errors",
        default="models/vae_crisis/reconstruction_errors.csv",
        help="Path to reconstruction error CSV emitted by VAE training.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/evaluation",
        help="Directory where evaluation tables/figures are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bert_summary = _load_json(Path(args.bert_summary))
    vae_summary = _load_json(Path(args.vae_summary))

    emotion_metrics = {
        "eval_metrics": bert_summary.get("eval_metrics", {}),
        "test_metrics": bert_summary.get("test_metrics", {}),
        "best_model_dir": bert_summary.get("best_model_dir", ""),
        "mlflow_run_id": bert_summary.get("mlflow_run_id", ""),
    }
    _write_json(output_dir / "emotion_metrics.json", emotion_metrics)

    errors_df = pd.read_csv(args.vae_errors)
    required_columns = {"text", "is_crisis", "predicted_crisis", "reconstruction_error"}
    missing = required_columns.difference(errors_df.columns)
    # Fail fast on schema drift so report numbers are reproducible across runs.
    if missing:
        raise ValueError(f"Missing required VAE errors columns: {sorted(missing)}")

    # Use a transparent lexical baseline to contextualize the learned detector's gains.
    errors_df["keyword_predicted"] = (
        errors_df["text"].astype(str).apply(_keyword_baseline_predict)
    )

    y_true = errors_df["is_crisis"].astype(int)
    vae_pred = errors_df["predicted_crisis"].astype(int)
    keyword_pred = errors_df["keyword_predicted"].astype(int)

    vae_metrics = _prf(y_true, vae_pred)
    keyword_metrics = _prf(y_true, keyword_pred)

    threshold_data = vae_summary.get("threshold", {})
    # Persist threshold provenance because this cutoff directly controls recall/precision tradeoff.
    threshold_rationale = {
        "threshold_percentile": threshold_data.get("percentile"),
        "threshold_value": threshold_data.get("value"),
        "validation_error_mean": threshold_data.get("validation_error_mean"),
        "validation_error_std": threshold_data.get("validation_error_std"),
        "justification": "Threshold is selected from validation reconstruction error percentile to prioritize recall-oriented risk detection.",
    }

    crisis_results = {
        "vae_metrics": vae_metrics,
        "keyword_metrics": keyword_metrics,
        "threshold_rationale": threshold_rationale,
    }
    _write_json(output_dir / "crisis_baseline_metrics.json", crisis_results)

    comparison_table = pd.DataFrame(
        [
            {"method": "vae", **vae_metrics},
            {"method": "keyword", **keyword_metrics},
        ]
    )
    comparison_table.to_csv(output_dir / "crisis_baseline_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["VAE", "Keyword"]
    f1_values = [vae_metrics["f1"], keyword_metrics["f1"]]
    recall_values = [vae_metrics["recall"], keyword_metrics["recall"]]

    # Plot both F1 and recall to expose the safety-critical tradeoff, not a single score only.
    x = range(len(methods))
    ax.bar([i - 0.15 for i in x], f1_values, width=0.3, label="F1")
    ax.bar([i + 0.15 for i in x], recall_values, width=0.3, label="Recall")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_title("Crisis Baseline Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "crisis_baseline_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Evaluation artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
