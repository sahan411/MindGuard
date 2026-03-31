from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

GO_EMOTIONS_LABELS: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


@dataclass(frozen=True)
class BertTrainConfig:
    dataset_source: str
    local_data_dir: str
    model_name: str
    max_length: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    num_epochs: int
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    seed: int
    checkpoint_dir: str
    best_model_dir: str
    save_total_limit: int
    eval_strategy: str
    logging_steps: int
    mlflow_tracking_uri: str
    run_name: str


def _parse_args() -> BertTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train BERT for GoEmotions multi-label classification."
    )
    parser.add_argument("--dataset-source", choices=["hf", "local"], default="hf")
    parser.add_argument("--local-data-dir", default="data/raw/go_emotions")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints/bert_emotion")
    parser.add_argument("--best-model-dir", default="models/bert_emotion/best_model")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch"], default="epoch")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--run-name", default="bert-goemotions")
    args = parser.parse_args()

    return BertTrainConfig(
        dataset_source=args.dataset_source,
        local_data_dir=args.local_data_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        best_model_dir=args.best_model_dir,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        run_name=args.run_name,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels_to_multihot(label_ids: List[int], num_labels: int) -> List[float]:
    vector = [0.0] * num_labels
    for label_id in label_ids:
        if not isinstance(label_id, int) or label_id < 0 or label_id >= num_labels:
            raise ValueError(f"Invalid label id: {label_id}")
        vector[label_id] = 1.0
    return vector


def _load_goemotions_dataset(config: BertTrainConfig) -> DatasetDict:
    if config.dataset_source == "hf":
        return load_dataset("google-research-datasets/go_emotions", "simplified")

    data_dir = Path(config.local_data_dir)
    data_files = {
        "train": str((data_dir / "train.jsonl").as_posix()),
        "validation": str((data_dir / "validation.jsonl").as_posix()),
        "test": str((data_dir / "test.jsonl").as_posix()),
    }
    return load_dataset("json", data_files=data_files)


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1)}


def _write_run_summary(summary_path: Path, payload: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config = _parse_args()
    _set_seed(config.seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    best_model_dir = Path(config.best_model_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_goemotions_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    num_labels = len(GO_EMOTIONS_LABELS)

    def preprocess(batch):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=config.max_length
        )
        encoded["labels"] = [
            _labels_to_multihot(label_ids, num_labels) for label_ids in batch["labels"]
        ]
        return encoded

    encoded_dataset = dataset.map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(GO_EMOTIONS_LABELS)},
        label2id={label: i for i, label in enumerate(GO_EMOTIONS_LABELS)},
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
        eval_strategy=config.eval_strategy,
        save_strategy=config.eval_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro_f1",
        greater_is_better=True,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("mindguard-bert-emotion")

    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.log_params(asdict(config))
        mlflow.log_param("num_labels", num_labels)

        train_output = trainer.train()
        eval_metrics = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
        test_metrics = trainer.evaluate(
            eval_dataset=encoded_dataset["test"], metric_key_prefix="test"
        )

        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in train_output.metrics.items()
                if isinstance(v, (int, float))
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in eval_metrics.items()
                if isinstance(v, (int, float))
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
        )

        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))

        label_map_path = best_model_dir / "label_map.json"
        label_map_path.write_text(
            json.dumps({"labels": GO_EMOTIONS_LABELS}, indent=2),
            encoding="utf-8",
        )

        run_summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run.info.run_id,
            "best_model_dir": str(best_model_dir.as_posix()),
            "checkpoint_dir": str(checkpoint_dir.as_posix()),
            "eval_metrics": eval_metrics,
            "test_metrics": test_metrics,
        }
        summary_path = Path("data/processed/bert_run_summary.json")
        _write_run_summary(summary_path, run_summary)
        mlflow.log_artifact(str(summary_path.as_posix()))


if __name__ == "__main__":
    main()
