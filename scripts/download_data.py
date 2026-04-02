from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from datasets import DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    config_name: str | None
    required_columns: List[str]
    validator: Callable[[pd.DataFrame], None]


def _validate_non_empty(df: pd.DataFrame, dataset_name: str) -> None:
    if df.empty:
        raise ValueError(f"{dataset_name}: split is empty")


def _validate_go_emotions(df: pd.DataFrame) -> None:
    _validate_non_empty(df, "go_emotions")

    if "id" not in df.columns and "comment_id" not in df.columns:
        raise ValueError("go_emotions: expected either 'id' or 'comment_id' column")

    invalid_rows: List[int] = []
    for idx, labels in enumerate(df["labels"]):
        if not hasattr(labels, "__iter__"):
            invalid_rows.append(idx)
            continue

        labels_list = list(labels)
        if len(labels_list) == 0:
            invalid_rows.append(idx)
            continue

        if not all(
            isinstance(label, Integral) and 0 <= int(label) <= 27
            for label in labels_list
        ):
            invalid_rows.append(idx)

    if invalid_rows:
        sample = invalid_rows[:5]
        raise ValueError(
            "go_emotions: found invalid labels in rows "
            f"{sample} (expected non-empty int labels in [0, 27])"
        )


def _validate_amod(df: pd.DataFrame) -> None:
    _validate_non_empty(df, "amod_mh_counseling")
    empty_context = (
        df["Context"].isna().sum() + (df["Context"].astype(str).str.strip() == "").sum()
    )
    empty_response = (
        df["Response"].isna().sum()
        + (df["Response"].astype(str).str.strip() == "").sum()
    )
    if empty_context > 0 or empty_response > 0:
        raise ValueError(
            "amod_mh_counseling: empty Context/Response rows found "
            f"(context={empty_context}, response={empty_response})"
        )


def _validate_esconv(df: pd.DataFrame) -> None:
    _validate_non_empty(df, "esconv")


DATASETS: Dict[str, DatasetSpec] = {
    "go_emotions": DatasetSpec(
        dataset_id="google-research-datasets/go_emotions",
        config_name="simplified",
        required_columns=["text", "labels"],
        validator=_validate_go_emotions,
    ),
    "amod_mh_counseling": DatasetSpec(
        dataset_id="Amod/mental_health_counseling_conversations",
        config_name=None,
        required_columns=["Context", "Response"],
        validator=_validate_amod,
    ),
    "esconv": DatasetSpec(
        dataset_id="thu-coai/esconv",
        config_name=None,
        required_columns=[],
        validator=_validate_esconv,
    ),
}


def _ensure_required_columns(
    df: pd.DataFrame, required_columns: List[str], dataset_name: str
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")


def _load_dataset(spec: DatasetSpec) -> DatasetDict:
    if spec.config_name:
        return load_dataset(spec.dataset_id, spec.config_name)
    return load_dataset(spec.dataset_id)


def _export_split(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)


def _process_dataset(dataset_key: str, raw_root: Path) -> Dict[str, object]:
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset key: {dataset_key}. Available keys: {sorted(DATASETS.keys())}"
        )

    spec = DATASETS[dataset_key]
    dataset_dict = _load_dataset(spec)
    dataset_dir = raw_root / dataset_key
    split_summaries: Dict[str, object] = {}

    for split_name, split_data in dataset_dict.items():
        df = split_data.to_pandas()
        dropped_empty_rows = 0

        if dataset_key == "amod_mh_counseling":
            before_rows = len(df)
            df = df.dropna(subset=["Context", "Response"]).copy()
            df = df[
                (df["Context"].astype(str).str.strip() != "")
                & (df["Response"].astype(str).str.strip() != "")
            ].copy()
            dropped_empty_rows = before_rows - len(df)

        _ensure_required_columns(df, spec.required_columns, dataset_key)
        spec.validator(df)

        output_path = dataset_dir / f"{split_name}.jsonl"
        _export_split(df, output_path)

        split_summaries[split_name] = {
            "rows": len(df),
            "dropped_empty_rows": dropped_empty_rows,
            "columns": list(df.columns),
            "output": str(output_path.as_posix()),
        }

    return {
        "dataset_id": spec.dataset_id,
        "config_name": spec.config_name,
        "splits": split_summaries,
    }


def _write_manifest(summary: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate project datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["go_emotions"],
        help="Dataset keys to download. Available: go_emotions, amod_mh_counseling, esconv",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for raw downloaded/validated splits.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/dataset_manifest.json",
        help="Path to write dataset manifest summary.",
    )
    parser.add_argument(
        "--glove", action="store_true", help="Print GloVe download reminder."
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    manifest_path = Path(args.manifest)

    summary: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for dataset_key in args.datasets:
        dataset_summary = _process_dataset(dataset_key=dataset_key, raw_root=raw_root)
        summary["datasets"][dataset_key] = dataset_summary

    if args.glove:
        print(
            "Reminder: download GloVe vectors into data/glove/ and record source/version."
        )

    _write_manifest(summary=summary, output_path=manifest_path)
    print(f"Dataset manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
