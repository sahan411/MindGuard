"""Download and prepare a crisis-labeled dataset for VAE training.

Downloads the 'vibhorag101/phr_suicidal_data' dataset from HuggingFace,
which contains Reddit posts labeled as suicidal (crisis) or non-suicidal.
Outputs a CSV with columns: text, label  (label=1 for crisis, 0 for non-crisis).

Usage:
    python scripts/prepare_crisis_dataset.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUTPUT_PATH = Path("data/processed/crisis_labeled_text.csv")
MAX_ROWS_PER_CLASS = 5000  # keep dataset balanced and manageable


def normalize(text: str) -> str:
    """Lightweight normalisation — same logic as nlp_preprocessor."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    # ourafla/Mental-Health_Text-Classification_Dataset has 4 classes:
    # Suicidal, Depression, Anxiety, Normal
    # We treat "Suicidal" as crisis=1, everything else as non-crisis=0.
    print("Downloading crisis dataset from HuggingFace...")
    # Using pandas directly to read the specific CSV file to bypass huggingface dataset builder schema mismatch issues
    df = pd.read_csv("hf://datasets/ourafla/Mental-Health_Text-Classification_Dataset/mental_heath_unbanlanced.csv")

    print(f"Raw rows: {len(df):,}")
    print(f"Columns:  {list(df.columns)}")
    print(f"Label distribution:\n{df.iloc[:, -1].value_counts()}\n")

    # The dataset has columns: text (or statement) + label
    text_col  = "text"      if "text"      in df.columns else \
                "statement" if "statement" in df.columns else df.columns[0]
    label_col = "label"     if "label"     in df.columns else df.columns[-1]

    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "label"]

    unique_labels = df["label"].unique()
    print(f"Unique label values: {unique_labels}")

    # Map "Suicidal" → 1 (crisis), all others (Normal, Anxiety, Depression) → 0
    CRISIS_LABELS = {"suicidal", "suicide", "1", "true"}
    if df["label"].dtype == object:
        df["label"] = df["label"].apply(
            lambda v: 1 if str(v).strip().lower() in CRISIS_LABELS else 0
        )
    else:
        df["label"] = df["label"].astype(int)

    print(f"After mapping — crisis: {df['label'].sum():,}, non-crisis: {(df['label'] == 0).sum():,}")

    # Normalise text
    df["text"] = df["text"].astype(str).apply(normalize)
    df = df[df["text"].str.len() >= 10].copy()  # drop near-empty rows

    # Balance: cap each class at MAX_ROWS_PER_CLASS
    crisis_df     = df[df["label"] == 1].sample(
        min(MAX_ROWS_PER_CLASS, df["label"].sum()), random_state=42
    )
    non_crisis_df = df[df["label"] == 0].sample(
        min(MAX_ROWS_PER_CLASS, (df["label"] == 0).sum()), random_state=42
    )
    balanced = pd.concat([crisis_df, non_crisis_df]).sample(frac=1, random_state=42)
    balanced = balanced.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved {len(balanced):,} rows to {OUTPUT_PATH}")
    print(f"   Crisis rows:     {balanced['label'].sum():,}")
    print(f"   Non-crisis rows: {(balanced['label'] == 0).sum():,}")
    print("\nNext step → run VAE training:")
    print("  python scripts/train_vae.py")


if __name__ == "__main__":
    main()
