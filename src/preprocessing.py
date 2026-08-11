"""
Module 1: Data loading and preprocessing for WorkSense AI.

This module prepares the employee feedback dataset for downstream NLP modules:
FastText embeddings, POS tagging, HMM tagging, parsing, LSTM language modeling,
and BERT classification.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


CATEGORY_MAP = {
    0: "Category 1: Risk",
    1: "Category 2: Average Performer",
    2: "Category 3: Solid Performer",
    3: "Category 4: Inconsistent Player",
    4: "Category 5: Core Player",
    5: "Category 6: High Performer",
    6: "Category 7: Potential Gem",
    7: "Category 8: High Potential",
    8: "Category 9: Star",
}

PERFORMANCE_MAP = {
    0: "Low Performance",
    1: "Medium Performance",
    2: "High Performance",
}

POTENTIAL_MAP = {
    0: "Low Potential",
    1: "Medium Potential",
    2: "High Potential",
}


def basic_clean_text(text: str) -> str:
    """Clean feedback text while preserving meaning for NLP models."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(RAW_DIR / "train_set.csv")
    val_df = pd.read_csv(RAW_DIR / "validation_set.csv")
    test_df = pd.read_csv(RAW_DIR / "test_set.csv")
    return train_df, val_df, test_df


def standardize_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Make column names and feature columns consistent across splits."""
    df = df.copy()

    if "updated" in df.columns and "adjusted" not in df.columns:
        df = df.rename(columns={"updated": "adjusted"})

    df["feedback"] = df["feedback"].fillna("")

    if "feedback_clean" not in df.columns:
        df["feedback_clean"] = df["feedback"].apply(basic_clean_text)
    else:
        df["feedback_clean"] = df["feedback_clean"].fillna(df["feedback"].apply(basic_clean_text))

    if "feedback_len" not in df.columns:
        df["feedback_len"] = df["feedback"].astype(str).apply(len)

    if "num_of_sent" not in df.columns:
        df["num_of_sent"] = df["feedback"].astype(str).apply(lambda x: max(1, x.count(".") + x.count("!") + x.count("?")))

    df["category_name"] = df["label"].map(CATEGORY_MAP)
    df["performance_name"] = df["performance_class"].map(PERFORMANCE_MAP)
    df["potential_name"] = df["potential_class"].map(POTENTIAL_MAP)
    df["data_type"] = split_name

    return df


def build_processed_dataset() -> pd.DataFrame:
    """Create one combined processed dataset and save split-wise CSV files."""
    train_df, val_df, test_df = load_data()

    train_df = standardize_columns(train_df, "train")
    val_df = standardize_columns(val_df, "validation")
    test_df = standardize_columns(test_df, "test")

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation_processed.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)
    combined_df.to_csv(PROCESSED_DIR / "combined_processed.csv", index=False)

    return combined_df


if __name__ == "__main__":
    df = build_processed_dataset()
    print("Processed dataset created successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nClass distribution:")
    print(df["category_name"].value_counts().sort_index())
