"""
Module 1B: Exploratory Data Analysis for WorkSense AI.

Creates plots required for the project report and dashboard foundation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "combined_processed.csv"
    if not path.exists():
        raise FileNotFoundError("Run preprocessing.py first to create processed data.")
    return pd.read_csv(path)


def plot_category_distribution(df: pd.DataFrame) -> None:
    counts = df["category_name"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Nine-Box Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.png", dpi=300)
    plt.close()


def plot_feedback_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    df["feedback_len"].plot(kind="hist", bins=30)
    plt.title("Feedback Length Distribution")
    plt.xlabel("Feedback Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feedback_length_distribution.png", dpi=300)
    plt.close()


def create_wordcloud(df: pd.DataFrame) -> None:
    text = " ".join(df["feedback_clean"].fillna("").astype(str).tolist())
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Employee Feedback Word Cloud")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feedback_wordcloud.png", dpi=300)
    plt.close()


def generate_summary(df: pd.DataFrame) -> None:
    summary = {
        "total_reviews": len(df),
        "total_categories": df["label"].nunique(),
        "avg_feedback_length": round(df["feedback_len"].mean(), 2),
        "avg_sentences_per_feedback": round(df["num_of_sent"].mean(), 2),
    }

    lines = ["WorkSense AI - Dataset Summary", "=" * 35]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")

    lines.append("\nCategory Distribution:")
    lines.append(df["category_name"].value_counts().sort_index().to_string())

    (OUTPUT_DIR / "dataset_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eda() -> None:
    df = load_processed_data()
    plot_category_distribution(df)
    plot_feedback_length_distribution(df)
    create_wordcloud(df)
    generate_summary(df)
    print("EDA completed. Check the outputs folder.")


if __name__ == "__main__":
    run_eda()
