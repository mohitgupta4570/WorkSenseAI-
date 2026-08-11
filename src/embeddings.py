"""
FastText Embedding Module for WorkSense AI
------------------------------------------
This module satisfies the Vector Embedding requirement of the assignment.

It trains a FastText model on employee feedback text and generates:
1. A reusable FastText word embedding model
2. Document-level feedback vectors
3. Semantic similarity reports for HR-related keywords
4. 2D PCA visualization of feedback embeddings by Nine-Box category

Run from project root:
    python src/embeddings.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "fasttext"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings"

MODEL_PATH = MODEL_DIR / "fasttext_employee_feedback.model"
DOC_VECTOR_PATH = MODEL_DIR / "feedback_document_vectors.pkl"
EMBEDDING_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "feedback_embeddings.csv"
SIMILARITY_REPORT_PATH = OUTPUT_DIR / "fasttext_similarity_report.txt"
PCA_PLOT_PATH = OUTPUT_DIR / "feedback_embedding_pca.png"


HR_KEYWORDS = [
    "performance",
    "potential",
    "leadership",
    "communication",
    "team",
    "deadline",
    "initiative",
    "quality",
    "attendance",
    "growth",
    "manager",
    "productive",
    "consistent",
    "improvement",
    "responsibility",
]


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simple_tokenize(text: str) -> List[str]:
    """Tokenize already-cleaned text safely."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [tok.strip() for tok in text.split() if len(tok.strip()) > 1]
    return tokens


def load_feedback_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run: python src/preprocessing.py"
        )

    df = pd.read_csv(path)

    if "feedback_clean" not in df.columns:
        raise ValueError("Expected column 'feedback_clean' was not found in processed data.")

    df["feedback_clean"] = df["feedback_clean"].fillna("").astype(str)
    df["tokens"] = df["feedback_clean"].apply(simple_tokenize)
    df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)
    return df


def train_fasttext_model(tokenized_sentences: Iterable[List[str]]) -> FastText:
    """
    Train FastText embeddings.

    FastText is chosen over plain Word2Vec because it learns subword character n-grams.
    This helps when employee feedback contains rare words, spelling variations, or unseen terms.
    """
    model = FastText(
        sentences=list(tokenized_sentences),
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,          # Skip-gram generally works well for semantic similarity
        epochs=50,
        min_n=3,
        max_n=6,
        seed=42,
        bucket=50000,   # keeps saved model compact for student project packaging
    )
    return model


def sentence_vector(tokens: List[str], model: FastText) -> np.ndarray:
    """Create a document vector by averaging word vectors."""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0)


def build_document_vectors(df: pd.DataFrame, model: FastText) -> np.ndarray:
    vectors = np.vstack(df["tokens"].apply(lambda tokens: sentence_vector(tokens, model)).values)
    return vectors


def save_embedding_dataset(df: pd.DataFrame, vectors: np.ndarray) -> None:
    vector_columns = [f"ft_{i}" for i in range(vectors.shape[1])]
    vector_df = pd.DataFrame(vectors, columns=vector_columns)

    metadata_cols = [
        col
        for col in [
            "id",
            "person_name",
            "feedback",
            "feedback_clean",
            "label",
            "category_name",
            "performance_name",
            "potential_name",
            "data_type",
        ]
        if col in df.columns
    ]

    final_df = pd.concat([df[metadata_cols].reset_index(drop=True), vector_df], axis=1)
    final_df.to_csv(EMBEDDING_DATASET_PATH, index=False)


def generate_similarity_report(model: FastText) -> str:
    lines = []
    lines.append("FastText Semantic Similarity Report")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This report shows semantically related terms learned from employee feedback text."
    )
    lines.append("")

    for keyword in HR_KEYWORDS:
        lines.append(f"Keyword: {keyword}")
        if keyword in model.wv:
            similar_words = model.wv.most_similar(keyword, topn=8)
            for word, score in similar_words:
                lines.append(f"  - {word:<20} similarity={score:.4f}")
        else:
            # FastText can still infer vectors for unseen words, but most_similar requires vocab membership.
            lines.append("  - Keyword not directly present in vocabulary.")
        lines.append("")

    report = "\n".join(lines)
    SIMILARITY_REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def plot_pca_embeddings(df: pd.DataFrame, vectors: np.ndarray) -> None:
    """Create a 2D PCA plot of feedback document embeddings."""
    if len(df) < 2:
        return

    scaled_vectors = StandardScaler().fit_transform(vectors)
    pca_vectors = PCA(n_components=2, random_state=42).fit_transform(scaled_vectors)

    plot_df = pd.DataFrame(
        {
            "PC1": pca_vectors[:, 0],
            "PC2": pca_vectors[:, 1],
            "category": df.get("category_name", pd.Series(["Unknown"] * len(df))).values,
        }
    )

    plt.figure(figsize=(12, 8))

    categories = sorted(plot_df["category"].astype(str).unique())
    for category in categories:
        subset = plot_df[plot_df["category"].astype(str) == category]
        plt.scatter(subset["PC1"], subset["PC2"], label=category, alpha=0.65, s=35)

    plt.title("FastText Feedback Embeddings - PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()

    print("Loading processed employee feedback dataset...")
    df = load_feedback_dataset()
    print(f"Loaded {len(df)} feedback records.")

    print("Training FastText model...")
    model = train_fasttext_model(df["tokens"].tolist())
    model.save(str(MODEL_PATH))
    print(f"FastText model saved to: {MODEL_PATH}")

    print("Generating document-level feedback vectors...")
    vectors = build_document_vectors(df, model)
    joblib.dump(vectors, DOC_VECTOR_PATH)
    save_embedding_dataset(df, vectors)
    print(f"Document vectors saved to: {DOC_VECTOR_PATH}")
    print(f"Embedding dataset saved to: {EMBEDDING_DATASET_PATH}")

    print("Generating semantic similarity report...")
    report = generate_similarity_report(model)
    print(report[:1200])
    print(f"Full report saved to: {SIMILARITY_REPORT_PATH}")

    print("Creating PCA visualization...")
    plot_pca_embeddings(df, vectors)
    print(f"PCA plot saved to: {PCA_PLOT_PATH}")

    print("FastText embedding module completed successfully.")


if __name__ == "__main__":
    main()
