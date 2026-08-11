"""
Phase 5: Assignment-style feedback classification for WorkSense AI.

This module adds the categories explicitly required by the assignment:
    - Positive Feedback
    - Negative Feedback
    - Neutral Feedback
    - Work Culture
    - Salary & Benefits
    - Career Growth
    - Management Issues

Because the provided MTurk dataset is a Nine-Box talent dataset and does not
include direct sentiment/workplace-issue labels, we create transparent weak
labels using NLP lexicons and domain keyword rules, then train TF-IDF based
classifiers that can be reused in the dashboard.

Run from project root:
    python src/feedback_classifier.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
MODEL_DIR = PROJECT_ROOT / "models" / "classifiers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "feedback"

SENTIMENT_CLASSES = ["Positive Feedback", "Neutral Feedback", "Negative Feedback"]
ISSUE_CLASSES = ["Work Culture", "Salary & Benefits", "Career Growth", "Management Issues"]

# Domain lexicon designed for employee-review style feedback.
POSITIVE_WORDS = {
    "excellent", "strong", "great", "good", "outstanding", "reliable", "consistent",
    "motivated", "motivates", "collaborative", "collaboration", "effective", "efficient",
    "leader", "leadership", "improved", "improves", "improvement", "potential",
    "skilled", "talented", "productive", "proactive", "dedicated", "valuable",
    "innovative", "successful", "success", "achieve", "achieves", "achieved",
    "supportive", "adaptable", "responsible", "dependable", "mentor", "mentors",
}

NEGATIVE_WORDS = {
    "poor", "weak", "late", "risk", "struggles", "struggle", "struggling", "declined",
    "decline", "misses", "missed", "inconsistent", "unreliable", "disrespectful",
    "argument", "argues", "complaint", "complaints", "liability", "low", "lack",
    "lacks", "difficult", "problem", "problems", "underperform", "underperformed",
    "substandard", "unsuitable", "negative", "conflict", "conflicts", "fails", "failed",
    "careless", "slow", "unmotivated", "needs", "concern", "concerns",
}

ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "Salary & Benefits": [
        "salary", "pay", "paid", "compensation", "benefit", "benefits", "bonus",
        "incentive", "reward", "rewards", "raise", "wage", "wages", "package",
        "insurance", "allowance", "allowances", "perk", "perks",
    ],
    "Career Growth": [
        "growth", "promotion", "promote", "promoted", "career", "learning", "learn",
        "training", "skill", "skills", "upskill", "development", "potential", "future",
        "mentor", "mentoring", "leadership", "leader", "improve", "improvement",
        "opportunity", "opportunities", "progress", "progressed", "advancement",
    ],
    "Management Issues": [
        "manager", "management", "supervisor", "lead", "leads", "leader", "leadership",
        "guidance", "feedback", "micromanage", "policy", "policies", "decision",
        "decisions", "reporting", "boss", "director", "communication", "communicate",
    ],
    "Work Culture": [
        "team", "teamwork", "culture", "environment", "coworker", "coworkers", "peer",
        "peers", "collaboration", "collaborative", "respect", "respectful", "disrespectful",
        "attendance", "punctual", "late", "deadline", "deadlines", "workload", "work",
        "office", "behavior", "attitude", "customers", "customer", "quality", "productivity",
    ],
}

CONCERN_KEYWORDS: Dict[str, List[str]] = {
    "Attendance / Punctuality": ["late", "attendance", "absent", "absence", "friday", "breaks", "punctual"],
    "Communication": ["communication", "communicate", "argues", "disrespectful", "feedback", "listen", "listening"],
    "Performance Quality": ["performance", "quality", "productive", "productivity", "deliver", "delivered", "results"],
    "Skill Gap": ["skill", "skills", "training", "learn", "learning", "master", "development"],
    "Leadership Potential": ["leader", "leadership", "potential", "mentor", "guidance", "future"],
    "Team Collaboration": ["team", "coworker", "coworkers", "peer", "peers", "collaboration", "collaborative"],
    "Career Growth": ["growth", "promotion", "career", "opportunity", "opportunities", "progress"],
    "Salary & Benefits": ["salary", "pay", "compensation", "benefit", "benefits", "bonus", "raise"],
}


def load_split(split: str) -> pd.DataFrame:
    """Load processed split if available; otherwise load raw split."""
    processed_path = PROCESSED_DIR / f"{split}_processed.csv"
    raw_path = RAW_DIR / f"{split}_set.csv"

    if processed_path.exists():
        return pd.read_csv(processed_path)
    if raw_path.exists():
        return pd.read_csv(raw_path)
    raise FileNotFoundError(f"Could not find data for split: {split}")


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def weak_sentiment_label(text: str, analyzer) -> str:
    """Create assignment sentiment label using VADER + domain lexicon adjustment."""
    tokens = tokenize(text)
    token_counts = Counter(tokens)

    if analyzer is not None:
        vader_score = analyzer.polarity_scores(str(text))["compound"]
    else:
        vader_score = 0.0
    positive_score = sum(token_counts[w] for w in POSITIVE_WORDS)
    negative_score = sum(token_counts[w] for w in NEGATIVE_WORDS)

    # Domain-adjusted sentiment score. VADER handles general tone; lexicon handles HR-specific words.
    adjusted_score = vader_score + (0.08 * positive_score) - (0.08 * negative_score)

    if adjusted_score >= 0.20:
        return "Positive Feedback"
    if adjusted_score <= -0.20:
        return "Negative Feedback"
    return "Neutral Feedback"



def derive_sentiment_label(row: pd.Series, analyzer) -> str:
    """Derive assignment sentiment label.

    The dataset contains performance_class values 0, 1 and 2. These are used as
    the primary weak supervision signal because they are human-provided HR
    labels and map naturally to negative, neutral and positive feedback tone.
    If performance_class is missing, the system falls back to text-based
    lexicon sentiment.
    """
    if "performance_class" in row and pd.notna(row["performance_class"]):
        try:
            performance = int(row["performance_class"])
            if performance == 2:
                return "Positive Feedback"
            if performance == 0:
                return "Negative Feedback"
            return "Neutral Feedback"
        except Exception:
            pass
    return weak_sentiment_label(str(row.get(TEXT_COL, "")), analyzer)

def weak_issue_label(text: str) -> str:
    """Create workplace issue label using transparent keyword scoring."""
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    scores: Dict[str, int] = {}

    for issue, keywords in ISSUE_KEYWORDS.items():
        scores[issue] = sum(token_counts[k] for k in keywords)

    best_issue, best_score = max(scores.items(), key=lambda item: item[1])

    # If no keyword appears, assign Work Culture because the dataset mainly contains workplace behavior feedback.
    if best_score == 0:
        return "Work Culture"
    return best_issue


def extract_concerns(text: str) -> str:
    """Return readable concern categories found in feedback text."""
    tokens = set(tokenize(text))
    concerns = []
    for concern, keywords in CONCERN_KEYWORDS.items():
        if any(keyword in tokens for keyword in keywords):
            concerns.append(concern)
    return "; ".join(concerns) if concerns else "General Employee Feedback"


def add_assignment_labels(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df["assignment_sentiment"] = df.apply(lambda row: derive_sentiment_label(row, analyzer), axis=1)
    df["assignment_issue_category"] = df[TEXT_COL].apply(weak_issue_label)
    df["extracted_concerns"] = df[TEXT_COL].apply(extract_concerns)
    return df


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8000,
                ),
            ),
            (
                "clf",
                MultinomialNB(alpha=0.3),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X: pd.Series, y: pd.Series, labels: List[str]) -> Dict[str, object]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        "classification_report": classification_report(y, preds, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
        "predictions": preds,
    }


def save_confusion_matrix(matrix: List[List[int]], labels: List[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, matrix[i][j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_distribution_plot(series: pd.Series, title: str, output_path: Path) -> None:
    counts = series.value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of Feedback Samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def top_features(model: Pipeline, class_names: List[str], top_n: int = 15) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []

    if hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_
    elif hasattr(clf, "coef_"):
        weights = clf.coef_
    else:
        return pd.DataFrame(columns=["class", "rank", "feature", "weight"])

    for class_index, class_name in enumerate(clf.classes_):
        top_indices = np.argsort(weights[class_index])[-top_n:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "weight": float(weights[class_index][idx]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading train, validation and test datasets...")
    train_df = add_assignment_labels(load_split("train"))
    val_df = add_assignment_labels(load_split("validation"))
    test_df = add_assignment_labels(load_split("test"))

    print("Training assignment sentiment classifier...")
    sentiment_model = build_classifier()
    sentiment_model.fit(train_df[TEXT_COL], train_df["assignment_sentiment"])

    print("Training workplace issue classifier...")
    issue_model = build_classifier()
    issue_model.fit(train_df[TEXT_COL], train_df["assignment_issue_category"])

    print("Evaluating classifiers...")
    sentiment_val = evaluate_model(sentiment_model, val_df[TEXT_COL], val_df["assignment_sentiment"], SENTIMENT_CLASSES)
    sentiment_test = evaluate_model(sentiment_model, test_df[TEXT_COL], test_df["assignment_sentiment"], SENTIMENT_CLASSES)

    issue_val = evaluate_model(issue_model, val_df[TEXT_COL], val_df["assignment_issue_category"], ISSUE_CLASSES)
    issue_test = evaluate_model(issue_model, test_df[TEXT_COL], test_df["assignment_issue_category"], ISSUE_CLASSES)

    # Save trained models.
    joblib.dump(sentiment_model, MODEL_DIR / "assignment_sentiment_classifier.joblib")
    joblib.dump(issue_model, MODEL_DIR / "workplace_issue_classifier.joblib")

    # Add predictions and save enhanced datasets.
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        df["predicted_assignment_sentiment"] = sentiment_model.predict(df[TEXT_COL])
        df["predicted_workplace_issue"] = issue_model.predict(df[TEXT_COL])
        df.to_csv(PROCESSED_DIR / f"{name}_assignment_classified.csv", index=False)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined["predicted_assignment_sentiment"] = sentiment_model.predict(combined[TEXT_COL])
    combined["predicted_workplace_issue"] = issue_model.predict(combined[TEXT_COL])
    combined.to_csv(PROCESSED_DIR / "combined_assignment_classified.csv", index=False)

    sample_cols = [
        "id", "person_name", TEXT_COL, "assignment_sentiment", "predicted_assignment_sentiment",
        "assignment_issue_category", "predicted_workplace_issue", "extracted_concerns",
    ]
    combined[sample_cols].head(75).to_csv(OUTPUT_DIR / "assignment_classification_sample_predictions.csv", index=False)

    # Save visuals.
    save_distribution_plot(
        combined["assignment_sentiment"],
        "Assignment Sentiment Distribution",
        OUTPUT_DIR / "assignment_sentiment_distribution.png",
    )
    save_distribution_plot(
        combined["assignment_issue_category"],
        "Workplace Issue Category Distribution",
        OUTPUT_DIR / "workplace_issue_distribution.png",
    )
    save_confusion_matrix(
        sentiment_test["confusion_matrix"],
        SENTIMENT_CLASSES,
        "Sentiment Classifier Confusion Matrix - Test Set",
        OUTPUT_DIR / "sentiment_confusion_matrix.png",
    )
    save_confusion_matrix(
        issue_test["confusion_matrix"],
        ISSUE_CLASSES,
        "Workplace Issue Classifier Confusion Matrix - Test Set",
        OUTPUT_DIR / "issue_confusion_matrix.png",
    )

    # Save model explanation features.
    top_features(sentiment_model, SENTIMENT_CLASSES).to_csv(OUTPUT_DIR / "sentiment_top_features.csv", index=False)
    top_features(issue_model, ISSUE_CLASSES).to_csv(OUTPUT_DIR / "issue_top_features.csv", index=False)

    metrics = {
        "sentiment_validation": {k: v for k, v in sentiment_val.items() if k not in {"predictions"}},
        "sentiment_test": {k: v for k, v in sentiment_test.items() if k not in {"predictions"}},
        "issue_validation": {k: v for k, v in issue_val.items() if k not in {"predictions"}},
        "issue_test": {k: v for k, v in issue_test.items() if k not in {"predictions"}},
        "note": (
            "Sentiment labels are derived from performance_class when available; workplace issue labels are weak labels created from domain keyword rules because the original dataset contains Nine-Box talent labels, not direct assignment issue labels."
        ),
    }
    with open(OUTPUT_DIR / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(OUTPUT_DIR / "classification_summary.txt", "w", encoding="utf-8") as f:
        f.write("WorkSense AI - Phase 5 Classification Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Implemented assignment-required classification categories:\n")
        f.write("- Positive Feedback\n- Negative Feedback\n- Neutral Feedback\n")
        f.write("- Work Culture\n- Salary & Benefits\n- Career Growth\n- Management Issues\n\n")
        f.write("Important dataset note:\n")
        f.write("The uploaded dataset is a Nine-Box employee talent dataset. It does not contain direct sentiment or workplace-issue labels.\n")
        f.write("Therefore, this phase uses transparent weak supervision. Sentiment is derived primarily from human-provided performance_class labels, while issue categories are derived from domain keyword rules. The generated labels are then used to train reusable ML classifiers.\n\n")
        f.write("Sentiment Test Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {sentiment_test['accuracy']:.4f}\n")
        f.write(f"Precision Macro: {sentiment_test['precision_macro']:.4f}\n")
        f.write(f"Recall Macro: {sentiment_test['recall_macro']:.4f}\n")
        f.write(f"F1 Macro: {sentiment_test['f1_macro']:.4f}\n\n")
        f.write(sentiment_test["classification_report"] + "\n")
        f.write("\nWorkplace Issue Test Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {issue_test['accuracy']:.4f}\n")
        f.write(f"Precision Macro: {issue_test['precision_macro']:.4f}\n")
        f.write(f"Recall Macro: {issue_test['recall_macro']:.4f}\n")
        f.write(f"F1 Macro: {issue_test['f1_macro']:.4f}\n\n")
        f.write(issue_test["classification_report"] + "\n")
        f.write("\nGenerated files are available in outputs/classification/.\n")

    print("\nPhase 5 complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
