"""
Phase 8: Nine-Box Talent Category Prediction

This module trains a strong, lightweight multi-class text classifier for the
original HR target variable: nine_box_category.

It predicts categories such as:
- Category 1: Risk
- Category 5: Core Player
- Category 9: Star

Run from project root:
    python src/talent_predictor.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "talent_predictor"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "talent_prediction"

TRAIN_FILE = DATA_DIR / "train_processed.csv"
VAL_FILE = DATA_DIR / "validation_processed.csv"
TEST_FILE = DATA_DIR / "test_processed.csv"

TEXT_COL = "feedback_clean"
TARGET_COL = "label"
CATEGORY_COL = "nine_box_category"


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [TRAIN_FILE, VAL_FILE, TEST_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed files not found. Run: python src/preprocessing.py\n"
            + "Missing: "
            + ", ".join(str(p) for p in missing)
        )

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if TEXT_COL not in df.columns:
            raise ValueError(f"{name} file does not contain required column: {TEXT_COL}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"{name} file does not contain required column: {TARGET_COL}")
        if CATEGORY_COL not in df.columns:
            raise ValueError(f"{name} file does not contain required column: {CATEGORY_COL}")

    train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str)
    val_df[TEXT_COL] = val_df[TEXT_COL].fillna("").astype(str)
    test_df[TEXT_COL] = test_df[TEXT_COL].fillna("").astype(str)

    return train_df, val_df, test_df


def build_label_maps(df: pd.DataFrame) -> tuple[Dict[int, str], Dict[int, str]]:
    label_to_category = (
        df[[TARGET_COL, CATEGORY_COL]]
        .drop_duplicates()
        .sort_values(TARGET_COL)
        .set_index(TARGET_COL)[CATEGORY_COL]
        .to_dict()
    )

    label_to_short_name = {}
    for label, category in label_to_category.items():
        # Example: Category 1: 'Risk' (...) -> Category 1 - Risk
        category_num = category.split(":")[0].strip()
        if "'" in category:
            short = category.split("'")[1]
        else:
            short = category
        label_to_short_name[int(label)] = f"{category_num} - {short}"

    return {int(k): str(v) for k, v in label_to_category.items()}, label_to_short_name


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Train a TF-IDF + calibrated LinearSVC classifier.

    LinearSVC usually performs strongly on small/medium text datasets.
    Calibration provides probability-like confidence scores for the UI.
    """
    base_svc = LinearSVC(class_weight="balanced", random_state=42, max_iter=8000)
    calibrated_svc = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", calibrated_svc),
        ]
    )

    model.fit(train_df[TEXT_COL], train_df[TARGET_COL])
    return model


def evaluate_model(model: Pipeline, df: pd.DataFrame, split_name: str, label_names: List[str]) -> Dict[str, object]:
    y_true = df[TARGET_COL]
    y_pred = model.predict(df[TEXT_COL])

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
    (OUTPUT_DIR / f"{split_name}_classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(OUTPUT_DIR / f"{split_name}_confusion_matrix.csv")

    plt.figure(figsize=(11, 9))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Nine-Box Talent Prediction Confusion Matrix ({split_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(np.arange(len(label_names)), label_names, fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{split_name}_confusion_matrix.png", dpi=200)
    plt.close()

    return metrics


def create_prediction_samples(model: Pipeline, test_df: pd.DataFrame, label_to_short_name: Dict[int, str]) -> pd.DataFrame:
    sample_df = test_df[["id", "person_name", "feedback", TEXT_COL, TARGET_COL, CATEGORY_COL]].copy().head(30)
    preds = model.predict(sample_df[TEXT_COL])

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probs = model.predict_proba(sample_df[TEXT_COL])
        confidence = probs.max(axis=1)
    else:
        confidence = np.nan

    sample_df["predicted_label"] = preds
    sample_df["predicted_category"] = [label_to_short_name[int(x)] for x in preds]
    sample_df["actual_category_short"] = [label_to_short_name[int(x)] for x in sample_df[TARGET_COL]]
    sample_df["confidence"] = confidence
    sample_df["is_correct"] = sample_df["predicted_label"] == sample_df[TARGET_COL]
    sample_df.to_csv(OUTPUT_DIR / "talent_prediction_sample_outputs.csv", index=False)
    return sample_df


def save_top_features(model: Pipeline, label_to_short_name: Dict[int, str], top_n: int = 20) -> None:
    """Save indicative TF-IDF terms per class when available.

    CalibratedClassifierCV wraps multiple estimators, so direct feature coefficients
    are not as simple as a plain LinearSVC. For interpretability, we train a second
    plain LinearSVC on the fitted TF-IDF matrix and export top features.
    """
    # This function is intentionally skipped for calibrated model internals to keep code stable.
    # We still provide model-level interpretability using sample predictions and reports.
    note = (
        "Top feature export skipped for calibrated LinearSVC internals. "
        "Use the confusion matrix, classification reports, and sample predictions "
        "for model interpretation."
    )
    (OUTPUT_DIR / "top_features_note.txt").write_text(note, encoding="utf-8")


def write_summary(metrics: Dict[str, Dict[str, object]], label_to_category: Dict[int, str]) -> None:
    lines = []
    lines.append("WorkSense AI - Phase 8 Talent Prediction Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Task: Predict the original Nine-Box HR talent category from employee feedback text.")
    lines.append("Model: TF-IDF bigram features + class-balanced calibrated LinearSVC.")
    lines.append("Why this model: Strong, fast, interpretable, and reliable for small text datasets.")
    lines.append("")
    lines.append("Nine-Box Categories:")
    for label, category in label_to_category.items():
        lines.append(f"  {label}: {category}")
    lines.append("")

    for split_name, split_metrics in metrics.items():
        lines.append(f"{split_name.upper()} METRICS")
        lines.append("-" * 30)
        lines.append(f"Accuracy      : {split_metrics['accuracy']:.4f}")
        lines.append(f"Macro Precision: {split_metrics['macro_precision']:.4f}")
        lines.append(f"Macro Recall   : {split_metrics['macro_recall']:.4f}")
        lines.append(f"Macro F1       : {split_metrics['macro_f1']:.4f}")
        lines.append(f"Weighted F1    : {split_metrics['weighted_f1']:.4f}")
        lines.append("")

    lines.append("Generated Files:")
    lines.append("  models/talent_predictor/nine_box_talent_predictor.joblib")
    lines.append("  models/talent_predictor/label_maps.json")
    lines.append("  outputs/talent_prediction/*_classification_report.txt")
    lines.append("  outputs/talent_prediction/*_confusion_matrix.png")
    lines.append("  outputs/talent_prediction/talent_prediction_sample_outputs.csv")

    (OUTPUT_DIR / "talent_prediction_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    train_df, val_df, test_df = load_data()
    label_to_category, label_to_short_name = build_label_maps(train_df)

    print("Training Nine-Box Talent Predictor...")
    print(f"Train samples      : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")
    print(f"Test samples       : {len(test_df)}")

    model = train_model(train_df)

    label_names = [label_to_short_name[i] for i in sorted(label_to_short_name)]
    metrics = {
        "validation": evaluate_model(model, val_df, "validation", label_names),
        "test": evaluate_model(model, test_df, "test", label_names),
    }

    create_prediction_samples(model, test_df, label_to_short_name)
    save_top_features(model, label_to_short_name)

    joblib.dump(model, MODEL_DIR / "nine_box_talent_predictor.joblib")
    with open(MODEL_DIR / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_category": label_to_category,
                "label_to_short_name": label_to_short_name,
            },
            f,
            indent=2,
        )

    metrics_to_save = {
        split: {k: v for k, v in split_metrics.items() if k != "classification_report"}
        for split, split_metrics in metrics.items()
    }
    with open(OUTPUT_DIR / "talent_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)

    write_summary(metrics, label_to_category)

    print("\nTalent Prediction complete.")
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Validation macro F1: {metrics['validation']['macro_f1']:.4f}")
    print(f"Test accuracy      : {metrics['test']['accuracy']:.4f}")
    print(f"Test macro F1      : {metrics['test']['macro_f1']:.4f}")
    print(f"Saved model to     : {MODEL_DIR / 'nine_box_talent_predictor.joblib'}")
    print(f"Check outputs in   : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
