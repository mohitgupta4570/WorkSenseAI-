"""
Phase 8B: BERT / DistilBERT Nine-Box Talent Prediction

Goal:
    Fine-tune a Transformer model to predict the original `nine_box_category`
    from employee feedback text.

Main output classes:
    Category 1 - Risk
    Category 2 - Average Performer
    Category 3 - Solid Performer
    Category 4 - Inconsistent Player
    Category 5 - Core Player
    Category 6 - High Performer
    Category 7 - Potential Gem
    Category 8 - High Potential
    Category 9 - Star

Run from project root:
    python src/bert_talent_predictor.py

GPU setup for NVIDIA RTX cards:
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Notes:
    - First run requires internet to download distilbert-base-uncased from Hugging Face.
    - After the model is downloaded/saved, later runs can use the cached/saved model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from datasets import Dataset
except ImportError as exc:
    raise ImportError("Please install datasets first: pip install datasets") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "bert_talent_predictor"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bert_talent_prediction"

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "feedback_clean"
TARGET_COL = "label"
CATEGORY_COL = "category_name"
RAW_TARGET_COL = "nine_box_category"

CATEGORY_NAME_FALLBACK = {
    0: "Category 1 - Risk",
    1: "Category 2 - Average Performer",
    2: "Category 3 - Solid Performer",
    3: "Category 4 - Inconsistent Player",
    4: "Category 5 - Core Player",
    5: "Category 6 - High Performer",
    6: "Category 7 - Potential Gem",
    7: "Category 8 - High Potential",
    8: "Category 9 - Star",
}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross entropy for imbalanced 9-class data."""

    def __init__(self, class_weights: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT/DistilBERT for Nine-Box Talent Prediction")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted loss")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train_processed.csv"
    val_path = DATA_DIR / "validation_processed.csv"
    test_path = DATA_DIR / "test_processed.csv"

    missing = [str(p) for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed files are missing. Run preprocessing first: python src/preprocessing.py\n"
            + "Missing files: " + ", ".join(missing)
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required = [TEXT_COL, TARGET_COL]
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{name} data is missing required column: {col}")
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return train_df, val_df, test_df


def build_label_maps(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    label_to_name: Dict[int, str] = {}

    for label in sorted(combined[TARGET_COL].unique()):
        subset = combined[combined[TARGET_COL] == label]
        if CATEGORY_COL in subset.columns and subset[CATEGORY_COL].notna().any():
            name = str(subset[CATEGORY_COL].dropna().iloc[0])
            if not name.lower().startswith("category"):
                name = f"Category {label + 1} - {name}"
        elif RAW_TARGET_COL in subset.columns and subset[RAW_TARGET_COL].notna().any():
            name = str(subset[RAW_TARGET_COL].dropna().iloc[0])
        else:
            name = CATEGORY_NAME_FALLBACK.get(int(label), f"Category {label + 1}")
        label_to_name[int(label)] = name

    id2label = {str(k): v for k, v in label_to_name.items()}
    label2id = {v: str(k) for k, v in label_to_name.items()}
    return {"id2label": id2label, "label2id": label2id}


def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    small_df = df[[TEXT_COL, TARGET_COL]].rename(columns={TARGET_COL: "labels"}).copy()
    return Dataset.from_pandas(small_df, preserve_index=False)


def compute_metrics(eval_pred) -> Dict[str, float]:  # noqa: ANN001
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def save_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], path: Path) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    path.write_text(report, encoding="utf-8")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(label_names))
    short_names = [name.replace("Category ", "Cat ").replace(" - ", "\n") for name in label_names]
    plt.xticks(ticks, short_names, rotation=45, ha="right")
    plt.yticks(ticks, short_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_training_curves(trainer: Trainer, output_dir: Path) -> None:
    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv(output_dir / "bert_training_log_history.csv", index=False)

    train_logs = logs[logs["loss"].notna()] if "loss" in logs.columns else pd.DataFrame()
    eval_logs = logs[logs["eval_loss"].notna()] if "eval_loss" in logs.columns else pd.DataFrame()

    if not train_logs.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(train_logs["step"], train_logs["loss"], marker="o")
        plt.title("BERT Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(output_dir / "bert_training_loss.png", dpi=160)
        plt.close()

    if not eval_logs.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], marker="o")
        plt.title("BERT Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.tight_layout()
        plt.savefig(output_dir / "bert_validation_loss.png", dpi=160)
        plt.close()

        if "eval_macro_f1" in eval_logs.columns:
            plt.figure(figsize=(9, 5))
            plt.plot(eval_logs["epoch"], eval_logs["eval_macro_f1"], marker="o")
            plt.title("BERT Validation Macro F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro F1")
            plt.tight_layout()
            plt.savefig(output_dir / "bert_validation_macro_f1.png", dpi=160)
            plt.close()


def predict_dataframe(trainer: Trainer, df: pd.DataFrame, ds: Dataset, label_names: List[str], split_name: str) -> pd.DataFrame:
    output = trainer.predict(ds)
    logits = output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    conf = probs.max(axis=-1)

    result = df.copy()
    result["bert_predicted_label"] = preds
    result["bert_predicted_category"] = [label_names[i] for i in preds]
    result["bert_prediction_confidence"] = conf
    result["split"] = split_name
    return result


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 8B: BERT Nine-Box Talent Prediction")
    print("=" * 80)
    print("Torch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. For RTX GPU training, install CUDA-enabled PyTorch.")

    train_df, val_df, test_df = load_frames()
    label_maps = build_label_maps(train_df, val_df, test_df)
    id2label = {int(k): v for k, v in label_maps["id2label"].items()}
    label_names = [id2label[i] for i in sorted(id2label)]
    num_labels = len(label_names)

    with open(MODEL_DIR / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    print(f"Train size: {len(train_df)} | Validation size: {len(val_df)} | Test size: {len(test_df)}")
    print(f"Number of labels: {num_labels}")
    print("Labels:")
    for i, name in enumerate(label_names):
        print(f"  {i}: {name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=args.max_length)

    train_ds = make_hf_dataset(train_df).map(tokenize, batched=True)
    val_ds = make_hf_dataset(val_df).map(tokenize, batched=True)
    test_ds = make_hf_dataset(test_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: label_names[i] for i in range(num_labels)},
        label2id={label_names[i]: i for i in range(num_labels)},
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if not args.no_class_weights:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(sorted(train_df[TARGET_COL].unique())),
            y=train_df[TARGET_COL].values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
        pd.DataFrame({"label": list(range(num_labels)), "category": label_names, "class_weight": weights}).to_csv(
            OUTPUT_DIR / "bert_class_weights.csv", index=False
        )

    # Transformers versions differ on argument names. eval_strategy is newer; evaluation_strategy is older.
    common_args = dict(
        output_dir=str(MODEL_DIR / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    val_predictions = predict_dataframe(trainer, val_df, val_ds, label_names, "validation")
    test_predictions = predict_dataframe(trainer, test_df, test_ds, label_names, "test")

    y_val = val_predictions[TARGET_COL].to_numpy()
    y_val_pred = val_predictions["bert_predicted_label"].to_numpy()
    y_test = test_predictions[TARGET_COL].to_numpy()
    y_test_pred = test_predictions["bert_predicted_label"].to_numpy()

    save_report(y_val, y_val_pred, label_names, OUTPUT_DIR / "bert_validation_classification_report.txt")
    save_report(y_test, y_test_pred, label_names, OUTPUT_DIR / "bert_test_classification_report.txt")
    save_confusion_matrix(y_val, y_val_pred, label_names, OUTPUT_DIR / "bert_validation_confusion_matrix.png", "BERT Validation Confusion Matrix")
    save_confusion_matrix(y_test, y_test_pred, label_names, OUTPUT_DIR / "bert_test_confusion_matrix.png", "BERT Test Confusion Matrix")

    val_predictions.head(50).to_csv(OUTPUT_DIR / "bert_validation_sample_predictions.csv", index=False)
    test_predictions.head(75).to_csv(OUTPUT_DIR / "bert_test_sample_predictions.csv", index=False)
    pd.concat([val_predictions, test_predictions], ignore_index=True).to_csv(
        OUTPUT_DIR / "bert_all_predictions.csv", index=False
    )

    save_training_curves(trainer, OUTPUT_DIR)

    metrics = {
        "model_name": args.model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "num_labels": int(num_labels),
        "validation_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "validation_macro_f1": float(f1_score(y_val, y_val_pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_macro_f1": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)),
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.patience,
            "class_weighted_loss": not args.no_class_weights,
        },
    }
    with open(OUTPUT_DIR / "bert_talent_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = f"""BERT Nine-Box Talent Prediction Summary
======================================

Model: {args.model_name}
Device: {metrics['device']}
GPU: {metrics['cuda_name']}

Dataset sizes:
- Train: {len(train_df)}
- Validation: {len(val_df)}
- Test: {len(test_df)}

Task:
Predict original nine_box_category from employee feedback text.

Validation Results:
- Accuracy: {metrics['validation_accuracy']:.4f}
- Macro F1: {metrics['validation_macro_f1']:.4f}
- Weighted F1: {metrics['validation_weighted_f1']:.4f}

Test Results:
- Accuracy: {metrics['test_accuracy']:.4f}
- Macro F1: {metrics['test_macro_f1']:.4f}
- Weighted F1: {metrics['test_weighted_f1']:.4f}

Saved model:
{MODEL_DIR / 'final_model'}

Important note:
This is a fine-grained 9-class HR talent prediction task, which is harder than binary sentiment classification. The model distinguishes categories based on performance and potential signals in feedback text.
"""
    (OUTPUT_DIR / "bert_talent_prediction_summary.txt").write_text(summary, encoding="utf-8")

    trainer.save_model(str(MODEL_DIR / "final_model"))
    tokenizer.save_pretrained(str(MODEL_DIR / "final_model"))

    print("\nBERT talent predictor complete.")
    print(summary)


if __name__ == "__main__":
    main()
