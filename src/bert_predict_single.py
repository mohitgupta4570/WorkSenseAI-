"""
Predict a single employee feedback text using the saved BERT Nine-Box model.

Run from project root:
    python src/bert_predict_single.py "Employee shows strong leadership and consistently delivers results."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bert_talent_predictor" / "final_model"
LABEL_MAP_PATH = PROJECT_ROOT / "models" / "bert_talent_predictor" / "label_maps.json"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/bert_predict_single.py "feedback text here"')

    text = " ".join(sys.argv[1:])
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Saved BERT model not found. Run: python src/bert_talent_predictor.py")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_maps = json.load(f)
    id2label = {int(k): v for k, v in label_maps["id2label"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    print("Feedback:", text)
    print("Predicted label:", pred_id)
    print("Predicted category:", id2label[pred_id])
    print("Confidence:", round(confidence, 4))

    topk = torch.topk(probs, k=min(3, len(probs)))
    print("\nTop predictions:")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"- {id2label[int(idx)]}: {score:.4f}")


if __name__ == "__main__":
    main()
