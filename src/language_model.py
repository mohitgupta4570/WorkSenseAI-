"""
Phase 7: Neural Language Model using LSTM

This module trains a lightweight LSTM next-word prediction model on employee
feedback text. It satisfies the assignment requirement:

    Neural Language Model: Develop a Neural LM to understand employee feedback
    language patterns and predict contextual words related to workplace sentiment.

Run from project root:
    python src/language_model.py
"""

from __future__ import annotations

import json
import math
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "lstm_lm"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "language_model"

TRAIN_FILE = DATA_DIR / "train_processed.csv"
VAL_FILE = DATA_DIR / "validation_processed.csv"

RANDOM_SEED = 42
MAX_VOCAB_SIZE = 1200
SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 32
HIDDEN_DIM = 48
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.003
MIN_WORD_FREQ = 1
MAX_TRAIN_SEQUENCES = 1200
MAX_VAL_SEQUENCES = 400

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class FeedbackSequenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], targets: List[int]):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index]


class LSTMNeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        last_hidden = output[:, -1, :]
        logits = self.fc(self.dropout(last_hidden))
        return logits


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return [token for token in text.split() if len(token) > 1]


def load_feedback_texts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing first: python src/preprocessing.py"
        )
    df = pd.read_csv(path)
    text_col = "feedback_clean" if "feedback_clean" in df.columns else "feedback"
    return df[text_col].fillna("").astype(str).tolist()


def build_vocabulary(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    words = [word for word, freq in counter.most_common() if freq >= MIN_WORD_FREQ]
    words = words[: max(0, MAX_VOCAB_SIZE - len(SPECIAL_TOKENS))]

    word_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for word in words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word, counter


def encode_tokens(tokens: List[str], word_to_idx: Dict[str, int]) -> List[int]:
    unk_idx = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(token, unk_idx) for token in tokens]


def create_training_sequences(
    texts: List[str], word_to_idx: Dict[str, int], sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[List[List[int]], List[int]]:
    sequences: List[List[int]] = []
    targets: List[int] = []

    for text in texts:
        encoded = encode_tokens(tokenize(text), word_to_idx)
        if len(encoded) <= sequence_length:
            continue
        for i in range(sequence_length, len(encoded)):
            sequences.append(encoded[i - sequence_length : i])
            targets.append(encoded[i])

    return sequences, targets


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, accuracy, perplexity


def predict_next_words(
    model: nn.Module,
    prompt: str,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    device,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    model.eval()
    tokens = tokenize(prompt)
    encoded = encode_tokens(tokens, word_to_idx)

    pad_idx = word_to_idx[PAD_TOKEN]
    if len(encoded) < SEQUENCE_LENGTH:
        encoded = [pad_idx] * (SEQUENCE_LENGTH - len(encoded)) + encoded
    else:
        encoded = encoded[-SEQUENCE_LENGTH:]

    x = torch.tensor([encoded], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(top_k, probs.shape[0]))

    predictions = []
    for idx, prob in zip(indices.cpu().tolist(), values.cpu().tolist()):
        word = idx_to_word.get(int(idx), UNK_TOKEN)
        if word not in SPECIAL_TOKENS:
            predictions.append((word, float(prob)))
    return predictions


def plot_history(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Neural Language Model Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_lm_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_perplexity"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("LSTM Neural Language Model Validation Perplexity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_lm_validation_perplexity.png", dpi=300)
    plt.close()


def main() -> None:
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading employee feedback text...")
    train_texts = load_feedback_texts(TRAIN_FILE)
    val_texts = load_feedback_texts(VAL_FILE)

    print("Building vocabulary...")
    word_to_idx, idx_to_word, counter = build_vocabulary(train_texts)

    print("Creating next-word prediction sequences...")
    train_sequences, train_targets = create_training_sequences(train_texts, word_to_idx)
    val_sequences, val_targets = create_training_sequences(val_texts, word_to_idx)

    if not train_sequences or not val_sequences:
        raise ValueError("Not enough text to train the language model.")

    # Keep the training lightweight enough for a college laptop while still demonstrating a real Neural LM.
    if len(train_sequences) > MAX_TRAIN_SEQUENCES:
        sampled_idx = np.random.choice(len(train_sequences), MAX_TRAIN_SEQUENCES, replace=False)
        train_sequences = [train_sequences[i] for i in sampled_idx]
        train_targets = [train_targets[i] for i in sampled_idx]

    if len(val_sequences) > MAX_VAL_SEQUENCES:
        sampled_idx = np.random.choice(len(val_sequences), MAX_VAL_SEQUENCES, replace=False)
        val_sequences = [val_sequences[i] for i in sampled_idx]
        val_targets = [val_targets[i] for i in sampled_idx]

    train_dataset = FeedbackSequenceDataset(train_sequences, train_targets)
    val_dataset = FeedbackSequenceDataset(val_sequences, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    model = LSTMNeuralLanguageModel(
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=word_to_idx[PAD_TOKEN],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_perplexity": val_ppl,
            }
        )
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "lstm_neural_language_model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "lstm_lm_training_history.csv", index=False)
    plot_history(history_df)

    vocab_payload = {
        "word_to_idx": word_to_idx,
        "idx_to_word": {str(k): v for k, v in idx_to_word.items()},
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_vocab_size": MAX_VOCAB_SIZE,
    }
    with open(MODEL_DIR / "lstm_lm_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2)

    torch.save(
        {
            "model_state_dict": torch.load(MODEL_DIR / "lstm_neural_language_model.pt", map_location="cpu"),
            "vocab_size": len(word_to_idx),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "sequence_length": SEQUENCE_LENGTH,
        },
        MODEL_DIR / "lstm_lm_checkpoint.pt",
    )

    model.load_state_dict(torch.load(MODEL_DIR / "lstm_neural_language_model.pt", map_location=device))

    prompts = [
        "employee consistently",
        "strong leadership and",
        "needs improvement in",
        "communication skills are",
        "performance has been",
        "shows great potential",
        "struggles with deadlines",
    ]

    prediction_rows = []
    for prompt in prompts:
        preds = predict_next_words(model, prompt, word_to_idx, idx_to_word, device, top_k=7)
        prediction_rows.append(
            {
                "prompt": prompt,
                "top_predictions": "; ".join([f"{word} ({prob:.4f})" for word, prob in preds]),
            }
        )

    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "next_word_prediction_examples.csv", index=False
    )

    top_words_df = pd.DataFrame(counter.most_common(100), columns=["word", "frequency"])
    top_words_df.to_csv(OUTPUT_DIR / "language_model_top_vocabulary.csv", index=False)

    final = history[-1]
    summary = f"""LSTM Neural Language Model Summary
=================================

Purpose:
The model learns workplace-feedback language patterns and predicts the next contextual word in employee feedback sentences.

Model Type: LSTM Neural Language Model
Training Text Column: feedback_clean
Sequence Length: {SEQUENCE_LENGTH} words
Vocabulary Size: {len(word_to_idx)}
Training Sequences: {len(train_sequences)}
Validation Sequences: {len(val_sequences)}
Embedding Dimension: {EMBEDDING_DIM}
Hidden Dimension: {HIDDEN_DIM}
Epochs: {EPOCHS}
Device Used: {device}

Final Metrics:
Train Loss: {final['train_loss']:.4f}
Validation Loss: {final['val_loss']:.4f}
Validation Accuracy: {final['val_accuracy']:.4f}
Validation Perplexity: {final['val_perplexity']:.4f}

Generated Files:
- models/lstm_lm/lstm_neural_language_model.pt
- models/lstm_lm/lstm_lm_checkpoint.pt
- models/lstm_lm/lstm_lm_vocab.json
- outputs/language_model/lstm_lm_training_history.csv
- outputs/language_model/lstm_lm_training_loss.png
- outputs/language_model/lstm_lm_validation_perplexity.png
- outputs/language_model/next_word_prediction_examples.csv
- outputs/language_model/language_model_top_vocabulary.csv

Assignment Mapping:
This satisfies the Neural Language Model requirement by predicting contextual words from employee feedback language patterns.
"""
    with open(OUTPUT_DIR / "language_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\nNeural Language Model phase completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
