"""
HMM-based Sequence Tagging for WorkSense AI
-------------------------------------------
This module satisfies the Hidden Markov Model requirement of the assignment.

It implements a simple supervised Hidden Markov Model from scratch using:
1. Hidden states representing employee feedback signals
2. Observed tokens from feedback text
3. Transition probabilities between hidden states
4. Emission probabilities from states to words
5. Viterbi decoding for token-level sequence tagging

Why from scratch instead of only calling a library?
- It is easier to explain in viva/interview.
- It clearly demonstrates HMM concepts: states, observations, transitions,
  emissions, and Viterbi decoding.
- It works without requiring manually annotated token-level labels.

Run from project root:
    python src/hmm_module.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hmm_analysis"
MODEL_DIR = PROJECT_ROOT / "models" / "hmm"

TOKEN_TAGS_PATH = OUTPUT_DIR / "hmm_token_tags_sample.csv"
SEQUENCE_REPORT_PATH = OUTPUT_DIR / "hmm_sequence_predictions.csv"
TRANSITION_MATRIX_PATH = OUTPUT_DIR / "hmm_transition_matrix.csv"
STATE_WORDS_PATH = OUTPUT_DIR / "hmm_top_emission_words.csv"
SUMMARY_PATH = OUTPUT_DIR / "hmm_summary.txt"
STATE_DISTRIBUTION_PLOT_PATH = OUTPUT_DIR / "hmm_state_distribution.png"
TRANSITION_HEATMAP_PATH = OUTPUT_DIR / "hmm_transition_heatmap.png"


STATES = [
    "POSITIVE_SIGNAL",
    "NEGATIVE_SIGNAL",
    "GROWTH_SIGNAL",
    "RISK_SIGNAL",
    "NEUTRAL_CONTEXT",
]

POSITIVE_WORDS = {
    "excellent", "great", "good", "strong", "outstanding", "reliable", "consistent",
    "effective", "productive", "proactive", "confident", "skilled", "valuable",
    "dependable", "efficient", "successful", "collaborative", "supportive", "motivated",
    "leader", "leadership", "achieve", "achieved", "exceeds", "exceptional", "star",
    "impressive", "quality", "positive", "high", "top", "best", "trustworthy",
}

NEGATIVE_WORDS = {
    "poor", "weak", "low", "bad", "limited", "inconsistent", "unreliable", "negative",
    "miss", "missed", "missing", "delay", "delayed", "late", "struggle", "struggles",
    "struggling", "problem", "problems", "issue", "issues", "difficult", "difficulty",
    "lack", "lacks", "lacking", "fail", "fails", "failed", "below", "substandard",
    "unproductive", "concern", "concerns", "mistake", "mistakes", "slow", "absent",
}

GROWTH_WORDS = {
    "potential", "growth", "improve", "improvement", "develop", "development", "learning",
    "training", "coach", "coaching", "mentor", "mentoring", "opportunity", "opportunities",
    "progress", "progressing", "future", "promotion", "promote", "career", "initiative",
    "aspire", "aspiration", "capacity", "capable", "talent", "emerging",
}

RISK_WORDS = {
    "risk", "critical", "warning", "urgent", "serious", "underperform", "underperforming",
    "termination", "disciplinary", "discipline", "attendance", "punctuality", "deadline",
    "deadlines", "behavior", "attitude", "conflict", "complaint", "complaints", "escalation",
    "requires", "needed", "needs", "immediate", "rarely", "never", "unable",
}

WORD_TO_STATE: Dict[str, str] = {}
for word in POSITIVE_WORDS:
    WORD_TO_STATE[word] = "POSITIVE_SIGNAL"
for word in NEGATIVE_WORDS:
    WORD_TO_STATE[word] = "NEGATIVE_SIGNAL"
for word in GROWTH_WORDS:
    WORD_TO_STATE[word] = "GROWTH_SIGNAL"
for word in RISK_WORDS:
    WORD_TO_STATE[word] = "RISK_SIGNAL"


@dataclass
class HMMModel:
    states: List[str]
    start_logprob: Dict[str, float]
    transition_logprob: Dict[str, Dict[str, float]]
    emission_logprob: Dict[str, Dict[str, float]]
    unknown_logprob: Dict[str, float]
    vocabulary: set


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run preprocessing first:\n"
            "    python src/preprocessing.py"
        )
    df = pd.read_csv(path)
    if "feedback_clean" not in df.columns and "feedback" not in df.columns:
        raise ValueError("Dataset must contain either 'feedback_clean' or 'feedback'.")
    return df


def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    return re.findall(r"[a-z]+", text)


def weak_label_token(token: str) -> str:
    return WORD_TO_STATE.get(token, "NEUTRAL_CONTEXT")


def build_weak_sequences(texts: Iterable[str]) -> List[List[Tuple[str, str]]]:
    sequences: List[List[Tuple[str, str]]] = []
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < 2:
            continue
        sequence = [(token, weak_label_token(token)) for token in tokens]
        sequences.append(sequence)
    return sequences


def train_supervised_hmm(sequences: Sequence[Sequence[Tuple[str, str]]], alpha: float = 0.1) -> HMMModel:
    """Estimate start, transition, and emission probabilities with add-alpha smoothing."""
    start_counts = Counter()
    transition_counts: Dict[str, Counter] = {state: Counter() for state in STATES}
    emission_counts: Dict[str, Counter] = {state: Counter() for state in STATES}
    state_counts = Counter()
    vocabulary = set()

    for sequence in sequences:
        if not sequence:
            continue
        first_state = sequence[0][1]
        start_counts[first_state] += 1

        for i, (token, state) in enumerate(sequence):
            vocabulary.add(token)
            emission_counts[state][token] += 1
            state_counts[state] += 1
            if i < len(sequence) - 1:
                next_state = sequence[i + 1][1]
                transition_counts[state][next_state] += 1

    num_states = len(STATES)
    vocab_size = max(len(vocabulary), 1)
    num_sequences = max(len(sequences), 1)

    start_logprob = {}
    for state in STATES:
        prob = (start_counts[state] + alpha) / (num_sequences + alpha * num_states)
        start_logprob[state] = math.log(prob)

    transition_logprob: Dict[str, Dict[str, float]] = {state: {} for state in STATES}
    for state in STATES:
        total = sum(transition_counts[state].values())
        for next_state in STATES:
            prob = (transition_counts[state][next_state] + alpha) / (total + alpha * num_states)
            transition_logprob[state][next_state] = math.log(prob)

    emission_logprob: Dict[str, Dict[str, float]] = {state: {} for state in STATES}
    unknown_logprob: Dict[str, float] = {}
    for state in STATES:
        total = sum(emission_counts[state].values())
        denominator = total + alpha * (vocab_size + 1)
        unknown_logprob[state] = math.log(alpha / denominator)
        for token in vocabulary:
            prob = (emission_counts[state][token] + alpha) / denominator
            emission_logprob[state][token] = math.log(prob)

    return HMMModel(
        states=list(STATES),
        start_logprob=start_logprob,
        transition_logprob=transition_logprob,
        emission_logprob=emission_logprob,
        unknown_logprob=unknown_logprob,
        vocabulary=vocabulary,
    )


def emission_score(model: HMMModel, state: str, token: str) -> float:
    return model.emission_logprob[state].get(token, model.unknown_logprob[state])


def viterbi_decode(model: HMMModel, tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    dp: List[Dict[str, float]] = []
    backpointer: List[Dict[str, str]] = []

    first_scores = {}
    first_back = {}
    for state in model.states:
        first_scores[state] = model.start_logprob[state] + emission_score(model, state, tokens[0])
        first_back[state] = ""
    dp.append(first_scores)
    backpointer.append(first_back)

    for t in range(1, len(tokens)):
        token = tokens[t]
        current_scores = {}
        current_back = {}
        for state in model.states:
            best_prev_state = None
            best_score = -float("inf")
            for prev_state in model.states:
                score = (
                    dp[t - 1][prev_state]
                    + model.transition_logprob[prev_state][state]
                    + emission_score(model, state, token)
                )
                if score > best_score:
                    best_score = score
                    best_prev_state = prev_state
            current_scores[state] = best_score
            current_back[state] = best_prev_state or model.states[0]
        dp.append(current_scores)
        backpointer.append(current_back)

    best_final_state = max(dp[-1], key=dp[-1].get)
    best_path = [best_final_state]
    for t in range(len(tokens) - 1, 0, -1):
        best_path.append(backpointer[t][best_path[-1]])
    best_path.reverse()
    return best_path


def state_counts_to_summary(tags: Sequence[str]) -> str:
    counts = Counter(tags)
    if not counts:
        return "No signal"
    dominant = counts.most_common(1)[0][0]
    readable = dominant.replace("_", " ").title()
    return readable


def save_transition_matrix(model: HMMModel) -> pd.DataFrame:
    rows = []
    for state in model.states:
        row = {"from_state": state}
        for next_state in model.states:
            row[next_state] = round(math.exp(model.transition_logprob[state][next_state]), 6)
        rows.append(row)
    matrix_df = pd.DataFrame(rows)
    matrix_df.to_csv(TRANSITION_MATRIX_PATH, index=False)
    return matrix_df


def save_top_emissions(sequences: Sequence[Sequence[Tuple[str, str]]], top_n: int = 20) -> pd.DataFrame:
    state_word_counts: Dict[str, Counter] = {state: Counter() for state in STATES}
    for sequence in sequences:
        for token, state in sequence:
            state_word_counts[state][token] += 1

    rows = []
    for state, counts in state_word_counts.items():
        for word, count in counts.most_common(top_n):
            rows.append({"state": state, "word": word, "count": count})
    df = pd.DataFrame(rows)
    df.to_csv(STATE_WORDS_PATH, index=False)
    return df


def create_state_distribution_plot(token_tags_df: pd.DataFrame) -> None:
    counts = token_tags_df["hmm_state"].value_counts()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("HMM State Distribution in Sample Feedback")
    plt.xlabel("Hidden State")
    plt.ylabel("Token Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(STATE_DISTRIBUTION_PLOT_PATH, dpi=160)
    plt.close()


def create_transition_heatmap(matrix_df: pd.DataFrame) -> None:
    states = model_states = [col for col in matrix_df.columns if col != "from_state"]
    values = matrix_df[states].values
    plt.figure(figsize=(8, 6))
    plt.imshow(values, aspect="auto")
    plt.xticks(range(len(states)), states, rotation=45, ha="right")
    plt.yticks(range(len(states)), matrix_df["from_state"].tolist())
    plt.colorbar(label="Transition Probability")
    plt.title("HMM Transition Probability Heatmap")
    plt.tight_layout()
    plt.savefig(TRANSITION_HEATMAP_PATH, dpi=160)
    plt.close()


def analyze_feedback_samples(model: HMMModel, df: pd.DataFrame, sample_size: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text_col = "feedback_clean" if "feedback_clean" in df.columns else "feedback"
    sample_df = df[[col for col in ["id", "person_name", "nine_box_category", text_col] if col in df.columns]].head(sample_size)

    token_rows = []
    sequence_rows = []

    for _, row in sample_df.iterrows():
        text = row[text_col]
        tokens = tokenize(text)
        predicted_states = viterbi_decode(model, tokens)
        state_counter = Counter(predicted_states)

        for token, state in zip(tokens, predicted_states):
            token_rows.append({
                "id": row.get("id", ""),
                "person_name": row.get("person_name", ""),
                "token": token,
                "hmm_state": state,
            })

        sequence_rows.append({
            "id": row.get("id", ""),
            "person_name": row.get("person_name", ""),
            "nine_box_category": row.get("nine_box_category", ""),
            "feedback_preview": str(row[text_col])[:220],
            "dominant_hmm_signal": state_counts_to_summary(predicted_states),
            "positive_signal_count": state_counter["POSITIVE_SIGNAL"],
            "negative_signal_count": state_counter["NEGATIVE_SIGNAL"],
            "growth_signal_count": state_counter["GROWTH_SIGNAL"],
            "risk_signal_count": state_counter["RISK_SIGNAL"],
            "neutral_context_count": state_counter["NEUTRAL_CONTEXT"],
        })

    token_tags_df = pd.DataFrame(token_rows)
    sequence_df = pd.DataFrame(sequence_rows)
    token_tags_df.to_csv(TOKEN_TAGS_PATH, index=False)
    sequence_df.to_csv(SEQUENCE_REPORT_PATH, index=False)
    return token_tags_df, sequence_df


def write_summary(
    df: pd.DataFrame,
    sequences: Sequence[Sequence[Tuple[str, str]]],
    token_tags_df: pd.DataFrame,
    sequence_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
) -> None:
    state_distribution = token_tags_df["hmm_state"].value_counts().to_dict() if not token_tags_df.empty else {}
    dominant_distribution = sequence_df["dominant_hmm_signal"].value_counts().to_dict() if not sequence_df.empty else {}

    lines = [
        "WorkSense AI - HMM Sequence Tagging Summary",
        "=" * 55,
        "",
        "Purpose:",
        "This module implements a Hidden Markov Model for token-level sequence tagging",
        "of employee feedback. It estimates start, transition, and emission probabilities",
        "and applies Viterbi decoding to identify positive, negative, growth, risk, and",
        "neutral feedback signals.",
        "",
        f"Dataset rows analyzed: {len(df)}",
        f"Training sequences created: {len(sequences)}",
        f"Hidden states: {', '.join(STATES)}",
        "",
        "Sample token-level state distribution:",
    ]
    for state, count in state_distribution.items():
        lines.append(f"- {state}: {count}")

    lines.extend([
        "",
        "Dominant sequence-level signals in sample:",
    ])
    for signal, count in dominant_distribution.items():
        lines.append(f"- {signal}: {count}")

    lines.extend([
        "",
        "Generated Files:",
        f"- {TOKEN_TAGS_PATH.relative_to(PROJECT_ROOT)}",
        f"- {SEQUENCE_REPORT_PATH.relative_to(PROJECT_ROOT)}",
        f"- {TRANSITION_MATRIX_PATH.relative_to(PROJECT_ROOT)}",
        f"- {STATE_WORDS_PATH.relative_to(PROJECT_ROOT)}",
        f"- {STATE_DISTRIBUTION_PLOT_PATH.relative_to(PROJECT_ROOT)}",
        f"- {TRANSITION_HEATMAP_PATH.relative_to(PROJECT_ROOT)}",
        "",
        "How to explain this in viva:",
        "The feedback text is treated as an observation sequence. The hidden states represent",
        "latent workplace feedback signals. The HMM learns how these states transition across",
        "a sentence and which words are likely to be emitted by each state. Viterbi decoding is",
        "then used to infer the most probable hidden-state sequence for new feedback.",
    ])

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    text_col = "feedback_clean" if "feedback_clean" in df.columns else "feedback"

    sequences = build_weak_sequences(df[text_col].fillna(""))
    model = train_supervised_hmm(sequences)

    matrix_df = save_transition_matrix(model)
    save_top_emissions(sequences)
    token_tags_df, sequence_df = analyze_feedback_samples(model, df)

    if not token_tags_df.empty:
        create_state_distribution_plot(token_tags_df)
    create_transition_heatmap(matrix_df)
    write_summary(df, sequences, token_tags_df, sequence_df, matrix_df)

    print("HMM sequence tagging completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Summary report: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
