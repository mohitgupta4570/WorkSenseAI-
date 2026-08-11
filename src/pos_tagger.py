"""
POS Tagging and Opinion/Action Word Extraction for WorkSense AI
---------------------------------------------------------------
This module satisfies the POS Tagging requirement of the assignment.

It uses spaCy to:
1. Perform Part-of-Speech tagging on employee feedback
2. Extract opinion words, mainly adjectives/adverbs
3. Extract action verbs related to performance behavior
4. Extract workplace concern nouns/noun phrases
5. Save CSV reports and plots for final project documentation

Run from project root:
    python src/pos_tagger.py

Before running for the first time:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.language import Language


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pos_analysis"

POS_SAMPLE_PATH = OUTPUT_DIR / "pos_tags_sample.csv"
OPINION_WORDS_PATH = OUTPUT_DIR / "opinion_words.csv"
ACTION_VERBS_PATH = OUTPUT_DIR / "action_verbs.csv"
WORKPLACE_TERMS_PATH = OUTPUT_DIR / "workplace_terms.csv"
POS_SUMMARY_PATH = OUTPUT_DIR / "pos_summary.txt"
OPINION_PLOT_PATH = OUTPUT_DIR / "top_opinion_words.png"
VERB_PLOT_PATH = OUTPUT_DIR / "top_action_verbs.png"
NOUN_PLOT_PATH = OUTPUT_DIR / "top_workplace_terms.png"

# Domain-specific stop terms that are too generic in this dataset.
GENERIC_TERMS = {
    "employee", "company", "work", "job", "time", "thing", "area", "way",
    "people", "person", "team", "day", "month", "year", "role", "position",
    "feedback", "performance", "potential", "ability", "skill", "task",
}

# Useful workplace issue keywords for a focused concern list.
WORKPLACE_CONCERN_KEYWORDS = {
    "communication", "attendance", "deadline", "quality", "leadership", "initiative",
    "productivity", "collaboration", "teamwork", "punctuality", "reliability",
    "responsibility", "growth", "improvement", "training", "motivation",
    "discipline", "focus", "attitude", "behavior", "management", "problem",
    "mistake", "delay", "absence", "learning", "ownership", "support",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_spacy_model() -> Language:
    """Load spaCy English pipeline with POS tagger and dependency parser."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model 'en_core_web_sm' is not installed. Run this command once:\n"
            "    python -m spacy download en_core_web_sm\n"
            "Then rerun:\n"
            "    python src/pos_tagger.py"
        ) from exc


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run: python src/preprocessing.py"
        )

    df = pd.read_csv(path)
    if "feedback" not in df.columns:
        raise ValueError("Expected column 'feedback' was not found.")

    df["feedback"] = df["feedback"].fillna("").astype(str)
    return df[df["feedback"].str.strip().astype(bool)].reset_index(drop=True)


def useful_token(token) -> bool:
    """Keep clean lexical tokens only."""
    return (
        not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
        and len(token.lemma_.strip()) > 2
    )


def analyze_document(doc, record_id: str, category: str) -> Tuple[List[Dict], List[str], List[str], List[str]]:
    """Return token-level POS rows plus extracted opinion/action/workplace terms."""
    pos_rows: List[Dict] = []
    opinion_words: List[str] = []
    action_verbs: List[str] = []
    workplace_terms: List[str] = []

    for token in doc:
        if not useful_token(token):
            continue

        lemma = token.lemma_.lower().strip()

        pos_rows.append(
            {
                "record_id": record_id,
                "category": category,
                "token": token.text,
                "lemma": lemma,
                "pos": token.pos_,
                "tag": token.tag_,
                "dependency": token.dep_,
                "head": token.head.text,
            }
        )

        if token.pos_ in {"ADJ", "ADV"}:
            opinion_words.append(lemma)

        if token.pos_ == "VERB":
            action_verbs.append(lemma)

        if token.pos_ in {"NOUN", "PROPN"}:
            if lemma in WORKPLACE_CONCERN_KEYWORDS or lemma not in GENERIC_TERMS:
                workplace_terms.append(lemma)

    # Add noun chunks because they are more meaningful than single nouns for concerns.
    for chunk in doc.noun_chunks:
        clean_chunk = " ".join(
            tok.lemma_.lower()
            for tok in chunk
            if useful_token(tok) and tok.lemma_.lower() not in GENERIC_TERMS
        ).strip()
        if clean_chunk and len(clean_chunk.split()) <= 4:
            workplace_terms.append(clean_chunk)

    return pos_rows, opinion_words, action_verbs, workplace_terms


def save_frequency_csv(counter: Counter, path: Path, column_name: str, top_n: int = 100) -> pd.DataFrame:
    df = pd.DataFrame(counter.most_common(top_n), columns=[column_name, "frequency"])
    df.to_csv(path, index=False)
    return df


def plot_top_terms(freq_df: pd.DataFrame, term_col: str, title: str, path: Path, top_n: int = 20) -> None:
    if freq_df.empty:
        return

    plot_df = freq_df.head(top_n).sort_values("frequency", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df[term_col], plot_df["frequency"])
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel(term_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_summary(
    df: pd.DataFrame,
    pos_counter: Counter,
    opinion_counter: Counter,
    verb_counter: Counter,
    workplace_counter: Counter,
) -> str:
    lines = []
    lines.append("POS Tagging and Workplace Language Analysis Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total feedback records analyzed: {len(df)}")
    lines.append("")
    lines.append("Top POS Tags:")
    for pos, count in pos_counter.most_common(10):
        lines.append(f"  - {pos:<8} {count}")

    lines.append("")
    lines.append("Top Opinion Words (Adjectives/Adverbs):")
    for word, count in opinion_counter.most_common(15):
        lines.append(f"  - {word:<20} {count}")

    lines.append("")
    lines.append("Top Action Verbs:")
    for word, count in verb_counter.most_common(15):
        lines.append(f"  - {word:<20} {count}")

    lines.append("")
    lines.append("Top Workplace Terms / Concerns:")
    for word, count in workplace_counter.most_common(15):
        lines.append(f"  - {word:<30} {count}")

    lines.append("")
    lines.append("How this satisfies the assignment:")
    lines.append(
        "POS tagging identifies opinion words, action verbs, workplace issues, "
        "and employee concerns from manager feedback."
    )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()

    print("Loading processed employee feedback dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} feedback records.")

    print("Loading spaCy English model...")
    nlp = load_spacy_model()

    all_pos_rows: List[Dict] = []
    opinion_counter: Counter = Counter()
    verb_counter: Counter = Counter()
    workplace_counter: Counter = Counter()
    pos_counter: Counter = Counter()

    print("Running POS tagging and extracting workplace language patterns...")
    for _, row in df.iterrows():
        record_id = str(row.get("id", ""))
        category = str(row.get("category_name", row.get("nine_box_category", "Unknown")))
        doc = nlp(row["feedback"])

        pos_rows, opinions, verbs, workplace_terms = analyze_document(doc, record_id, category)
        all_pos_rows.extend(pos_rows)
        opinion_counter.update(opinions)
        verb_counter.update(verbs)
        workplace_counter.update(workplace_terms)
        pos_counter.update([item["pos"] for item in pos_rows])

    pos_df = pd.DataFrame(all_pos_rows)
    # Keep sample compact enough for GitHub/assignment packaging.
    pos_df.head(5000).to_csv(POS_SAMPLE_PATH, index=False)

    opinion_df = save_frequency_csv(opinion_counter, OPINION_WORDS_PATH, "opinion_word")
    verb_df = save_frequency_csv(verb_counter, ACTION_VERBS_PATH, "action_verb")
    workplace_df = save_frequency_csv(workplace_counter, WORKPLACE_TERMS_PATH, "workplace_term")

    plot_top_terms(opinion_df, "opinion_word", "Top Opinion Words in Employee Feedback", OPINION_PLOT_PATH)
    plot_top_terms(verb_df, "action_verb", "Top Action Verbs in Employee Feedback", VERB_PLOT_PATH)
    plot_top_terms(workplace_df, "workplace_term", "Top Workplace Terms and Concerns", NOUN_PLOT_PATH)

    summary = build_summary(df, pos_counter, opinion_counter, verb_counter, workplace_counter)
    POS_SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print(summary[:1600])
    print("\nSaved POS analysis outputs to:", OUTPUT_DIR)
    print(f"- {POS_SAMPLE_PATH}")
    print(f"- {OPINION_WORDS_PATH}")
    print(f"- {ACTION_VERBS_PATH}")
    print(f"- {WORKPLACE_TERMS_PATH}")
    print(f"- {POS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
