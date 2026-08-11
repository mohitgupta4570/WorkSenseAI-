"""
Phase 6: Dependency Parsing + Relationship Extraction
------------------------------------------------------
This module uses spaCy dependency parsing to analyze employee feedback sentence
structure and extract meaningful relationships between employee opinions/actions
and workplace targets.

It supports the assignment requirement:
"Parsing: Perform syntactic parsing to analyze sentence structure and extract
meaningful relationships between employee opinions, workplace factors, and
organizational policies."

Outputs:
- parsed_relationships_sample.csv
- extracted_relationships.csv
- concern_target_pairs.csv
- parsing_summary.txt
- top_relationship_targets.png
- top_opinion_action_terms.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "parsing_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "combined_assignment_classified.csv"
FALLBACK_INPUT_FILE = DATA_DIR / "combined_processed.csv"

# Workplace-specific words that are useful targets in feedback.
WORKPLACE_TARGET_KEYWORDS = {
    "performance", "potential", "leadership", "communication", "team", "teamwork",
    "manager", "management", "deadline", "deadlines", "quality", "work", "culture",
    "attendance", "punctuality", "productivity", "initiative", "ownership", "skill",
    "skills", "growth", "career", "learning", "promotion", "responsibility", "behavior",
    "attitude", "collaboration", "project", "projects", "task", "tasks", "goal",
    "goals", "confidence", "decision", "decisions", "delivery", "execution", "effort",
    "salary", "pay", "benefits", "compensation", "bonus", "environment", "support",
}

# Words that commonly express praise, problems, risk, improvement or action.
OPINION_ACTION_POS = {"ADJ", "VERB", "ADV"}
TARGET_POS = {"NOUN", "PROPN", "PRON"}


def load_spacy_model():
    """Load spaCy English model with a helpful error message.

    The real dependency parser requires ``en_core_web_sm``. A blank fallback is
    returned only so the script can fail gracefully in restricted environments;
    users should install the model for full parsing quality.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print(
            "WARNING: spaCy model 'en_core_web_sm' is not installed. "
            "Using a lightweight fallback tokenizer. For full dependency parsing, run: "
            "python -m spacy download en_core_web_sm"
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.meta["fallback_mode"] = True
        return nlp


def load_dataset() -> pd.DataFrame:
    """Load the latest processed dataset."""
    if INPUT_FILE.exists():
        return pd.read_csv(INPUT_FILE)
    if FALLBACK_INPUT_FILE.exists():
        return pd.read_csv(FALLBACK_INPUT_FILE)
    raise FileNotFoundError(
        "No processed dataset found. Run preprocessing and classification first."
    )


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def compound_phrase(token) -> str:
    """Return a compact noun phrase including compounds/adjectival modifiers."""
    modifiers = [child.text for child in token.lefts if child.dep_ in {"compound", "amod"}]
    phrase = " ".join(modifiers + [token.text])
    return phrase.lower().strip()


def extract_dependency_relationships(doc) -> List[Dict[str, str]]:
    """
    Extract opinion/action -> target relationships from dependency parses.

    Relationship types:
    - adjective_modifies_noun: strong leadership, poor communication
    - verb_object: misses deadlines, improves performance
    - verb_subject: employee struggles, employee delivers
    - copular_description: performance is strong, communication is poor
    - prepositional_link: struggles with deadlines
    """
    relationships: List[Dict[str, str]] = []

    for token in doc:
        # ADJ -> NOUN, e.g., poor communication, strong leadership
        if token.dep_ in {"amod", "acomp", "advmod"} and token.head.pos_ in TARGET_POS:
            relationships.append({
                "opinion_or_action": token.lemma_.lower(),
                "target": compound_phrase(token.head),
                "relationship_type": "adjective_modifies_noun",
                "dependency": token.dep_,
                "sentence": token.sent.text.strip(),
            })

        # VERB -> direct object, e.g., improves performance, misses deadlines
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in {"dobj", "obj", "attr"} and child.pos_ in TARGET_POS:
                    relationships.append({
                        "opinion_or_action": token.lemma_.lower(),
                        "target": compound_phrase(child),
                        "relationship_type": "verb_object",
                        "dependency": child.dep_,
                        "sentence": token.sent.text.strip(),
                    })

            # VERB + preposition + object, e.g., struggles with deadlines
            for prep in [c for c in token.children if c.dep_ == "prep"]:
                for pobj in [c for c in prep.children if c.dep_ in {"pobj", "obj"} and c.pos_ in TARGET_POS]:
                    relationships.append({
                        "opinion_or_action": token.lemma_.lower(),
                        "target": compound_phrase(pobj),
                        "relationship_type": f"verb_preposition_{prep.text.lower()}",
                        "dependency": f"{prep.dep_}->{pobj.dep_}",
                        "sentence": token.sent.text.strip(),
                    })

            # Subject-verb pairs, e.g., employee struggles / employee delivers
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in TARGET_POS:
                    relationships.append({
                        "opinion_or_action": token.lemma_.lower(),
                        "target": compound_phrase(child),
                        "relationship_type": "verb_subject",
                        "dependency": child.dep_,
                        "sentence": token.sent.text.strip(),
                    })

        # Copular constructions: performance is strong, attitude is poor
        if token.pos_ == "ADJ":
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in TARGET_POS:
                    relationships.append({
                        "opinion_or_action": token.lemma_.lower(),
                        "target": compound_phrase(child),
                        "relationship_type": "copular_description",
                        "dependency": child.dep_,
                        "sentence": token.sent.text.strip(),
                    })

    # Deduplicate while preserving order.
    seen: set[Tuple[str, str, str, str]] = set()
    unique = []
    for rel in relationships:
        key = (
            rel["opinion_or_action"],
            rel["target"],
            rel["relationship_type"],
            rel["sentence"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


def extract_concern_target_pairs(relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter dependency relationships to workplace-relevant target pairs."""
    pairs = []
    for rel in relationships:
        target_tokens = set(rel["target"].lower().split())
        if target_tokens & WORKPLACE_TARGET_KEYWORDS:
            severity = infer_severity(rel["opinion_or_action"])
            pairs.append({
                "concern_or_signal": rel["opinion_or_action"],
                "workplace_target": rel["target"],
                "relationship_type": rel["relationship_type"],
                "severity_or_signal": severity,
                "sentence": rel["sentence"],
            })
    return pairs


def infer_severity(term: str) -> str:
    """Simple interpretable rule layer for signal/severity classification."""
    negative = {
        "poor", "low", "weak", "miss", "missed", "struggle", "struggles", "lack",
        "lacks", "delay", "delayed", "inconsistent", "unreliable", "risk", "fail",
        "fails", "limited", "concern", "concerns", "difficult", "difficulty", "problem",
    }
    positive = {
        "strong", "excellent", "great", "good", "reliable", "consistent", "lead",
        "leads", "improve", "improves", "deliver", "delivers", "support", "supports",
        "collaborate", "collaborates", "mentor", "mentors", "high", "effective",
    }
    growth = {"improve", "develop", "learn", "grow", "potential", "progress", "adapt"}

    lemma = term.lower()
    if lemma in negative:
        return "Concern / Risk Signal"
    if lemma in positive:
        return "Positive Strength Signal"
    if lemma in growth:
        return "Growth Signal"
    return "Neutral Context Signal"


def save_bar_chart(counter: Counter, title: str, xlabel: str, output_path: Path, top_n: int = 15):
    items = counter.most_common(top_n)
    if not items:
        return
    labels, values = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.barh(list(labels)[::-1], list(values)[::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def extract_rule_based_relationships(text: str) -> List[Dict[str, str]]:
    """Fallback relationship extractor used only when spaCy parser is unavailable."""
    import re

    words = re.findall(r"[A-Za-z]+", text.lower())
    relationships: List[Dict[str, str]] = []

    opinion_words = {
        "poor", "strong", "weak", "excellent", "good", "great", "low", "high",
        "consistent", "inconsistent", "reliable", "unreliable", "effective",
        "limited", "positive", "negative", "average", "outstanding", "substandard",
    }
    action_words = {
        "miss", "misses", "missed", "improve", "improves", "improved",
        "struggle", "struggles", "struggled", "lead", "leads", "led",
        "deliver", "delivers", "delivered", "communicate", "communicates",
        "collaborate", "collaborates", "manage", "manages", "support", "supports",
        "lack", "lacks", "lacked", "develop", "develops", "developed",
    }

    for i, word in enumerate(words):
        if word in opinion_words or word in action_words:
            window = words[i + 1 : i + 6]
            target = next((w for w in window if w in WORKPLACE_TARGET_KEYWORDS), "")
            if target:
                relationships.append({
                    "opinion_or_action": word,
                    "target": target,
                    "relationship_type": "rule_based_fallback",
                    "dependency": "fallback",
                    "sentence": text.strip(),
                })
    return relationships

def main(sample_limit: int = 250):
    print("\n[Phase 6] Running dependency parsing and relationship extraction...")

    df = load_dataset()
    nlp = load_spacy_model()

    if "feedback" not in df.columns:
        raise ValueError("Dataset must contain a 'feedback' column.")

    # Parse a useful sample to keep runtime reasonable. The full dataset can be parsed
    # by increasing sample_limit or setting it to len(df).
    sample_df = df.head(sample_limit).copy()

    relationship_rows = []
    parsed_sample_rows = []
    concern_rows = []

    texts = [normalize_text(x) for x in sample_df["feedback"].tolist()]
    docs = list(nlp.pipe(texts, batch_size=32))

    for row_idx, (idx, row) in enumerate(sample_df.iterrows()):
        doc = docs[row_idx]
        if nlp.meta.get("fallback_mode"):
            relationships = extract_rule_based_relationships(normalize_text(row.get("feedback", "")))
        else:
            relationships = extract_dependency_relationships(doc)
        concern_pairs = extract_concern_target_pairs(relationships)

        parsed_tokens = []
        for token in doc:
            if not token.is_space:
                parsed_tokens.append({
                    "token": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "head": token.head.text,
                })

        parsed_sample_rows.append({
            "id": row.get("id", idx),
            "person_name": row.get("person_name", ""),
            "feedback": row.get("feedback", ""),
            "assignment_sentiment": row.get("assignment_sentiment", ""),
            "assignment_issue_category": row.get("assignment_issue_category", ""),
            "dependency_tokens_json": json.dumps(parsed_tokens, ensure_ascii=False),
            "relationships_json": json.dumps(relationships, ensure_ascii=False),
            "concern_target_pairs_json": json.dumps(concern_pairs, ensure_ascii=False),
        })

        for rel in relationships:
            relationship_rows.append({
                "id": row.get("id", idx),
                "person_name": row.get("person_name", ""),
                "assignment_sentiment": row.get("assignment_sentiment", ""),
                "assignment_issue_category": row.get("assignment_issue_category", ""),
                **rel,
            })

        for pair in concern_pairs:
            concern_rows.append({
                "id": row.get("id", idx),
                "person_name": row.get("person_name", ""),
                "assignment_sentiment": row.get("assignment_sentiment", ""),
                "assignment_issue_category": row.get("assignment_issue_category", ""),
                **pair,
            })

    parsed_df = pd.DataFrame(parsed_sample_rows)
    relationships_df = pd.DataFrame(relationship_rows)
    concerns_df = pd.DataFrame(concern_rows)

    parsed_df.to_csv(OUTPUT_DIR / "parsed_relationships_sample.csv", index=False)
    relationships_df.to_csv(OUTPUT_DIR / "extracted_relationships.csv", index=False)
    concerns_df.to_csv(OUTPUT_DIR / "concern_target_pairs.csv", index=False)

    target_counter = Counter(concerns_df["workplace_target"]) if not concerns_df.empty else Counter()
    opinion_counter = Counter(concerns_df["concern_or_signal"]) if not concerns_df.empty else Counter()
    relation_counter = Counter(relationships_df["relationship_type"]) if not relationships_df.empty else Counter()

    save_bar_chart(
        target_counter,
        "Top Workplace Targets Found by Dependency Parsing",
        "Frequency",
        OUTPUT_DIR / "top_relationship_targets.png",
    )
    save_bar_chart(
        opinion_counter,
        "Top Opinion / Action Terms Found by Dependency Parsing",
        "Frequency",
        OUTPUT_DIR / "top_opinion_action_terms.png",
    )

    with open(OUTPUT_DIR / "parsing_summary.txt", "w", encoding="utf-8") as f:
        f.write("Phase 6: Dependency Parsing + Relationship Extraction Summary\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Input file: {INPUT_FILE if INPUT_FILE.exists() else FALLBACK_INPUT_FILE}\n")
        f.write(f"Parsed feedback rows: {len(sample_df)}\n")
        f.write(f"Total dependency relationships extracted: {len(relationships_df)}\n")
        f.write(f"Workplace concern/target pairs extracted: {len(concerns_df)}\n\n")
        f.write("Top relationship types:\n")
        for item, count in relation_counter.most_common(10):
            f.write(f"- {item}: {count}\n")
        f.write("\nTop workplace targets:\n")
        for item, count in target_counter.most_common(15):
            f.write(f"- {item}: {count}\n")
        f.write("\nTop opinion/action terms:\n")
        for item, count in opinion_counter.most_common(15):
            f.write(f"- {item}: {count}\n")
        f.write("\nExample output format:\n")
        f.write("poor -> communication\n")
        f.write("strong -> leadership\n")
        f.write("miss -> deadlines\n")
        f.write("improve -> performance\n")

    print("[Phase 6] Dependency parsing complete.")
    print(f"[Phase 6] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
