"""
Phase 9: Streamlit Virtual Lab Dashboard for WorkSense AI

Run from project root:
    streamlit run app/streamlit_app.py

This app combines the assignment-required NLP modules:
- Sentiment classification: Positive / Negative / Neutral Feedback
- Workplace issue classification: Work Culture / Salary & Benefits / Career Growth / Management Issues
- POS tagging
- HMM-style sequence tagging
- Dependency parsing and relationship extraction
- Nine-Box Talent Prediction using the saved lightweight model, with optional BERT support
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SENTIMENT_MODEL_PATH = MODEL_DIR / "classifiers" / "assignment_sentiment_classifier.joblib"
ISSUE_MODEL_PATH = MODEL_DIR / "classifiers" / "workplace_issue_classifier.joblib"
TALENT_MODEL_PATH = MODEL_DIR / "talent_predictor" / "nine_box_talent_predictor.joblib"
LABEL_MAP_PATH = MODEL_DIR / "talent_predictor" / "label_maps.json"
BERT_MODEL_DIR = MODEL_DIR / "bert_talent_predictor" / "final_model"
BERT_LABEL_MAP_PATH = MODEL_DIR / "bert_talent_predictor" / "label_maps.json"

POSITIVE_WORDS = {
    "excellent", "strong", "great", "good", "outstanding", "reliable", "consistent",
    "motivated", "collaborative", "effective", "efficient", "leader", "leadership",
    "improved", "improves", "improvement", "potential", "skilled", "talented",
    "productive", "proactive", "dedicated", "valuable", "innovative", "successful",
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

GROWTH_WORDS = {
    "growth", "promotion", "career", "learning", "training", "skill", "skills",
    "development", "potential", "future", "mentor", "leadership", "opportunity",
    "opportunities", "progress", "advancement", "improve", "improvement",
}

RISK_WORDS = {
    "late", "absent", "missed", "misses", "deadline", "deadlines", "risk", "poor",
    "weak", "struggle", "struggles", "low", "lack", "lacks", "fails", "failed",
    "conflict", "complaint", "underperform", "unreliable", "inconsistent",
}

WORKPLACE_TERMS = {
    "team", "communication", "performance", "quality", "leadership", "attendance",
    "deadline", "deadlines", "manager", "management", "culture", "salary", "pay",
    "benefits", "career", "growth", "skill", "skills", "training", "promotion",
    "collaboration", "productivity", "initiative", "ownership", "responsibility",
}

CONCERN_KEYWORDS: Dict[str, List[str]] = {
    "Attendance / Punctuality": ["late", "attendance", "absent", "absence", "breaks", "punctual"],
    "Communication": ["communication", "communicate", "argues", "disrespectful", "feedback", "listen", "listening"],
    "Performance Quality": ["performance", "quality", "productive", "productivity", "deliver", "results"],
    "Skill Gap": ["skill", "skills", "training", "learn", "learning", "development"],
    "Leadership Potential": ["leader", "leadership", "potential", "mentor", "guidance", "future"],
    "Team Collaboration": ["team", "coworker", "coworkers", "peer", "peers", "collaboration", "collaborative"],
    "Career Growth": ["growth", "promotion", "career", "opportunity", "opportunities", "progress"],
    "Salary & Benefits": ["salary", "pay", "compensation", "benefit", "benefits", "bonus", "raise"],
}

st.set_page_config(
    page_title="WorkSense AI Virtual Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> List[str]:
    return normalize_text(text).split()


@st.cache_resource
def load_spacy_model():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


@st.cache_resource
def load_joblib_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    combined = DATA_DIR / "combined_processed.csv"
    train = DATA_DIR / "train_processed.csv"
    val = DATA_DIR / "validation_processed.csv"
    test = DATA_DIR / "test_processed.csv"
    if combined.exists():
        return pd.read_csv(combined)
    frames = []
    for p in [train, val, test]:
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_assignment_data() -> pd.DataFrame:
    combined = DATA_DIR / "combined_assignment_classified.csv"
    if combined.exists():
        return pd.read_csv(combined)
    return pd.DataFrame()


@st.cache_data
def load_label_maps() -> Dict[str, Dict[str, str]]:
    if LABEL_MAP_PATH.exists():
        return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {"label_to_short_name": {}, "label_to_category": {}}


@st.cache_resource
def load_bert_assets():
    if not BERT_MODEL_DIR.exists() or AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return None, None, None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        if BERT_LABEL_MAP_PATH.exists():
            maps = json.loads(BERT_LABEL_MAP_PATH.read_text(encoding="utf-8"))
        else:
            maps = load_label_maps()
        return tokenizer, model, device, maps
    except Exception:
        return None, None, None, None


def predict_with_proba(model, text: str) -> Tuple[str, float, Dict[str, float]]:
    if model is None:
        return "Model not found", 0.0, {}
    pred = model.predict([text])[0]
    confidence = 0.0
    distribution = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        confidence = float(np.max(probs))
        distribution = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    return str(pred), confidence, distribution


def predict_talent_lightweight(text: str) -> Tuple[str, float, Dict[str, float]]:
    talent_model = load_joblib_model(TALENT_MODEL_PATH)
    maps = load_label_maps()
    label_to_short_name = maps.get("label_to_short_name", {})
    pred, confidence, distribution = predict_with_proba(talent_model, normalize_text(text))
    pred_label = str(int(float(pred))) if pred.replace(".", "", 1).isdigit() else str(pred)
    readable = label_to_short_name.get(pred_label, label_to_short_name.get(str(pred), str(pred)))
    readable_dist = {}
    for key, value in distribution.items():
        try:
            readable_key = label_to_short_name.get(str(int(float(key))), str(key))
        except Exception:
            readable_key = str(key)
        readable_dist[readable_key] = value
    return readable, confidence, readable_dist


def predict_talent_bert(text: str) -> Tuple[str, float, Dict[str, float]]:
    tokenizer, model, device, maps = load_bert_assets()
    if tokenizer is None or model is None:
        return predict_talent_lightweight(text)
    label_to_short_name = maps.get("label_to_short_name", {}) if maps else {}
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    readable = label_to_short_name.get(str(pred_label), f"Label {pred_label}")
    distribution = {label_to_short_name.get(str(i), f"Label {i}"): float(p) for i, p in enumerate(probs)}
    return readable, float(np.max(probs)), distribution


def hmm_tag_token(token: str) -> str:
    t = token.lower()
    if t in NEGATIVE_WORDS:
        return "NEGATIVE_SIGNAL"
    if t in RISK_WORDS:
        return "RISK_SIGNAL"
    if t in POSITIVE_WORDS:
        return "POSITIVE_SIGNAL"
    if t in GROWTH_WORDS:
        return "GROWTH_SIGNAL"
    return "NEUTRAL_CONTEXT"


def extract_concerns(text: str) -> List[str]:
    token_set = set(tokens(text))
    found = []
    for concern, keywords in CONCERN_KEYWORDS.items():
        if any(keyword in token_set for keyword in keywords):
            found.append(concern)
    return found or ["General Employee Feedback"]


def pos_analysis(text: str) -> pd.DataFrame:
    nlp = load_spacy_model()
    if nlp is None:
        return pd.DataFrame({"message": ["spaCy model not found. Run: python -m spacy download en_core_web_sm"]})
    doc = nlp(text)
    rows = []
    for token in doc:
        if token.is_space:
            continue
        rows.append({
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dependency": token.dep_,
            "head": token.head.text,
        })
    return pd.DataFrame(rows)


def dependency_relationships(text: str) -> pd.DataFrame:
    nlp = load_spacy_model()
    if nlp is None:
        return pd.DataFrame({"message": ["spaCy model not found. Run: python -m spacy download en_core_web_sm"]})
    doc = nlp(text)
    rows = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        token_l = token.lemma_.lower()
        if token.pos_ in {"ADJ", "VERB"} or token_l in POSITIVE_WORDS or token_l in NEGATIVE_WORDS:
            targets = []
            for child in token.children:
                if child.pos_ in {"NOUN", "PROPN", "PRON"} or child.dep_ in {"dobj", "nsubj", "pobj", "attr"}:
                    targets.append(child.text)
            if token.head is not token and token.head.pos_ in {"NOUN", "PROPN", "VERB"}:
                targets.append(token.head.text)
            targets = list(dict.fromkeys(targets))
            if targets:
                rows.append({
                    "opinion_or_action": token.text,
                    "relation": token.dep_,
                    "target": ", ".join(targets),
                    "interpretation": f"{token.text} → {', '.join(targets)}",
                })
    if not rows:
        rows.append({"opinion_or_action": "No strong pair found", "relation": "-", "target": "-", "interpretation": "No relationship extracted"})
    return pd.DataFrame(rows)


def display_metric_card(label: str, value: str, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def sidebar_status():
    st.sidebar.title("🧠 WorkSense AI")
    st.sidebar.caption("NLP Virtual Lab Dashboard")
    st.sidebar.divider()
    st.sidebar.subheader("Model Status")
    st.sidebar.write("Sentiment model:", "✅" if SENTIMENT_MODEL_PATH.exists() else "❌")
    st.sidebar.write("Issue model:", "✅" if ISSUE_MODEL_PATH.exists() else "❌")
    st.sidebar.write("Nine-Box model:", "✅" if TALENT_MODEL_PATH.exists() else "❌")
    st.sidebar.write("BERT model:", "✅" if BERT_MODEL_DIR.exists() else "Optional / not trained")
    st.sidebar.write("spaCy parser:", "✅" if load_spacy_model() is not None else "❌")
    st.sidebar.divider()
    st.sidebar.info("Run from project root: streamlit run app/streamlit_app.py")


def page_dashboard():
    st.title("WorkSense AI: Employee Feedback NLP Dashboard")
    st.write("An interactive Virtual Lab for employee feedback analysis, sentiment prediction, workplace issue classification, and Nine-Box talent prediction.")

    df = load_processed_data()
    assignment_df = load_assignment_data()

    if df.empty:
        st.warning("Processed data not found. Run `python src/preprocessing.py` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        display_metric_card("Total Feedback", f"{len(df):,}")
    with c2:
        display_metric_card("Nine-Box Classes", f"{df['label'].nunique() if 'label' in df else 'N/A'}")
    with c3:
        display_metric_card("Avg Feedback Length", f"{df['feedback_len'].mean():.0f}" if "feedback_len" in df else "N/A")
    with c4:
        display_metric_card("Avg Sentences", f"{df['num_of_sent'].mean():.2f}" if "num_of_sent" in df else "N/A")

    st.subheader("Nine-Box Category Distribution")
    if "nine_box_category" in df.columns:
        category_counts = df["nine_box_category"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig = px.bar(category_counts, x="category", y="count", title="Employee Talent Category Distribution")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    if not assignment_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Assignment Sentiment Distribution")
            senti = assignment_df["assignment_sentiment"].value_counts().reset_index()
            senti.columns = ["sentiment", "count"]
            st.plotly_chart(px.pie(senti, names="sentiment", values="count"), use_container_width=True)
        with col_b:
            st.subheader("Workplace Issue Distribution")
            issue = assignment_df["assignment_issue_category"].value_counts().reset_index()
            issue.columns = ["issue", "count"]
            st.plotly_chart(px.bar(issue, x="issue", y="count"), use_container_width=True)

    st.subheader("Sample Dataset")
    cols = [c for c in ["person_name", "feedback", "nine_box_category", "feedback_len", "num_of_sent"] if c in df.columns]
    st.dataframe(df[cols].head(20), use_container_width=True)


def page_analyze_feedback():
    st.title("Analyze Single Employee Feedback")
    sample = "The employee consistently delivers high quality work, shows strong leadership potential, and communicates effectively with the team."
    text = st.text_area("Enter employee feedback", value=sample, height=160)
    use_bert = st.toggle("Use BERT for Nine-Box prediction if trained", value=False)

    if not text.strip():
        st.warning("Enter feedback text to analyze.")
        return

    if st.button("Analyze Feedback", type="primary"):
        sentiment_model = load_joblib_model(SENTIMENT_MODEL_PATH)
        issue_model = load_joblib_model(ISSUE_MODEL_PATH)

        sentiment, sentiment_conf, sentiment_dist = predict_with_proba(sentiment_model, text)
        issue, issue_conf, issue_dist = predict_with_proba(issue_model, text)
        if use_bert:
            talent, talent_conf, talent_dist = predict_talent_bert(text)
        else:
            talent, talent_conf, talent_dist = predict_talent_lightweight(text)

        st.subheader("Prediction Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            display_metric_card("Sentiment", sentiment, f"Confidence: {sentiment_conf:.2%}")
        with c2:
            display_metric_card("Workplace Issue", issue, f"Confidence: {issue_conf:.2%}")
        with c3:
            display_metric_card("Nine-Box Talent", talent, f"Confidence: {talent_conf:.2%}")

        st.subheader("Extracted Employee Concerns")
        st.write(", ".join(extract_concerns(text)))

        col1, col2, col3 = st.columns(3)
        with col1:
            if sentiment_dist:
                st.plotly_chart(px.bar(x=list(sentiment_dist.keys()), y=list(sentiment_dist.values()), labels={"x": "Sentiment", "y": "Probability"}, title="Sentiment Confidence"), use_container_width=True)
        with col2:
            if issue_dist:
                st.plotly_chart(px.bar(x=list(issue_dist.keys()), y=list(issue_dist.values()), labels={"x": "Issue", "y": "Probability"}, title="Issue Confidence"), use_container_width=True)
        with col3:
            if talent_dist:
                dist_df = pd.DataFrame({"category": list(talent_dist.keys()), "probability": list(talent_dist.values())}).sort_values("probability", ascending=False).head(5)
                st.plotly_chart(px.bar(dist_df, x="category", y="probability", title="Top Talent Categories"), use_container_width=True)

        st.subheader("POS Tagging")
        st.dataframe(pos_analysis(text), use_container_width=True)

        st.subheader("HMM-Style Sequence Tags")
        hmm_df = pd.DataFrame([{"token": tok, "hmm_state": hmm_tag_token(tok)} for tok in tokens(text)])
        st.dataframe(hmm_df, use_container_width=True)

        st.subheader("Dependency Parsing Relationships")
        st.dataframe(dependency_relationships(text), use_container_width=True)


def page_nlp_explorer():
    st.title("NLP Explorer")
    text = st.text_area("Try a sentence", value="The manager praised her leadership but noted poor attendance and missed deadlines.", height=140)
    if text.strip():
        st.subheader("Token-Level POS and Dependency Analysis")
        st.dataframe(pos_analysis(text), use_container_width=True)

        st.subheader("Extracted Opinion/Action → Target Relationships")
        st.dataframe(dependency_relationships(text), use_container_width=True)

        st.subheader("HMM Sequence Pattern")
        hmm_df = pd.DataFrame([{"token": tok, "state": hmm_tag_token(tok)} for tok in tokens(text)])
        st.dataframe(hmm_df, use_container_width=True)
        state_counts = hmm_df["state"].value_counts().reset_index()
        state_counts.columns = ["state", "count"]
        st.plotly_chart(px.bar(state_counts, x="state", y="count", title="HMM State Distribution"), use_container_width=True)


def page_outputs():
    st.title("Generated Project Outputs")
    st.write("This page previews important generated files from the pipeline.")

    output_groups = {
        "EDA": OUTPUT_DIR,
        "Embeddings": OUTPUT_DIR / "embeddings",
        "POS Analysis": OUTPUT_DIR / "pos_analysis",
        "HMM Analysis": OUTPUT_DIR / "hmm_analysis",
        "Classification": OUTPUT_DIR / "classification",
        "Parsing": OUTPUT_DIR / "parsing_analysis",
        "Language Model": OUTPUT_DIR / "language_model",
        "Talent Prediction": OUTPUT_DIR / "talent_prediction",
        "BERT Talent Prediction": OUTPUT_DIR / "bert_talent_prediction",
    }

    for group, path in output_groups.items():
        with st.expander(group, expanded=False):
            if not path.exists():
                st.info("No outputs generated yet for this module.")
                continue
            files = sorted([p for p in path.glob("**/*") if p.is_file()])
            if not files:
                st.info("No files found.")
            for p in files[:30]:
                st.write(f"`{p.relative_to(PROJECT_ROOT)}`")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    st.image(str(p), use_container_width=True)
                elif p.suffix.lower() in {".txt", ".md"}:
                    try:
                        st.text(p.read_text(encoding="utf-8")[:2500])
                    except Exception:
                        pass
                elif p.suffix.lower() == ".csv":
                    try:
                        st.dataframe(pd.read_csv(p).head(10), use_container_width=True)
                    except Exception:
                        pass


def page_how_to_run():
    st.title("How to Run the Complete Project")
    st.code("""
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
streamlit run app/streamlit_app.py
""", language="bash")

    st.subheader("Run BERT Talent Model Separately")
    st.code("""
# Recommended for RTX GPU users
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python src/preprocessing.py
python src/bert_talent_predictor.py
""", language="bash")

    st.subheader("Manual Pipeline")
    st.code("""
python src/preprocessing.py
python src/eda.py
python src/embeddings.py
python src/pos_tagger.py
python src/hmm_module.py
python src/feedback_classifier.py
python src/parser_module.py
python src/language_model.py
python src/talent_predictor.py
streamlit run app/streamlit_app.py
""", language="bash")


def main():
    sidebar_status()
    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Analyze Feedback",
            "NLP Explorer",
            "Generated Outputs",
            "How to Run",
        ],
    )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Analyze Feedback":
        page_analyze_feedback()
    elif page == "NLP Explorer":
        page_nlp_explorer()
    elif page == "Generated Outputs":
        page_outputs()
    else:
        page_how_to_run()


if __name__ == "__main__":
    main()
