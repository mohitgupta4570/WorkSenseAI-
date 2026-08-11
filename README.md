# WorkSense AI

## Intelligent Employee Feedback Analysis and Talent Prediction using NLP

This project analyzes employee performance feedback using NLP and predicts the employee's Nine-Box Talent Matrix category.

## Current Modules

1. Data preprocessing
2. Exploratory Data Analysis
3. FastText vector embeddings and semantic similarity analysis
4. POS tagging with opinion/action/workplace term extraction

## Upcoming Modules

1. HMM Sequence Tagging
2. Dependency Parsing
3. LSTM Neural Language Model
4. BERT Talent Classifier
5. Streamlit Dashboard

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/preprocessing.py
python src/eda.py
python src/embeddings.py
python src/pos_tagger.py
```

Outputs will be saved in the `outputs/` folder.

## Phase 2: FastText Vector Embeddings

This phase implements the assignment requirement: **Vector Embedding**.

### Files Added

- `src/embeddings.py`
- `src/embedding_similarity_demo.py`
- `models/fasttext/fasttext_employee_feedback.model`
- `models/fasttext/feedback_document_vectors.pkl`
- `data/processed/feedback_embeddings.csv`
- `outputs/embeddings/fasttext_similarity_report.txt`
- `outputs/embeddings/feedback_embedding_pca.png`

### Run FastText Training

```bash
python src/embeddings.py
```

### Test Semantic Similarity

```bash
python src/embedding_similarity_demo.py leadership
python src/embedding_similarity_demo.py communication
python src/embedding_similarity_demo.py performance
```

### What This Module Does

- Trains FastText word embeddings on employee feedback.
- Converts every feedback review into a 100-dimensional document vector.
- Generates semantic similarity results for HR terms such as performance, leadership, communication, growth, and attendance.
- Creates a PCA visualization of feedback embeddings grouped by Nine-Box category.

FastText is used because it handles rare words and unseen terms better than traditional Word2Vec by learning subword patterns.


## Run Everything from Scratch

From the project root, run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

On Windows, you can also run:

```bash
run_all.bat
```

On macOS/Linux, you can also run:

```bash
./run_all.sh
```

## Phase 3: POS Tagging and Workplace Language Extraction

This phase implements the assignment requirement: **POS Tagging**.

### Files Added

- `src/pos_tagger.py`
- `run_all.py`
- `run_all.bat`
- `run_all.sh`

### Run POS Tagging Only

```bash
python src/pos_tagger.py
```

### Outputs Generated

- `outputs/pos_analysis/pos_tags_sample.csv`
- `outputs/pos_analysis/opinion_words.csv`
- `outputs/pos_analysis/action_verbs.csv`
- `outputs/pos_analysis/workplace_terms.csv`
- `outputs/pos_analysis/pos_summary.txt`
- `outputs/pos_analysis/top_opinion_words.png`
- `outputs/pos_analysis/top_action_verbs.png`
- `outputs/pos_analysis/top_workplace_terms.png`

### What This Module Does

- Tags every important token with POS labels such as noun, verb, adjective, and adverb.
- Extracts opinion words from adjectives and adverbs.
- Extracts action verbs related to employee behavior.
- Extracts workplace concern terms and noun phrases.
- Saves visual and tabular outputs for the final assignment report.


## Phase 4: HMM Sequence Tagging

This phase adds an HMM-based probabilistic sequence tagging module.

Run individually:

```bash
python src/hmm_module.py
```

Generated outputs:

```text
outputs/hmm_analysis/hmm_token_tags_sample.csv
outputs/hmm_analysis/hmm_sequence_predictions.csv
outputs/hmm_analysis/hmm_transition_matrix.csv
outputs/hmm_analysis/hmm_top_emission_words.csv
outputs/hmm_analysis/hmm_state_distribution.png
outputs/hmm_analysis/hmm_transition_heatmap.png
outputs/hmm_analysis/hmm_summary.txt
```

The HMM module tags tokens into hidden states such as `POSITIVE_SIGNAL`, `NEGATIVE_SIGNAL`, `GROWTH_SIGNAL`, `RISK_SIGNAL`, and `NEUTRAL_CONTEXT` using transition probabilities, emission probabilities, and Viterbi decoding.

## Phase 5: Assignment Feedback Classification

This phase implements the classification categories required in the assignment:

### Sentiment categories
- Positive Feedback
- Negative Feedback
- Neutral Feedback

### Workplace issue categories
- Work Culture
- Salary & Benefits
- Career Growth
- Management Issues

Because the uploaded MTurk employee review dataset is originally a Nine-Box talent dataset, it does not contain direct labels for these assignment categories. The project therefore uses transparent weak supervision rules to generate assignment-style labels, then trains reusable TF-IDF + Logistic Regression classifiers.

Run:

```bash
python src/feedback_classifier.py
```

Generated outputs:

```text
outputs/classification/assignment_classification_sample_predictions.csv
outputs/classification/classification_summary.txt
outputs/classification/classification_metrics.json
outputs/classification/assignment_sentiment_distribution.png
outputs/classification/workplace_issue_distribution.png
outputs/classification/sentiment_confusion_matrix.png
outputs/classification/issue_confusion_matrix.png
outputs/classification/sentiment_top_features.csv
outputs/classification/issue_top_features.csv
```

Saved models:

```text
models/classifiers/assignment_sentiment_classifier.joblib
models/classifiers/workplace_issue_classifier.joblib
```

Run full project from the beginning:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

## Phase 6: Dependency Parsing + Relationship Extraction

This phase implements syntactic parsing to identify meaningful relationships between employee opinions/actions and workplace targets.

Examples of extracted relationships:

```text
poor -> communication
strong -> leadership
miss -> deadlines
improve -> performance
lack -> initiative
```

Run individually:

```bash
python src/parser_module.py
```

Generated outputs:

```text
outputs/parsing_analysis/parsed_relationships_sample.csv
outputs/parsing_analysis/extracted_relationships.csv
outputs/parsing_analysis/concern_target_pairs.csv
outputs/parsing_analysis/parsing_summary.txt
outputs/parsing_analysis/top_relationship_targets.png
outputs/parsing_analysis/top_opinion_action_terms.png
```

Important: for full dependency parsing quality, install the spaCy English model before running:

```bash
python -m spacy download en_core_web_sm
```

If the model is missing, the script uses a lightweight fallback tokenizer so the pipeline does not break, but the actual dependency parser should be used for the final project/demo.

Run full project from the beginning:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
```

## Phase 7: Neural Language Model using LSTM

This phase trains a lightweight LSTM Neural Language Model on employee feedback text.
It learns contextual language patterns and predicts the next word from a short workplace-feedback prompt.

Run only Phase 7:

```bash
python src/language_model.py
```

Generated files:

```text
models/lstm_lm/lstm_neural_language_model.pt
models/lstm_lm/lstm_lm_checkpoint.pt
models/lstm_lm/lstm_lm_vocab.json
outputs/language_model/lstm_lm_training_history.csv
outputs/language_model/lstm_lm_training_loss.png
outputs/language_model/lstm_lm_validation_perplexity.png
outputs/language_model/next_word_prediction_examples.csv
outputs/language_model/language_model_top_vocabulary.csv
outputs/language_model/language_model_summary.txt
```

This satisfies the assignment requirement: **Neural Language Model**.

## Phase 8B: BERT Nine-Box Talent Prediction

This phase fine-tunes `distilbert-base-uncased` to predict the original `nine_box_category` from employee feedback text.

### GPU PyTorch setup for RTX 4080

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Run BERT training

```bash
python src/preprocessing.py
python src/bert_talent_predictor.py
```

Optional custom run:

```bash
python src/bert_talent_predictor.py --epochs 8 --batch-size 8 --max-length 256 --lr 2e-5
```

### Predict one feedback after training

```bash
python src/bert_predict_single.py "The employee shows excellent leadership, strong ownership, and consistently exceeds expectations."
```

### BERT outputs

```text
models/bert_talent_predictor/final_model/
models/bert_talent_predictor/label_maps.json
outputs/bert_talent_prediction/bert_validation_classification_report.txt
outputs/bert_talent_prediction/bert_test_classification_report.txt
outputs/bert_talent_prediction/bert_validation_confusion_matrix.png
outputs/bert_talent_prediction/bert_test_confusion_matrix.png
outputs/bert_talent_prediction/bert_training_loss.png
outputs/bert_talent_prediction/bert_validation_loss.png
outputs/bert_talent_prediction/bert_validation_macro_f1.png
outputs/bert_talent_prediction/bert_test_sample_predictions.csv
outputs/bert_talent_prediction/bert_talent_metrics.json
outputs/bert_talent_prediction/bert_talent_prediction_summary.txt
```

Note: `run_all.py` keeps the lightweight talent predictor in the main pipeline so the full project can run quickly. Run BERT separately when you want the advanced GPU model.

## Phase 9: Streamlit Virtual Lab Dashboard

This phase implements the bonus-friendly Virtual Lab interface for the complete NLP system.

### Files Added

- `app/streamlit_app.py`
- `run_dashboard.py`
- `run_dashboard.bat`
- `run_dashboard.sh`

### Dashboard Features

- Dataset overview and category distributions
- Single feedback analysis
- Positive / Negative / Neutral feedback classification
- Workplace issue classification
- Extracted employee concerns
- POS tagging table
- HMM-style sequence tags
- Dependency parsing relationship extraction
- Nine-Box Talent Prediction
- Optional BERT prediction if the fine-tuned BERT model is trained and saved
- Generated output preview page

### Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

or:

```bash
python run_dashboard.py
```

### Complete Run from Scratch

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_all.py
streamlit run app/streamlit_app.py
```

### Optional BERT Training

```bash
python src/preprocessing.py
python src/bert_talent_predictor.py
streamlit run app/streamlit_app.py
```

If `models/bert_talent_predictor/final_model/` exists, the dashboard can use BERT for Nine-Box prediction. Otherwise, it automatically falls back to the saved lightweight Nine-Box model.
