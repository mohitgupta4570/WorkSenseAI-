#!/usr/bin/env bash
set -e
echo "Running WorkSense AI pipeline: Phase 1 to Phase 7"
python src/preprocessing.py
python src/eda.py
python src/embeddings.py
python src/pos_tagger.py
python src/hmm_module.py
python src/feedback_classifier.py
python src/parser_module.py
python src/language_model.py
python src/talent_predictor.py
echo "Done. Check the outputs folder."
