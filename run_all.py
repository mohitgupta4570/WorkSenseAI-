"""
Run the WorkSense AI pipeline from Phase 1 to Phase 8.

Run from project root:
    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Phase 1: Preprocessing", "src/preprocessing.py"),
    ("Phase 1: EDA", "src/eda.py"),
    ("Phase 2: FastText Embeddings", "src/embeddings.py"),
    ("Phase 3: POS Tagging", "src/pos_tagger.py"),
    ("Phase 4: HMM Sequence Tagging", "src/hmm_module.py"),
    ("Phase 5: Assignment Feedback Classification", "src/feedback_classifier.py"),
    ("Phase 6: Dependency Parsing", "src/parser_module.py"),
    ("Phase 7: LSTM Neural Language Model", "src/language_model.py"),
    ("Phase 8: Nine-Box Talent Prediction", "src/talent_predictor.py"),
]


def run_step(title: str, script: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    result = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Pipeline stopped at: {title}")


def main() -> None:
    for title, script in STEPS:
        run_step(title, script)
    print("\nAll completed successfully. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
