"""
Run only the BERT Talent Prediction phase.

Run from project root:
    python run_bert_talent.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, "src/bert_talent_predictor.py"], cwd=PROJECT_ROOT, check=True)
