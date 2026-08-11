"""
Interactive FastText Similarity Demo
------------------------------------
Use this after running src/embeddings.py.

Examples:
    python src/embedding_similarity_demo.py leadership
    python src/embedding_similarity_demo.py communication
"""

from __future__ import annotations

import sys
from pathlib import Path

from gensim.models import FastText


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fasttext" / "fasttext_employee_feedback.model"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/embedding_similarity_demo.py <word>")
        return

    word = sys.argv[1].lower().strip()

    if not MODEL_PATH.exists():
        print("FastText model not found. First run: python src/embeddings.py")
        return

    model = FastText.load(str(MODEL_PATH))

    print(f"\nSemantic neighbors for: {word}\n" + "-" * 40)

    if word not in model.wv:
        print(
            "This exact word is not in the training vocabulary. "
            "FastText can still infer its vector, but direct nearest-neighbor lookup works best for known words."
        )
        return

    for similar_word, score in model.wv.most_similar(word, topn=10):
        print(f"{similar_word:<20} {score:.4f}")


if __name__ == "__main__":
    main()
