"""Сохранить эмбеддинги вопросов без импорта FAISS (отдельный процесс для macOS)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from bankrag.embed import embed
from bankrag.evaluation import load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.index / "meta.json").read_text(encoding="utf-8"))
    questions = load_questions(args.questions)
    vectors = embed([q["question"] for q in questions], "query", metadata["model"])
    np.save(args.index / "query_vectors.npy", vectors)
    (args.index / "query_vectors.meta.json").write_text(json.dumps({
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "question_ids": [q["id"] for q in questions], "model": metadata["model"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Эмбеддинги запросов: {vectors.shape}")


if __name__ == "__main__":
    main()
