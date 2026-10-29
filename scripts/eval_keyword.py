"""Простой TF-IDF baseline на том же корпусе и тех же вопросах."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bankrag.evaluation import load_questions, score_retrieval, validate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "index")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/retrieval_keyword.json")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k должно быть положительным")
    chunks = json.loads((args.index / "chunks.json").read_text(encoding="utf-8"))
    questions = load_questions(args.questions)
    validate_questions(questions, chunks)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    corpus = vectorizer.fit_transform([c["text"] for c in chunks])
    queries = vectorizer.transform([q["question"] for q in questions])
    scores = (queries @ corpus.T).toarray()
    all_hits = [[{**chunks[int(i)], "score": float(row[i])}
                 for i in np.argsort(-row, kind="stable")[:args.k]] for row in scores]
    report = score_retrieval(questions, all_hits, args.k)
    report.update(method="TF-IDF word unigrams and bigrams; sublinear_tf=True; cosine",
                  evaluation_scope="same exploratory development set; basic lexical baseline, not a full BM25 system")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "questions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
