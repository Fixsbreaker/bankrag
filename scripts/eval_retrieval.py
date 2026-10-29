"""Оценка поиска на размеченных вопросах. Никаких вызовов платной LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
from bankrag.evaluation import load_questions, score_retrieval, validate_questions
from bankrag.search import Searcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "index")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/retrieval.json")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse-queries", action="store_true", help="использовать сохранённые эмбеддинги после проверки хеша вопросов и модели")
    parser.add_argument("--method", choices=["vector", "hybrid"], default="vector")
    args = parser.parse_args()
    if args.k < 1 or args.threads < 1:
        parser.error("k и threads должны быть положительными")
    faiss.omp_set_num_threads(args.threads)
    searcher = Searcher.load(args.index)
    questions = load_questions(args.questions)
    validate_questions(questions, searcher.chunks)
    # Изолируем PyTorch и FAISS: их macOS wheels могут загружать разные libomp.
    if not args.reuse_queries:
        subprocess.run([sys.executable, str(ROOT / "scripts/encode_queries.py"),
                        "--index", str(args.index), "--questions", str(args.questions)], check=True)
    query_meta = json.loads((args.index / "query_vectors.meta.json").read_text(encoding="utf-8"))
    if (query_meta["questions_sha256"] != hashlib.sha256(args.questions.read_bytes()).hexdigest()
            or query_meta["model"] != searcher.model_name
            or query_meta["question_ids"] != [q["id"] for q in questions]):
        raise ValueError("сохранённые эмбеддинги не соответствуют вопросам или модели; запустите без --reuse-queries")
    vectors = np.load(args.index / "query_vectors.npy", allow_pickle=False)
    if vectors.shape != (len(questions), searcher.index.d):
        raise ValueError("размерность эмбеддингов запросов не совпадает с индексом")
    if args.method == "hybrid":
        from bankrag.hybrid import HybridRetriever
        retriever = HybridRetriever(searcher.index, searcher.chunks)
        hits = [retriever.search(q["question"], v.reshape(1, -1), args.k) for q, v in zip(questions, vectors)]
    else:
        scores, ids = searcher.index.search(vectors, args.k)
        hits = [[{**searcher.chunks[int(i)], "score": float(s)} for s, i in zip(ss, ii) if i >= 0]
                for ss, ii in zip(scores, ids)]
    report = score_retrieval(questions, hits, args.k)
    report.update({"model": searcher.model_name, "method": args.method, "corpus_chunks": len(searcher.chunks),
                   "corpus_documents": len({c["source"] for c in searcher.chunks}),
                   "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                   "index_sha256": hashlib.sha256((args.index / "faiss.index").read_bytes()).hexdigest(),
                   "evaluation_scope": "exploratory development set; not a held-out test or production accuracy"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "questions"}, ensure_ascii=False, indent=2))
    print(f"Подробности: {args.out}")


if __name__ == "__main__":
    main()
