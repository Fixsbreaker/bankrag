"""Показывает топ-k фрагментов для вопроса.

    python scripts/ask.py "Какая комиссия за перевод?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bankrag.search import Searcher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--index", default=ROOT / "index", type=Path)
    args = parser.parse_args()

    for rank, hit in enumerate(Searcher.load(args.index).search(args.question, args.k), 1):
        preview = " ".join(hit["text"].split())[:200]
        print(f"[{rank}] {hit['score']:.3f}  {hit['chunk_id']}\n    {preview}\n")


if __name__ == "__main__":
    main()
