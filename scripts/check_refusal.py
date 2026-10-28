"""Проверка на галлюцинации: модель должна отказаться там, где ответа в документах нет.

    python scripts/check_refusal.py [-k 5] [--min-score 0.8]

Контрольные вопросы (ответ в документах есть) нужны, чтобы модель, которая всегда
отказывается, не прошла проверку. Для подбора --min-score смотрите колонку top.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bankrag.answer import answer  # noqa: E402
from bankrag.search import Searcher  # noqa: E402


def _norm(text: str) -> str:
    return text.lower().replace(" ", "").replace(" ", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--index", default=ROOT / "index", type=Path)
    parser.add_argument("--questions", default=ROOT / "data" / "eval" / "refusal.json", type=Path)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    searcher = Searcher.load(args.index)

    refused_ok = 0
    print("Вопросы без ответа в документах (ожидаем отказ):")
    for q in data["unanswerable"]:
        hits = searcher.search(q, args.k)
        r = answer(q, hits, args.min_score)
        refused_ok += r["refused"]
        top = hits[0]["score"] if hits else 0.0
        print(f"  {'OK  ' if r['refused'] else 'FAIL'} top={top:.3f}  {q}")
        if not r["refused"]:
            print(f"       ответ модели: {r['answer']}")

    controls_ok = 0
    print("\nКонтрольные вопросы (ответ есть, отказ = ошибка):")
    for item in data["answerable"]:
        hits = searcher.search(item["q"], args.k)
        r = answer(item["q"], hits, args.min_score)
        ok = (not r["refused"]) and bool(r["sources"]) and all(_norm(e) in _norm(r["answer"]) for e in item["expect"])
        controls_ok += ok
        top = hits[0]["score"] if hits else 0.0
        print(f"  {'OK  ' if ok else 'FAIL'} top={top:.3f}  {item['q']}")
        if not ok:
            print(f"       ответ модели: {r['answer']}  (ожидали: {item['expect']}, источники: {r['sources']})")

    n_un, n_ctl = len(data["unanswerable"]), len(data["answerable"])
    print(f"\nОтказов: {refused_ok}/{n_un}. Верных ответов на контрольные: {controls_ok}/{n_ctl}.")
    sys.exit(0 if refused_ok == n_un and controls_ok == n_ctl else 1)


if __name__ == "__main__":
    main()
