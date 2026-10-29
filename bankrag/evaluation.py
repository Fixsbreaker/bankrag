"""Проверка разметки и метрики поиска, без вызова LLM."""
from __future__ import annotations

import json
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"строка {line_number}: отсутствует question")
        if "expected_source" not in item:
            raise ValueError(f"строка {line_number}: требуется expected_source, null для вопроса без ответа")
        if item["expected_source"] is not None:
            if not item["expected_source"] or not item.get("expected_text"):
                raise ValueError(f"строка {line_number}: нужны источник и непустой expected_text")
        item.setdefault("id", f"q{line_number:03d}")
        questions.append(item)
    if not questions:
        raise ValueError("набор вопросов пуст")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("id вопросов должны быть уникальны")
    return questions


def is_relevant(hit: dict, question: dict) -> bool:
    """Недостаточно найти тот же документ: нужна размеченная фраза внутри фрагмента."""
    return (
        question["expected_source"] is not None
        and hit["source"] == question["expected_source"]
        and (question.get("expected_page") is None or hit.get("page") == question["expected_page"])
        and normalize(question["expected_text"]) in normalize(hit["text"])
    )


def validate_questions(questions: list[dict], chunks: list[dict]) -> None:
    missing = [q["id"] for q in questions if q["expected_source"] is not None
               and not any(is_relevant(c, q) for c in chunks)]
    if missing:
        raise ValueError("разметка не совпадает с корпусом, проверьте версию и границы фрагментов: " + ", ".join(missing))


def score_retrieval(questions: list[dict], all_hits: list[list[dict]], k: int) -> dict:
    if k < 1 or len(questions) != len(all_hits):
        raise ValueError("некорректное k или число списков результатов")
    rows = []
    ranks = []
    for question, hits in zip(questions, all_hits):
        rank = next((i for i, hit in enumerate(hits[:k], 1) if is_relevant(hit, question)), None)
        answerable = question["expected_source"] is not None
        if answerable:
            ranks.append(rank)
        rows.append({"id": question["id"], "question": question["question"],
                     "answerable": answerable, "first_relevant_rank": rank,
                     "hits": [{"chunk_id": h["chunk_id"], "source": h["source"],
                               **({"page": h["page"]} if "page" in h else {}),
                               **({"source_url": h["source_url"]} if "source_url" in h else {}),
                               "score": float(h["score"]), "text": h["text"]} for h in hits[:k]]})
    count = len(ranks)
    return {"k": k, "answerable_count": count,
            "unanswerable_count": len(questions) - count,
            f"hit_at_{k}": sum(r is not None for r in ranks) / count if count else None,
            f"mrr_at_{k}": sum(1 / r for r in ranks if r is not None) / count if count else None,
            "refusal_evaluated": False,
            "note": "Вопросы без ответа исключены из Hit/MRR. Их выдача сохранена, но качество отказа здесь не оценивается.",
            "questions": rows}
