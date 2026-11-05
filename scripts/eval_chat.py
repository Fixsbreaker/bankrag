"""Оценка чата на 30 вопросах: выбор функции, параметры, проверка чисел, отказы, время ответа.

Нужна запущенная Ollama и скачанная модель. Результат дописывается в reports/chat_metrics.json
по имени модели, а reports/chat_metrics.md строится заново по всем моделям в файле.

    .venv/bin/python scripts/eval_chat.py --model qwen3:14b
    .venv/bin/python scripts/eval_chat.py --model qwen3:8b
    .venv/bin/python scripts/eval_chat.py --model qwen3:8b --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bankrag.faiss_safe import faiss  # noqa: E402, F401

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from bankrag.chat.agent import ChatSession  # noqa: E402
from bankrag.chat.llm import OllamaClient  # noqa: E402
from bankrag.data_loader import load_config  # noqa: E402
from bankrag.kb_index import KnowledgeBase, e5_embedder  # noqa: E402
from bankrag.similarity import SimilarityIndex  # noqa: E402
from bankrag.tools.data import KZ, load_case_data  # noqa: E402

QUESTIONS = ROOT / "tests" / "chat_eval" / "questions.yaml"
REPORT_JSON = ROOT / "reports" / "chat_metrics.json"
REPORT_MD = ROOT / "reports" / "chat_metrics.md"


def fill(text: str, accounts: dict[str, str]) -> str:
    return text.format(**accounts) if isinstance(text, str) else text


def args_match(expected: dict, actual: dict, normalize) -> bool:
    """Ожидаемые параметры совпали: счёт тот же, список счетов его содержит, период тот же."""
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(got, list):
            if normalize(value) not in [normalize(v) for v in got]:
                return False
        elif got is None or normalize(str(got)) != normalize(str(value)):
            return False
    return True


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def tools_time(m: dict) -> str:
    """Время до результата функции; у старых прогонов его не измеряли."""
    if m.get("tools_seconds_mean") is None:
        return "не измерялось"
    return f"{m['tools_seconds_mean']:.1f} / {m['tools_seconds_p95']:.1f}"


def render(metrics: dict) -> str:
    """Таблица по всем моделям, прогнанным скриптом."""
    lines = ["# Метрики чата", "",
             "Получено запуском `scripts/eval_chat.py` на казахстанском сценарии (сгенерирован командой). "
             "Вопросы и ожидаемые функции — в `tests/chat_eval/questions.yaml`.", "",
             "| Модель | Вопросов | Верная функция | Верные параметры | Ответ не прошёл проверку чисел | "
             "Показан шаблон | Верные отказы | Результат функции на экране, среднее / p95, с | "
             "Проверенный ответ, среднее / p95, с |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['questions']} | {m['tool_correct']} из {m['tool_questions']} "
                     f"({percent(m['tool_accuracy'])}) | {m['args_correct']} из {m['tool_questions']} "
                     f"({percent(m['args_accuracy'])}) | {m['check_failed']} из {m['questions']} "
                     f"({percent(m['check_failed_share'])}) | {m['fallback']} | {m['refusal_correct']} из "
                     f"{m['refusal_questions']} | {tools_time(m)} | {m['seconds_mean']:.1f} / {m['seconds_p95']:.1f} |")
    lines += ["", "«Не прошёл проверку чисел» — первый ответ модели содержал число или счёт, которых нет в результатах "
              "функций. Такой ответ не показывается: модель переписывает его, а при второй ошибке показывается шаблон "
              "из результата функции. Поэтому ни один показанный ответ не содержит непроверенных чисел.", "",
              "«Результат функции на экране» — когда интерфейс показывает результат функции (числа посчитаны кодом), "
              "пока модель пишет пересказ. Считается по вопросам, где функция вызывалась.", ""]
    for name, m in metrics.items():
        wrong = [r for r in m["rows"] if not r["correct"]]
        if wrong:
            lines += [f"### Ошибки {name}", "", "| Вопрос | Ожидалось | Вызвано |", "| --- | --- | --- |"]
            for row in wrong:
                lines.append(f"| {row['question']} | {row['expected']} | {row['called'] or 'ничего'} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="модель Ollama, по умолчанию из config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="только первые N вопросов (быстрая проверка)")
    args = parser.parse_args()
    config = load_config()
    client = OllamaClient.from_config(config["llm"], args.model)
    if not client.is_ready():
        sys.exit(f"Модель {client.name} недоступна: запустите Ollama и выполните ollama pull {client.name}")

    data = load_case_data(KZ, config, ROOT)
    data.similarity = SimilarityIndex.load(ROOT / "models", "similarity_kz")
    data.knowledge = KnowledgeBase.load(ROOT / "models", e5_embedder())
    demo = json.loads((ROOT / "reports" / "demo_case.json").read_text(encoding="utf-8"))
    accounts = {"first_drop": demo["first_drop"], "second": demo["layers"][1][0], "split1": demo["layers"][-1][0],
                "split2": demo["layers"][-1][-1], "victim": demo["victim"],
                "listed": json.loads((ROOT / "reports" / "watchlist.json").read_text(encoding="utf-8"))["demo_accounts"]["DEMO-1"]}
    items = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))[:args.limit]

    rows = []
    for number, item in enumerate(items, start=1):
        session = ChatSession(data, client, config)
        if item.get("context"):
            session.ask(fill(item["context"], accounts))
        question = fill(item["question"], accounts)
        turn = session.ask(question)
        called = [c["name"] for c in turn.calls]
        first_check = not turn.retried
        if item.get("refusal"):
            correct = bool(re.search(item["expect"], turn.answer, re.IGNORECASE))
            expected, tool_ok, args_ok = "отказ", None, None
        else:
            expected = item["tool"]
            tool_ok = bool(called) and called[0] == expected
            wanted = {k: fill(v, accounts) for k, v in (item.get("args") or {}).items()}
            args_ok = tool_ok and args_match(wanted, turn.calls[0]["arguments"], data.normalize_account)
            correct = bool(args_ok)
        rows.append({"question": question, "expected": expected, "called": ", ".join(called), "tool_ok": tool_ok,
                     "args_ok": args_ok, "correct": correct, "first_check_passed": first_check,
                     "fallback": turn.fallback, "seconds": round(turn.seconds, 2),
                     "tools_seconds": round(turn.tools_seconds, 2) if turn.tools_seconds is not None else None,
                     "answer": turn.answer})
        print(f"{number:2d}. {'OK ' if correct else 'ERR'} {turn.seconds:5.1f} с  {question}  →  {called}", flush=True)

    tool_rows = [r for r in rows if r["tool_ok"] is not None]
    refusal_rows = [r for r in rows if r["tool_ok"] is None]
    seconds = np.array([r["seconds"] for r in rows])
    # Время до результата функции — только у вопросов, где функция вызывалась.
    tools_seconds = np.array([r["tools_seconds"] for r in rows if r["tools_seconds"] is not None])
    result = {
        "questions": len(rows), "tool_questions": len(tool_rows), "refusal_questions": len(refusal_rows),
        "tool_correct": sum(r["tool_ok"] for r in tool_rows), "args_correct": sum(bool(r["args_ok"]) for r in tool_rows),
        "tool_accuracy": float(np.mean([r["tool_ok"] for r in tool_rows])) if tool_rows else 0.0,
        "args_accuracy": float(np.mean([bool(r["args_ok"]) for r in tool_rows])) if tool_rows else 0.0,
        "check_failed": sum(not r["first_check_passed"] for r in rows),
        "check_failed_share": float(np.mean([not r["first_check_passed"] for r in rows])),
        "fallback": sum(r["fallback"] for r in rows),
        "refusal_correct": sum(r["correct"] for r in refusal_rows),
        "seconds_mean": float(seconds.mean()), "seconds_p95": float(np.percentile(seconds, 95)),
        "tools_seconds_mean": float(tools_seconds.mean()) if len(tools_seconds) else None,
        "tools_seconds_p95": float(np.percentile(tools_seconds, 95)) if len(tools_seconds) else None, "rows": rows,
    }
    if args.limit is None:
        metrics = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
        metrics[client.name] = result
        REPORT_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_MD.write_text(render(metrics), encoding="utf-8")
    print(render({client.name: result}))


if __name__ == "__main__":
    main()
