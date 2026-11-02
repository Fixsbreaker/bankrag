"""Проверка локальной модели: Ollama отвечает и модель умеет вызывать функцию.

    .venv/bin/python scripts/check_llm.py
    .venv/bin/python scripts/check_llm.py --model qwen3:8b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankrag.chat.llm import LLMError, OllamaClient  # noqa: E402
from bankrag.data_loader import load_config  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "account_profile",
    "description": "Профиль счёта: риск и признаки. Вызывай, когда спрашивают про конкретный счёт.",
    "parameters": {"type": "object", "properties": {"account": {"type": "string", "description": "номер счёта"}},
                   "required": ["account"]}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="имя модели Ollama, по умолчанию из config.yaml")
    args = parser.parse_args()
    client = OllamaClient.from_config(load_config()["llm"], args.model)
    try:
        models = client.available_models()
    except LLMError as error:
        sys.exit(f"Ollama не отвечает: {error}\nЗапустите: brew services start ollama")
    print("Скачанные модели:", ", ".join(models) or "нет")
    if not client.is_ready():
        sys.exit(f"Модели {client.name} нет. Скачайте: ollama pull {client.name}")

    messages = [{"role": "user", "content": "Почему счёт KZ-1001 подозрительный?"}]
    first = client.chat(messages, [TOOL])
    print(f"Загрузка и первый ответ: {first.seconds:.1f} с")
    reply = client.chat(messages, [TOOL])
    print(f"Ответ с вызовом функции: {reply.seconds:.1f} с")
    if not reply.tool_calls:
        sys.exit(f"Модель не вызвала функцию. Ответ: {reply.content[:200]}")
    call = reply.tool_calls[0]
    ok = call["name"] == "account_profile" and call["arguments"].get("account") == "KZ-1001"
    print("Вызов:", call, "— верно" if ok else "— НЕВЕРНО")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
