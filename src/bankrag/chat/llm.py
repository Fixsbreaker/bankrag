"""Клиент локальной модели через HTTP API Ollama.

Запросы идут прямо в /api/chat стандартной библиотекой: отдельный пакет не нужен,
а в тестах клиент легко подменить заглушкой с тем же методом chat.
Режим «размышлений» Qwen3 выключается полем think (см. docs/api.md в репозитории Ollama).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(RuntimeError):
    """Модель недоступна или ответила с ошибкой."""


@dataclass
class LLMReply:
    """Ответ модели: текст и запрошенные вызовы функций."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class ChatModel(Protocol):
    """То, что нужно циклу чата от модели. Реальный клиент и заглушка тестов устроены одинаково."""

    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply: ...


class OllamaClient:
    """Чат с моделью Ollama без потоковой передачи."""

    def __init__(self, url: str, model: str, temperature: float = 0.0, think: bool = False,
                 timeout_seconds: float = 180.0, keep_alive: str = "30m"):
        self.url = url.rstrip("/")
        self.name = model
        self.temperature = temperature
        self.think = think
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    @classmethod
    def from_config(cls, settings: dict, model: str | None = None) -> "OllamaClient":
        """Клиент по секции llm из config.yaml."""
        return cls(settings["url"], model or settings["model"], settings["temperature"], settings["think"],
                   settings["timeout_seconds"], settings["keep_alive"])

    def _post(self, path: str, payload: dict | None, timeout: float) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"},
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LLMError(f"Ollama ответила {error.code}: {error.read().decode('utf-8', 'replace')[:300]}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise LLMError(f"Ollama недоступна по адресу {self.url}: {error}") from error

    def available_models(self) -> list[str]:
        """Имена скачанных моделей. Ошибка, если Ollama не запущена."""
        return [model["name"] for model in self._post("/api/tags", None, 3).get("models", [])]

    def is_ready(self) -> bool:
        """Ollama запущена и нужная модель скачана."""
        try:
            names = self.available_models()
        except LLMError:
            return False
        return any(name == self.name or name.split(":latest")[0] == self.name for name in names)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        """Один запрос к модели. Функции не выполняются здесь: их вызывает цикл чата."""
        payload = {"model": self.name, "messages": messages, "stream": False, "think": self.think,
                   "keep_alive": self.keep_alive, "options": {"temperature": self.temperature}}
        if tools:
            payload["tools"] = tools
        started = time.perf_counter()
        answer = self._post("/api/chat", payload, self.timeout_seconds)
        message = answer.get("message", {})
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append({"name": function.get("name", ""), "arguments": arguments})
        return LLMReply(message.get("content") or "", calls, time.perf_counter() - started)
