"""Цикл чата: вопрос → модель выбирает функции → код их выполняет → модель пишет ответ → проверка чисел.

Если модель недоступна или дважды ошиблась в числах, ответ собирается по шаблону из результатов функций.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bankrag.chat.checker import Allowed, check_answer
from bankrag.chat.llm import ChatModel, LLMError
from bankrag.chat.specs import TOOL_NAMES, TOOL_SPECS
from bankrag.tools import fmt
from bankrag.tools import functions as F
from bankrag.tools.data import CaseData

THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_TOOL_TEXT = 6000
NO_DATA = "Не могу ответить по данным: нужные функции не вернули результат."

SYSTEM_PROMPT = """Ты — BankRAG, помощник аналитика антифрода банка. Отвечай по-русски, коротко и по делу.

Ответ — 2–4 предложения: главный вывод и ключевые числа. Списки шагов, отправителей и счетов не переписывай: код покажет их таблицей под ответом.

Правила:
1. Все числа, суммы, даты, время и номера счетов бери только из результатов функций. Сам ничего не считай: переписывай готовые строки из полей *_text и summary.
2. Если для ответа нужны данные, вызови функцию. Не выдумывай счета и суммы.
3. На вопрос о законе, статье или схеме всегда вызывай search_knowledge. Тексты законов не цитируй, не пересказывай и не называй номера статей по памяти: код покажет дословный текст под ответом. Назови только найденный документ по его заголовку, не описывай его смысл, и добавь, что правовую оценку даёт уполномоченный орган.
4. Ты не можешь блокировать или приостанавливать счета, звонить клиентам, записывать решения «подтвердил» или «отклонил». Если просят, вежливо откажи: решение и приостановку делает аналитик в системах банка. Предложи подготовить справку.
5. Не давай юридических заключений.
6. Если данных нет, так и скажи. О владельцах счетов (имя, адрес, счета в других банках) данных нет.
7. Если вопрос не про антифрод и переводы, вежливо откажи.
8. «Этот счёт», «он», «туда» — последний обсуждаемый счёт: {focus}.
9. Если просят показать на графе или нарисовать, вызови show_graph.
10. Не называй функции в ответе и не предлагай их вызвать.
11. Справку собирай по всем счетам дела: {case}.

Данные: {dataset}. Переводы с {start} по {end}."""
FORCE_TOOL_HINT = "Сначала вызови подходящую функцию: отвечать можно только по её результату."
# Вопросы о счёте или законе без вызова функции получили бы выдуманный ответ. Просьбы заблокировать
# счёт сюда не входят: на них помощник сразу отказывает.
NEEDS_TOOL = re.compile(r"KZ-?\d{4}|\d+_[0-9A-F]{6,}|стать|закон|норм|УК\b", re.IGNORECASE)
REFUSAL_REQUEST = re.compile(r"блок|приостан|заморо|арест|звон", re.IGNORECASE)


@dataclass
class Turn:
    """Один ход разговора со всем, что нужно интерфейсу и журналу."""

    question: str
    answer: str
    calls: list[dict] = field(default_factory=list)
    checked: bool = True
    retried: bool = False
    fallback: bool = False
    unknown: list[str] = field(default_factory=list)
    seconds: float = 0.0
    mode: str = "llm"
    graph: dict | None = None
    show_graph: bool = False
    fragments: list[dict] = field(default_factory=list)
    report_path: str | None = None
    accounts: list[str] = field(default_factory=list)


def run_tool(data: CaseData, name: str, arguments: dict, session: "ChatSession | None" = None) -> dict:
    """Выполняет функцию по имени. Ошибка в аргументах возвращается как результат, а не исключение."""
    arguments = {k: v for k, v in (arguments or {}).items() if v not in (None, "")}
    try:
        if name == "show_graph":
            graph = session.last_graph if session else None
            if not graph:
                text = "Графа пока нет: сначала проследите деньги, отправителей или сеть."
                return {"error": text, "summary": text, "accounts": []}
            text = (f"Граф «{graph['title']}»: {fmt.plural(len(graph['nodes']), 'счёт', 'счёта', 'счетов')}, "
                    f"{fmt.plural(len(graph['edges']), 'связь', 'связи', 'связей')}. Красный — подозрительно, "
                    "оранжевый — стоит проверить, синий — пострадавший.")
            return {"summary": text, "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]),
                    "graph": graph, "accounts": [n["id"] for n in graph["nodes"]]}
        function = getattr(F, name) if name in TOOL_NAMES else None
        if function is None:
            text = f"Функции {name} нет."
            return {"error": text, "summary": text, "accounts": []}
        return function(data, **arguments)
    except (TypeError, ValueError, KeyError) as error:
        text = f"Не удалось выполнить {name}: {error}"
        return {"error": text, "summary": text, "accounts": []}


# Длинные списки модель не получает: их показывает таблица под ответом, а модель пишет только вывод.
# Так ответ короче и быстрее, а закон модель не может процитировать, потому что не видит текста.
HIDDEN_FROM_MODEL = {"graph", "path", "steps", "chain", "senders", "accounts_detail", "similar", "fragments", "rows"}


def _for_model(name: str, result: dict) -> str:
    """Короткая версия результата для модели: итоговые числа и готовые строки."""
    brief = {k: v for k, v in result.items() if k not in HIDDEN_FROM_MODEL}
    if name == "trace_money" and "summary" in brief:
        brief["summary"] = brief["summary"].split("\n")[-1]
        brief["chain_accounts"] = [line["account"] for line in result.get("chain", [])]
    elif name == "trace_sources":
        brief["complaint_senders"] = [s["account"] for s in result.get("senders", []) if s["complaint"]]
    elif name == "list_suspicious":
        brief["top"] = [{k: r[k] for k in ("account", "flags", "main_reason")} for r in result.get("rows", [])[:5]]
        brief["summary"] = brief.get("summary", "").split("\n")[0]
    elif name == "find_similar":
        brief["top_similar"] = [r["account"] for r in result.get("similar", [])[:5]]
    text = json.dumps(brief, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_TOOL_TEXT else text[:MAX_TOOL_TEXT] + "…"


def template_answer(calls: list[dict]) -> str:
    """Ответ по шаблону: summary каждой вызванной функции."""
    parts = [call["result"].get("summary", "") for call in calls if call["result"].get("summary")]
    return "\n\n".join(parts) or NO_DATA


class ChatSession:
    """Разговор одного аналитика: история, последние результаты функций, счёт в фокусе."""

    def __init__(self, data: CaseData, model: ChatModel | None, config: dict, journal: Path | None = None,
                 analyst: str = ""):
        self.data = data
        self.model = model
        self.config = config
        self.journal = journal
        self.analyst = analyst
        self.history: list[dict] = []
        self.turns: list[Turn] = []
        self.last_graph: dict | None = None
        self.last_accounts: list[str] = []
        self.case_accounts: list[str] = []
        self.previous = Allowed()

    @property
    def focus(self) -> str:
        return self.last_accounts[0] if self.last_accounts else "пока нет"

    def _system(self) -> dict:
        data = self.data
        case = ", ".join(self.case_accounts[:15] or self.last_accounts[:12]) or "пока нет"
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            focus=self.focus, case=case, dataset=data.title, start=fmt.when(data.start), end=fmt.when(data.end))}

    def _remember(self, turn: Turn) -> Turn:
        for call in turn.calls:
            result = call["result"]
            if result.get("graph"):
                self.last_graph = turn.graph = result["graph"]
                turn.show_graph = turn.show_graph or call["name"] == "show_graph"
            if result.get("fragments"):
                turn.fragments = result["fragments"]
            if result.get("path"):
                turn.report_path = result["path"]
            accounts = [a for a in result.get("accounts", []) if a != F.CASH_NODE]
            if accounts and call["name"] != "show_graph":
                turn.accounts = accounts
                # В фокусе первый счёт ответа, кроме пострадавшего в начале цепочки.
                # У справки и сети нет поля account, а список начинается с пострадавших.
                focus = result.get("account") or next((a for a in accounts if a not in self.data.victims), accounts[0])
                self.last_accounts = [focus, *[a for a in accounts if a != focus]]
            # Счета дела: цепочка денег и пострадавшие с жалобами. По ним собирается справка.
            if call["name"] == "trace_money" and not result.get("error"):
                self.case_accounts = list(dict.fromkeys([*self.case_accounts, *accounts]))
            if call["name"] == "trace_sources" and not result.get("error"):
                victims = [s["account"] for s in result.get("senders", []) if s["complaint"]]
                self.case_accounts = list(dict.fromkeys([*self.case_accounts, *victims]))
        self.previous = Allowed()
        for call in turn.calls:
            self.previous.add(call["result"])
        self.history += [{"role": "user", "content": turn.question}, {"role": "assistant", "content": turn.answer}]
        self.history = self.history[-2 * self.config["chat"]["history_turns"]:]
        self.turns.append(turn)
        if self.journal is not None:
            from bankrag.chat.journal import log_query
            log_query(self.journal, self.analyst, self.data.name, turn.mode, turn.question, turn.calls, turn.answer,
                      turn.checked, turn.fallback, turn.seconds)
        return turn

    def run_buttons(self, question: str, name: str, arguments: dict) -> Turn:
        """Запасной режим: функция вызывается напрямую, ответ по шаблону."""
        started = time.perf_counter()
        result = run_tool(self.data, name, arguments, self)
        calls = [{"name": name, "arguments": arguments, "result": result}]
        return self._remember(Turn(question, template_answer(calls), calls, mode="buttons",
                                   seconds=time.perf_counter() - started))

    def ask(self, question: str) -> Turn:
        """Ответ на вопрос аналитика через модель с проверкой чисел."""
        if self.model is None:
            raise LLMError("модель не подключена")
        started = time.perf_counter()
        limit = self.config["llm"]["max_tool_calls"]
        messages = [self._system(), *self.history, {"role": "user", "content": question}]
        calls: list[dict] = []
        reply = self.model.chat(messages, TOOL_SPECS)
        if not reply.tool_calls and NEEDS_TOOL.search(question) and not REFUSAL_REQUEST.search(question):
            # Вопрос о конкретном счёте без вызова функции — ответ был бы выдуман.
            messages += [{"role": "assistant", "content": reply.content or ""},
                         {"role": "user", "content": FORCE_TOOL_HINT}]
            reply = self.model.chat(messages, TOOL_SPECS)
        while reply.tool_calls and len(calls) < limit:
            messages.append({"role": "assistant", "content": reply.content or "",
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in reply.tool_calls]})
            for call in reply.tool_calls:
                if len(calls) >= limit:
                    break
                result = run_tool(self.data, call["name"], call["arguments"], self)
                calls.append({"name": call["name"], "arguments": call["arguments"], "result": result})
                messages.append({"role": "tool", "tool_name": call["name"], "content": _for_model(call["name"], result)})
            # После лимита вызовов модель только пишет ответ.
            reply = self.model.chat(messages, TOOL_SPECS if len(calls) < limit else None)
        answer = THINK_TAG.sub("", reply.content or "").strip()

        allowed = Allowed()
        allowed.add_text(question)
        allowed.numbers |= self.previous.numbers
        allowed.times |= self.previous.times
        allowed.accounts |= self.previous.accounts
        for call in calls:
            allowed.add(call["result"])
        check = check_answer(answer, allowed)
        turn = Turn(question, answer, calls, unknown=check.unknown)
        if not answer or not check.ok:
            hint = ("В ответе есть числа или счета, которых нет в результатах функций: " + ", ".join(check.unknown)
                    + ". Перепиши ответ, используя только числа и счета из результатов.") if answer else \
                "Ответ пустой. Напиши ответ по результатам функций."
            retry = self.model.chat([*messages, {"role": "assistant", "content": answer},
                                     {"role": "user", "content": hint}])
            turn.retried = True
            answer = THINK_TAG.sub("", retry.content or "").strip()
            check = check_answer(answer, allowed)
            turn.unknown = check.unknown
            if answer and check.ok:
                turn.answer = answer
            else:
                turn.answer, turn.fallback, turn.checked = template_answer(calls), True, False
        turn.seconds = time.perf_counter() - started
        return self._remember(turn)
