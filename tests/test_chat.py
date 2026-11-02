"""Тесты чата без Ollama: модель подменяется заглушкой с заранее заданными ответами."""
from __future__ import annotations

from pathlib import Path

from bankrag.chat.agent import ChatSession, run_tool, template_answer
from bankrag.chat.checker import Allowed, check_answer, extract
from bankrag.chat.journal import read_queries
from bankrag.chat.llm import LLMReply
from bankrag.data_loader import load_config
from bankrag.tools.data import CaseData


class StubModel:
    """Отдаёт ответы по очереди и запоминает, что ей прислали."""

    name = "stub"

    def __init__(self, replies: list[LLMReply]):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        self.requests.append(list(messages))
        return self.replies.pop(0)


def _call(name: str, **arguments) -> LLMReply:
    return LLMReply("", [{"name": name, "arguments": arguments}])


def test_extract_handles_spaces_commas_units_and_times() -> None:
    numbers, times, accounts = extract("1. В 14:05 KZ-1001 получил 500 000 ₸, ушло 96%, всего 2,1 млн ₸ за 12 мин.")
    assert [value for _, value in numbers] == [500_000, 96, 2_100_000, 12]
    assert times == ["14:05"] and accounts == ["KZ-1001"]


def test_check_answer_accepts_known_and_rejects_invented() -> None:
    allowed = Allowed()
    allowed.add({"amount": 480000.0, "share": 0.96, "total_short_text": "2,1 млн ₸", "time_text": "3 ноября 14:18",
                 "accounts": ["KZ-1001"]})
    assert check_answer("KZ-1001 перевёл 480 000 ₸ (96%) 3 ноября в 14:18, всего 2,1 млн ₸.", allowed).ok
    bad = check_answer("KZ-1001 перевёл 490 000 ₸ в 14:19 на KZ-5555.", allowed)
    assert not bad.ok and set(bad.unknown) == {"490 000", "14:19", "KZ-5555"}


def test_chat_calls_tool_and_passes_check(data: CaseData) -> None:
    model = StubModel([_call("trace_money", account="KZ-1001", start_time="2026-11-03 10:00", amount=100000),
                       LLMReply("Деньги ушли с KZ-1001 на KZ-1002 через 12 мин, сняли наличными 91 000 ₸.")])
    turn = ChatSession(data, model, load_config()).ask("Клиентка в 10:00 перевела 100 000 ₸ на KZ-1001. Куда ушли деньги?")
    assert [c["name"] for c in turn.calls] == ["trace_money"]
    assert turn.checked and not turn.retried and not turn.fallback
    assert turn.graph is not None and not turn.show_graph
    # Результат функции ушёл модели сообщением с ролью tool.
    assert model.requests[1][-1]["role"] == "tool"


def test_invented_number_triggers_retry_then_template(data: CaseData) -> None:
    model = StubModel([_call("account_profile", account="KZ-1002"),
                       LLMReply("На счёт пришло 777 777 ₸."), LLMReply("Нет, 888 888 ₸.")])
    turn = ChatSession(data, model, load_config()).ask("Почему KZ-1002 подозрителен?")
    assert turn.retried and turn.fallback and not turn.checked
    assert turn.answer == template_answer(turn.calls)
    assert "777 777" not in turn.answer and "KZ-1002" in turn.answer


def test_retry_fixes_answer(data: CaseData) -> None:
    model = StubModel([_call("trace_sources", account="KZ-1001", window_hours=120),
                       LLMReply("Пришло 9 переводов."), LLMReply("Пришло 6 переводов от 6 отправителей.")])
    turn = ChatSession(data, model, load_config()).ask("Кто ещё переводил на KZ-1001?")
    assert turn.retried and turn.checked and not turn.fallback
    assert turn.answer == "Пришло 6 переводов от 6 отправителей."


def test_refusal_without_tools_and_context_for_graph(data: CaseData, tmp_path: Path) -> None:
    journal = tmp_path / "journal.sqlite"
    session = ChatSession(data, None, load_config(), journal=journal, analyst="Альнур")
    session.model = StubModel([
        _call("trace_money", account="KZ-1001"), LLMReply("Цепочка найдена."),
        _call("show_graph"), LLMReply("Показал граф."),
        LLMReply("Я не блокирую счета: решение и приостановку делает аналитик в системах банка.")])
    session.ask("Куда ушли деньги с KZ-1001?")
    assert session.focus == "KZ-1001"
    graph_turn = session.ask("Покажи на графе")
    assert graph_turn.show_graph and graph_turn.graph["nodes"]
    refusal = session.ask("Заблокируй этот счёт")
    assert refusal.calls == [] and refusal.checked
    log = read_queries(journal)
    assert len(log) == 3 and set(log["analyst"]) == {"Альнур"}


def test_tool_limit_per_turn(data: CaseData) -> None:
    config = load_config()
    many = [_call("account_profile", account="KZ-1001") for _ in range(6)]
    model = StubModel([*many[:config["llm"]["max_tool_calls"]], LLMReply("Готово.")])
    turn = ChatSession(data, model, config).ask("Проверь KZ-1001 много раз")
    assert len(turn.calls) == config["llm"]["max_tool_calls"]
    # Последний запрос после лимита идёт без функций.
    assert turn.answer == "Готово."


def test_buttons_mode_and_bad_arguments(data: CaseData) -> None:
    session = ChatSession(data, None, load_config())
    turn = session.run_buttons("Откуда пришли деньги на KZ-1001", "trace_sources", {"account": "KZ-1001",
                                                                                     "window_hours": 120})
    assert turn.mode == "buttons" and "KZ-2001" in turn.answer
    assert "Не удалось" in run_tool(data, "trace_money", {"wrong": 1})["summary"]
    assert "нет" in run_tool(data, "block_account", {})["summary"]



def test_question_about_account_forces_tool_call(data: CaseData) -> None:
    model = StubModel([LLMReply("Счёт подозрителен, риск высокий."), _call("account_profile", account="KZ-1002"),
                       LLMReply("У KZ-1002 есть флажок дробления.")])
    turn = ChatSession(data, model, load_config()).ask("Почему KZ-1002 подозрителен?")
    assert [c["name"] for c in turn.calls] == ["account_profile"]
    assert model.requests[1][-1]["content"].startswith("Сначала вызови")
