"""BankRAG: чат-помощник аналитика антифрода.

Аналитик пишет вопрос, локальная модель (Ollama) вызывает функции, код считает все числа.
Если Ollama не запущена, работает режим кнопок: те же функции, ответ по шаблону.
LightGBM здесь не загружается: оценки читаются из data/processed.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bankrag.faiss_safe import faiss  # noqa: E402, F401  # должен загружаться раньше scikit-learn

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bankrag.chat.agent import ChatSession, Turn  # noqa: E402
from bankrag.chat.journal import read_queries  # noqa: E402
from bankrag.chat.llm import LLMError, OllamaClient  # noqa: E402
from bankrag.data_loader import load_config  # noqa: E402
from bankrag.evaluate import render_markdown  # noqa: E402
from bankrag.feedback import CONFIRMED, REJECTED, all_latest_decisions, read_decisions, save_decision  # noqa: E402
from bankrag.flags import COLOR_HEX, COLOR_TITLES  # noqa: E402
from bankrag.graph_view import case_graph_html  # noqa: E402
from bankrag.kb_index import KnowledgeBase, e5_embedder  # noqa: E402
from bankrag.similarity import SimilarityIndex  # noqa: E402
from bankrag.tools import fmt  # noqa: E402
from bankrag.tools import functions as F  # noqa: E402
from bankrag.tools.data import DATASET_TITLES, IBM, KZ, CaseData, load_case_data  # noqa: E402

BANNER = ("Прототип. Данные синтетические (IBM AML, NeurIPS 2023) и сценарии, сгенерированные командой. "
          "Решение принимает аналитик.")
FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback.sqlite"
JOURNAL_PATH = ROOT / "data" / "feedback" / "journal.sqlite"
SIMILARITY_INDEX = {KZ: "similarity_kz", IBM: "similarity_test"}
DOTS = {"red": "🔴", "orange": "🟠", "grey": "⚪", "green": "🟢", "blue": "🔵"}
DECISION_TITLES = {CONFIRMED: "подтвердил", REJECTED: "отклонил"}
MAX_DECISION_ACCOUNTS = 8


@st.cache_resource(show_spinner="Загружаю базу знаний и модель эмбеддингов…")
def load_knowledge() -> KnowledgeBase:
    return KnowledgeBase.load(ROOT / "models", e5_embedder())


@st.cache_resource(show_spinner="Загружаю переводы и считаю флажки…")
def load_data(name: str) -> CaseData:
    data = load_case_data(name, load_config(), ROOT)
    data.similarity = SimilarityIndex.load(ROOT / "models", SIMILARITY_INDEX[name])
    data.knowledge = load_knowledge()
    return data


def model_status(config: dict, model_name: str) -> tuple[OllamaClient, bool]:
    """Клиент Ollama и готова ли модель. Проверка быстрая, без запроса к модели."""
    client = OllamaClient.from_config(config["llm"], model_name)
    return client, client.is_ready()


def session_for(data: CaseData, client: OllamaClient | None, config: dict, analyst: str) -> ChatSession:
    """Разговор хранится в состоянии страницы отдельно для каждого набора данных."""
    key = f"chat_{data.name}"
    if key not in st.session_state:
        st.session_state[key] = ChatSession(data, client, config, JOURNAL_PATH, analyst)
    session: ChatSession = st.session_state[key]
    session.model, session.analyst = client, analyst
    return session


def quote(text: str) -> str:
    """Цитата в markdown: номер части статьи остаётся текстом, а не становится маркером списка."""
    lines = [re.sub(r"^(\d+)\.", r"\1\\.", line) for line in text.split("\n")]
    return "> " + "  \n> ".join(lines)


def dot(color: str) -> str:
    return f"{DOTS.get(color, '')} {COLOR_TITLES.get(color, '')}"


def render_legend() -> None:
    items = " ".join(f"<span style='display:inline-block;width:12px;height:12px;border-radius:6px;"
                     f"background:{COLOR_HEX[c]};margin:0 4px 0 12px'></span>{COLOR_TITLES[c]}"
                     for c in ("red", "orange", "grey", "green", "blue"))
    st.markdown(f"<div style='font-size:13px'>{items}<span style='margin-left:12px'>"
                "▭ снятие наличных · толстая рамка — подтверждён аналитиком</span></div>", unsafe_allow_html=True)


def render_details(call: dict, data: CaseData) -> None:
    """Таблица к результату функции: шаги, отправители, сеть и т. п. Числа те же, что в результате."""
    result, name = call["result"], call["name"]
    if result.get("error"):
        return
    table = None
    if name == "trace_money":
        table = pd.DataFrame([{"№": s["step"], "Время": s["time_text"], "Откуда": s["from"], "Куда": s["to"],
                               "Сумма": s["amount_text"], "Через": s.get("delay_text") or "—",
                               "Доля": s.get("share_text") or "—"} for s in result["steps"]])
    elif name == "trace_sources":
        table = pd.DataFrame([{"Отправитель": s["account"], "Оценка": dot(s["color"]), "Переводов": s["transfers"],
                               "Сумма": s["total_text"], "Первый перевод": s["first_time_text"],
                               "Жалоба": "есть" if s["complaint"] else ""} for s in result["senders"]])
    elif name == "find_network":
        table = pd.DataFrame([{"Счёт": a["account"], "Оценка": dot(a["color"]), "Флажков": a["flags"]}
                              for a in result["accounts_detail"]])
    elif name == "find_similar":
        st.caption("Похожий почерк: эти счета не обязательно связаны переводами с выбранным.")
        table = pd.DataFrame([{"Счёт": r["account"], "Оценка": dot(r["color"]), "Флажков": r["flags"],
                               "Близость": r["similarity_text"],
                               "Связан переводами": "да" if r["linked_by_transfers"] else "нет"} for r in result["similar"]])
    elif name == "list_suspicious":
        table = pd.DataFrame([{"Счёт": r["account"], "Оценка": dot(r["color"]), "Флажков": r["flags"],
                               "Риск модели": r["risk_text"], "Главная причина": r["main_reason"]} for r in result["rows"]])
    elif name == "account_profile":
        st.markdown(f"**{result['account']}** — {dot(result['color'])}")
        for phrase in result["flag_phrases"] or ["флажков нет"]:
            st.markdown(f"- {phrase}")
        if result.get("history_text"):
            st.caption(result["history_text"].capitalize())
        st.caption(f"Оценка модели риска: {result['risk_text']} ({result['risk_level']}).")
    elif name == "search_knowledge":
        st.markdown("**Фрагменты базы знаний (дословно)**")
        for fragment in result["fragments"]:
            st.markdown(quote(fragment["text"]))
            st.caption(f"{fragment['title']}. Источник: {fragment['source']}")
        st.caption("Текст приведён для сведения. Правовую оценку даёт уполномоченный орган.")
    if table is not None and len(table):
        st.dataframe(table, hide_index=True, width="stretch")


def render_decisions(turn_index: int, accounts: list[str], data: CaseData, analyst: str) -> None:
    """Кнопки решения у счетов ответа. Помощник решения не записывает, только человек."""
    decisions = all_latest_decisions(FEEDBACK_PATH)
    shown = [a for a in accounts if a not in data.victims and data.has(a)][:MAX_DECISION_ACCOUNTS]
    if not shown:
        return
    st.markdown("**Решение аналитика**")
    for account in shown:
        name, confirm, reject, state = st.columns([3, 1, 1, 2])
        name.markdown(f"{DOTS.get(data.color(account), '')} {account}")
        if confirm.button("Подтвердить", key=f"confirm_{turn_index}_{account}"):
            save_decision(FEEDBACK_PATH, account, CONFIRMED, data.risk(account), analyst, data.name)
            decisions[account] = CONFIRMED
        if reject.button("Отклонить", key=f"reject_{turn_index}_{account}"):
            save_decision(FEEDBACK_PATH, account, REJECTED, data.risk(account), analyst, data.name)
            decisions[account] = REJECTED
        if account in decisions:
            state.markdown(f"сохранено: {DECISION_TITLES[decisions[account]]}")


def render_turn(index: int, turn: Turn, session: ChatSession, data: CaseData, analyst: str) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        if turn.fallback:
            st.caption("Ответ модели не прошёл проверку чисел, показан ответ по шаблону из результата функции.")
        elif turn.mode == "buttons":
            st.caption("Режим кнопок: ответ собран по шаблону из результата функции.")
        for call in turn.calls:
            render_details(call, data)

        buttons = st.columns([1, 1, 4])
        graph_key = f"graph_open_{index}"
        if turn.graph and buttons[0].button("Граф", key=f"graph_{index}"):
            st.session_state[graph_key] = not st.session_state.get(graph_key, turn.show_graph)
        if turn.graph and st.session_state.get(graph_key, turn.show_graph):
            st.markdown(f"**{turn.graph['title']}**")
            # Решения могли появиться после ответа: рамку подтверждённых счетов берём на момент показа.
            graph = {**turn.graph, "nodes": [{**n, "confirmed": n["id"] in data.confirmed} for n in turn.graph["nodes"]]}
            st.iframe(case_graph_html(graph), height=540)
            render_legend()
        accounts = [a for a in turn.accounts if a != F.CASH_NODE]
        report_path = turn.report_path
        if accounts and not report_path and buttons[1].button("Справка", key=f"report_{index}"):
            result = F.make_report(data, accounts, "Справка по делу")
            report_path = turn.report_path = result.get("path")
        if report_path and Path(report_path).exists():
            buttons[1].download_button("Скачать DOCX", Path(report_path).read_bytes(), file_name=Path(report_path).name,
                                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       key=f"download_{index}")
        if accounts:
            render_decisions(index, accounts, data, analyst)
        with st.expander("Как получен ответ"):
            st.caption(f"Режим: {'модель ' + (session.model.name if session.model else '') if turn.mode == 'llm' else 'кнопки'}"
                       f" · {turn.seconds:.1f} с · проверка чисел: {'пройдена' if turn.checked else 'не пройдена'}"
                       + (" со второй попытки" if turn.retried and turn.checked else ""))
            if turn.unknown:
                st.caption("Числа, которых не было в результатах: " + ", ".join(turn.unknown))
            if not turn.calls:
                st.write("Функции не вызывались.")
            for call in turn.calls:
                st.markdown(f"**{call['name']}** `{json.dumps(call['arguments'], ensure_ascii=False)}`")
                st.json({k: v for k, v in call["result"].items() if k not in ("graph",)}, expanded=False)


def demo_questions() -> list[str]:
    """Вопросы демо с реальными номерами счетов из reports/demo_case.json."""
    path = ROOT / "reports" / "demo_case.json"
    if not path.exists():
        return []
    demo = json.loads(path.read_text(encoding="utf-8"))
    when = pd.Timestamp(demo["victim_time"])
    first = demo["first_drop"]
    return [f"Клиентка {when.day} {fmt.MONTHS[when.month - 1]} в {when:%H:%M} перевела "
            f"{fmt.money(demo['victim_tenge'], 'KZT')} на {first}. Куда ушли деньги?",
            "Покажи на графе", f"Кто ещё переводил на {first}?", "Какая статья к этому относится?",
            "Подготовь справку для полиции", f"Заблокируй {first}"]


def render_buttons_mode(session: ChatSession, data: CaseData) -> None:
    """Запасной режим без модели: те же функции по кнопкам."""
    demo_path = ROOT / "reports" / "demo_case.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8")) if data.name == KZ and demo_path.exists() else {}
    columns = st.columns(3)
    account = columns[0].text_input("Счёт", value=demo.get("first_drop", ""), key=f"btn_account_{data.name}")
    start = columns[1].text_input("Время перевода", value=str(demo.get("victim_time", ""))[:16],
                                  key=f"btn_time_{data.name}")
    amount = columns[2].text_input("Сумма", value=str(int(demo["victim_tenge"])) if demo else "",
                                   key=f"btn_amount_{data.name}")
    actions = st.columns(4)
    if actions[0].button("Проследить деньги", width="stretch"):
        session.run_buttons(f"Проследить деньги: {account}", "trace_money",
                            {"account": account, "start_time": start, "amount": amount})
    if actions[1].button("Откуда пришли деньги", width="stretch"):
        session.run_buttons(f"Откуда пришли деньги: {account}", "trace_sources", {"account": account})
    if actions[2].button("Показать сеть", width="stretch"):
        session.run_buttons(f"Показать сеть: {account}", "find_network", {"accounts": [account]})
    if actions[3].button("Справка", width="stretch"):
        accounts = session.last_accounts or [account]
        session.run_buttons(f"Справка: {', '.join(accounts)}", "make_report",
                            {"accounts": accounts, "case_title": "Справка по делу"})


def render_chat(session: ChatSession, data: CaseData, ready: bool, analyst: str) -> None:
    if not ready:
        st.warning("Модель не запущена, работает режим кнопок. Запуск: `brew services start ollama`.")
        render_buttons_mode(session, data)
    for index, turn in enumerate(session.turns):
        render_turn(index, turn, session, data, analyst)

    question = None
    if ready and data.name == KZ:
        st.caption("Быстрые вопросы для демо")
        quick = demo_questions()
        columns = st.columns(len(quick)) if quick else []
        for column, text in zip(columns, quick):
            if column.button(text if len(text) < 40 else text[:37] + "…", help=text, key=f"quick_{text}",
                             width="stretch"):
                question = text
    typed = st.chat_input("Спросите, например: куда ушли деньги клиентки?", disabled=not ready)
    question = typed or question
    if question:
        try:
            with st.spinner("Модель выбирает функции и пишет ответ…"):
                session.ask(question)
        except LLMError as error:
            st.error(f"Модель не ответила: {error}. Используйте режим кнопок.")
        st.rerun()


def render_queue(data: CaseData, analyst: str) -> None:
    """Очередь: счета по числу флажков и оценке модели, под ней профиль выбранного."""
    table = data.flags[~data.flags.index.isin(data.businesses | data.victims)].copy()
    table["risk"] = data.scores["risk_score"].reindex(table.index).fillna(0)
    table = table[(table["flag_count"] > 0) | (table["risk"] >= data.high_risk * data.config["colors"]["medium_risk_share"])]
    table = table.sort_values(["flag_count", "risk"], ascending=False).head(data.config["app"]["queue_rows"])
    decisions = all_latest_decisions(FEEDBACK_PATH)
    view = pd.DataFrame({
        "Счёт": table.index, "Оценка": [dot(data.color(a)) for a in table.index], "Флажков": table["flag_count"].to_numpy(),
        "Риск модели": table["risk"].to_numpy(),
        "Главная причина": [(F.flag_phrases(data, a) or data.model_reasons(a) or ["—"])[0] for a in table.index],
        "Решение": [DECISION_TITLES.get(decisions.get(a), "") for a in table.index]})
    st.caption(f"Счетов с флажками или заметным риском: {len(view)} из {len(data.flags)}. Выберите строку.")
    selection = st.dataframe(view, hide_index=True, width="stretch", height=360, on_select="rerun",
                             selection_mode="single-row", key=f"queue_{data.name}",
                             column_config={"Риск модели": st.column_config.NumberColumn(format="%.3f")})
    if not selection.selection.rows:
        return
    account = str(view.iloc[selection.selection.rows[0]]["Счёт"])
    profile = F.account_profile(data, account)
    render_details({"name": "account_profile", "result": profile}, data)
    network = F.find_network(data, [account], depth=1)
    st.markdown(f"**{network['graph']['title']}**")
    st.iframe(case_graph_html(network["graph"]), height=480)
    render_legend()
    render_decisions(-1, [account], data, analyst)


def render_metrics() -> None:
    """Метрики читаются из reports/, числа здесь не считаются."""
    path = ROOT / "reports" / "metrics.json"
    chat = ROOT / "reports" / "chat_metrics.md"
    if chat.exists():
        st.markdown(chat.read_text(encoding="utf-8"))
        st.divider()
    if path.exists():
        st.markdown(render_markdown(json.loads(path.read_text(encoding="utf-8"))))
    else:
        st.warning("Файла reports/metrics.json нет: сначала запустите make eval.")


def render_journal() -> None:
    st.markdown("**Запросы к помощнику**")
    queries = read_queries(JOURNAL_PATH)
    if len(queries):
        queries["calls"] = queries["calls"].map(lambda text: ", ".join(c["name"] for c in json.loads(text or "[]")))
        st.dataframe(queries.rename(columns={
            "created_at": "Когда (UTC)", "analyst": "Аналитик", "dataset": "Набор", "mode": "Режим", "question": "Вопрос",
            "calls": "Функции", "answer": "Ответ", "checked": "Числа проверены", "fallback": "Шаблон",
            "seconds": "Секунд"}).drop(columns="id"), hide_index=True, width="stretch")
    else:
        st.caption("Запросов пока нет.")
    st.markdown("**Решения аналитиков**")
    decisions = read_decisions(FEEDBACK_PATH)
    if len(decisions):
        decisions["decision"] = decisions["decision"].map(DECISION_TITLES)
        st.dataframe(decisions.rename(columns={"created_at": "Когда (UTC)", "account_id": "Счёт", "decision": "Решение",
                                               "risk_score": "Риск", "analyst": "Аналитик", "dataset": "Набор"})
                     .drop(columns="id"), hide_index=True, width="stretch")
    else:
        st.caption("Решений пока нет.")


def main() -> None:
    st.set_page_config(page_title="BankRAG", layout="wide")
    config = load_config()
    with st.sidebar:
        st.header("BankRAG")
        analyst = st.text_input("Аналитик", value=st.session_state.get("analyst", "Аналитик"), key="analyst")
        dataset = st.radio("Набор данных", [KZ, IBM], format_func=DATASET_TITLES.get)
        model_name = st.selectbox("Модель", [config["llm"]["model"], config["llm"]["fallback_model"]])
        client, ready = model_status(config, model_name)
        st.caption(f"Модель {model_name}: {'готова' if ready else 'недоступна'}")
        st.caption("Помощник не блокирует счета и не записывает решения: это делает аналитик.")
    st.warning(BANNER)
    st.title("BankRAG: чат-помощник аналитика антифрода")
    data = load_data(dataset)
    confirmed = {a for a, d in all_latest_decisions(FEEDBACK_PATH).items() if d == CONFIRMED}
    data.set_confirmed(confirmed)
    session = session_for(data, client if ready else None, config, analyst)
    chat_tab, queue_tab, metrics_tab, journal_tab = st.tabs(["Чат", "Очередь", "Метрики", "Журнал"])
    with chat_tab:
        render_chat(session, data, ready, analyst)
    with queue_tab:
        render_queue(data, analyst)
    with metrics_tab:
        render_metrics()
    with journal_tab:
        render_journal()


if __name__ == "__main__":
    main()
