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
from pandas.io.formats.style import Styler  # noqa: E402
import streamlit as st  # noqa: E402

from bankrag.chat.agent import ChatSession, Turn, template_answer  # noqa: E402
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
from bankrag.tools.timeline import timeline_chart  # noqa: E402

BANNER = ("Прототип. Данные синтетические (IBM AML, NeurIPS 2023) и сценарии, сгенерированные командой. "
          "Решение принимает аналитик.")
FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback.sqlite"
JOURNAL_PATH = ROOT / "data" / "feedback" / "journal.sqlite"
SIMILARITY_INDEX = {KZ: "similarity_kz", IBM: "similarity_test"}
MARK = ROOT / "app" / "assets" / "bankrag-mark.png"
# Подпись уровня темнее заливки на графе: текст на белом должен читаться (контраст от 4,5:1).
LABEL_HEX = {"red": "#8e1f25", "orange": "#7a4800", "grey": "#5a5e68", "green": "#1f5b4e", "blue": "#1f5168"}
LABEL_COLORS = {f"● {title}": LABEL_HEX[color] for color, title in COLOR_TITLES.items()}
DECISION_TITLES = {CONFIRMED: "подтвердил", REJECTED: "отклонил"}
MAX_DECISION_ACCOUNTS = 8
MAX_SUGGESTIONS = 4
# Стартовые подсказки для набора IBM: конкретных номеров счетов тут не назвать, они длинные и случайные.
GENERIC_STARTERS = ["Покажи 10 самых подозрительных счетов", "Кого проверить в первую очередь?",
                    "Что такое схема сбора (fan-in)?"]
GUARDRAIL = ("Помощник только показывает данные. Блокировку счёта, звонок клиенту и решение "
             "«подтвердил / отклонил» делает аналитик: у модели таких функций нет.")


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


def mark(color: str) -> str:
    """Уровень подозрения для таблицы: символ ● и подпись, цвет ставит styled()."""
    return f"● {COLOR_TITLES.get(color, '')}"


def mark_html(color: str, text: str) -> str:
    """Цветной кружок перед текстом для markdown: цвет из палитры графа, а не эмодзи."""
    return f"<span style='color:{COLOR_HEX.get(color, COLOR_HEX['grey'])}'>●</span> {text}"


def styled(table: pd.DataFrame) -> pd.DataFrame | Styler:
    """Таблица с колонкой «Оценка», окрашенной по уровню подозрения."""
    if "Оценка" not in table.columns:
        return table
    return table.style.map(lambda value: f"color: {LABEL_COLORS.get(value, 'inherit')}", subset=["Оценка"])


def render_legend() -> None:
    items = " ".join(f"<span style='display:inline-block;width:12px;height:12px;border-radius:6px;"
                     f"background:{COLOR_HEX[c]};margin:0 4px 0 12px'></span>{COLOR_TITLES[c]}"
                     for c in ("red", "orange", "grey", "green", "blue"))
    st.markdown(f"<div style='font-size:13px'>{items}<span style='margin-left:12px'>"
                "▭ снятие наличных · толстая рамка — подтверждён аналитиком</span></div>", unsafe_allow_html=True)


def render_numbers(pairs: list[tuple[str, str]]) -> None:
    """Главные числа результата функции: подпись и готовая строка из полей *_text."""
    for column, (title, value) in zip(st.columns(len(pairs)), pairs):
        column.metric(title, value)


def render_sources(turn: Turn) -> None:
    """Откуда числа ответа: какие функции их дали и прошёл ли ответ проверку."""
    if not turn.calls:
        return
    names = " · ".join(f"`{call['name']}`" for call in turn.calls)
    checked = "проверка чисел пройдена" if turn.checked else "числа не сошлись, показан шаблон"
    st.caption(f"Числа из функций: {names} — {checked}.")


def render_details(call: dict, data: CaseData) -> None:
    """Таблица к результату функции: шаги, отправители, сеть и т. п. Числа те же, что в результате."""
    result, name = call["result"], call["name"]
    if result.get("error"):
        return
    table = None
    if name == "trace_money":
        render_numbers([("Шагов", str(result["step_count"])), ("Счетов в цепочке", str(result["chain_count"])),
                        ("Снято наличными", result["cash_text"]), ("Осталось на счетах", result["retained_text"]),
                        ("Время цепочки", result["duration_text"])])
        chart = timeline_chart(result["steps"])
        if chart is not None:
            st.markdown("**Таймлайн: как быстро уходили деньги**")
            st.altair_chart(chart, width="stretch")
            st.caption("Полоса — сколько деньги лежали на счёте до следующего перевода, точка — поступление, "
                       "красный ромб — снятие наличных.")
        table = pd.DataFrame([{"№": s["step"], "Время": s["time_text"], "Откуда": s["from"], "Куда": s["to"],
                               "Сумма": s["amount_text"], "Через": s.get("delay_text") or "—",
                               "Доля": s.get("share_text") or "—"} for s in result["steps"]])
    elif name == "trace_sources":
        render_numbers([("Отправителей", str(result["sender_count"])), ("Переводов", str(result["transfer_count"])),
                        ("Сумма", result["total_short_text"]), ("С жалобами", str(result["complaint_count"]))])
        table = pd.DataFrame([{"Отправитель": s["account"], "Оценка": mark(s["color"]), "Переводов": s["transfers"],
                               "Сумма": s["total_text"], "Первый перевод": s["first_time_text"],
                               "Жалоба": "есть" if s["complaint"] else ""} for s in result["senders"]])
    elif name == "find_network":
        table = pd.DataFrame([{"Счёт": a["account"], "Оценка": mark(a["color"]), "Флажков": a["flags"]}
                              for a in result["accounts_detail"]])
    elif name == "find_similar":
        st.caption("Похожий почерк: эти счета не обязательно связаны переводами с выбранным.")
        table = pd.DataFrame([{"Счёт": r["account"], "Оценка": mark(r["color"]), "Флажков": r["flags"],
                               "Близость": r["similarity_text"],
                               "Связан переводами": "да" if r["linked_by_transfers"] else "нет"} for r in result["similar"]])
    elif name == "list_suspicious":
        table = pd.DataFrame([{"Счёт": r["account"], "Оценка": mark(r["color"]), "Флажков": r["flags"],
                               "Риск модели": r["risk_text"], "Главная причина": r["main_reason"]} for r in result["rows"]])
    elif name == "account_profile":
        st.markdown(f"**{result['account']}** — " + mark_html(result["color"], COLOR_TITLES[result["color"]]),
                    unsafe_allow_html=True)
        render_numbers([("Риск модели", f"{result['risk_text']} ({result['risk_level']})"),
                        ("Флажков", str(result["flag_count"])), ("Получено", result["received_text"]),
                        ("Отправлено", result["sent_text"])])
        for phrase in result.get("list_phrases", []):
            (st.warning if phrase.startswith("владелец похож") else st.error)(phrase)
        for phrase in result["flag_phrases"] or ["флажков нет"]:
            st.markdown(f"- {phrase}")
        if result.get("history_text"):
            st.caption(result["history_text"].capitalize())
        render_model_factors(result)
    elif name == "check_lists":
        render_lists(result)
    elif name == "search_knowledge":
        st.markdown("**Фрагменты базы знаний (дословно)**")
        for fragment in result["fragments"]:
            st.markdown(quote(fragment["text"]))
            st.caption(f"{fragment['title']}. Источник: {fragment['source']}")
        st.caption("Текст приведён для сведения. Правовую оценку даёт уполномоченный орган.")
    if table is not None and len(table):
        st.dataframe(styled(table), hide_index=True, width="stretch")


def render_model_factors(result: dict) -> None:
    """Что повысило и что понизило оценку модели: доли вкладов признаков LightGBM."""
    if not result.get("model_up") and not result.get("model_down"):
        return
    with st.expander(f"Почему у модели оценка {result['risk_text']}"):
        up, down = st.columns(2)
        for column, title, key in ((up, "Повысили оценку", "up"), (down, "Понизили оценку", "down")):
            with column:
                st.markdown(f"**{title}**")
                for factor in result.get(f"model_{key}", []):
                    st.markdown(f"- {factor['text']} — {factor['share_text']}")
                st.caption(f"Особенности данных IBM (валюты, форматы, суммы, переводы себе, связи соседей): "
                           f"{result[f'model_{key}_hidden_text']}")
        st.caption("Доля — часть суммы вкладов признаков (pred_contrib LightGBM) в эту сторону. Признаки модели "
                   "считаются за своё окно, поэтому числа могут немного отличаться от флажков.")


def render_decisions(turn_index: int, accounts: list[str], data: CaseData, analyst: str) -> None:
    """Кнопки решения у счетов ответа. Помощник решения не записывает, только человек."""
    decisions = all_latest_decisions(FEEDBACK_PATH)
    shown = [a for a in accounts if a not in data.victims and data.has(a)][:MAX_DECISION_ACCOUNTS]
    if not shown:
        return
    st.markdown("**Решение аналитика**")
    st.caption(GUARDRAIL + " Решение пишется в журнал с вашим логином и оценкой риска на момент нажатия.")
    for account in shown:
        name, confirm, reject, state = st.columns([3, 1, 1, 2])
        name.markdown(mark_html(data.color(account), account), unsafe_allow_html=True)
        if confirm.button("Подтвердить", key=f"confirm_{turn_index}_{account}"):
            save_decision(FEEDBACK_PATH, account, CONFIRMED, data.risk(account), analyst, data.name)
            decisions[account] = CONFIRMED
        if reject.button("Отклонить", key=f"reject_{turn_index}_{account}"):
            save_decision(FEEDBACK_PATH, account, REJECTED, data.risk(account), analyst, data.name)
            decisions[account] = REJECTED
        if account in decisions:
            state.markdown(f"сохранено: {DECISION_TITLES[decisions[account]]}")


def render_turn(index: int, turn: Turn, session: ChatSession, data: CaseData, analyst: str) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(turn.question)
    with st.chat_message("assistant", avatar=str(MARK)):
        st.markdown(turn.answer)
        if turn.fallback:
            st.caption("Ответ модели не прошёл проверку чисел, показан ответ по шаблону из результата функции.")
        elif turn.mode == "buttons":
            st.caption("Режим кнопок: ответ собран по шаблону из результата функции.")
        render_sources(turn)
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
            appendix = Path(report_path).with_suffix(".json")
            if appendix.exists():
                buttons[1].download_button("Скачать JSON", appendix.read_bytes(), file_name=appendix.name,
                                           mime="application/json", key=f"download_json_{index}")
        if accounts:
            render_decisions(index, accounts, data, analyst)
        with st.expander("Как получен ответ", expanded=True):
            st.caption(f"Режим: {'модель ' + (session.model.name if session.model else '') if turn.mode == 'llm' else 'кнопки'}"
                       + (f" · результат функции через {turn.tools_seconds:.1f} с" if turn.tools_seconds else "")
                       + f" · ответ через {turn.seconds:.1f} с · проверка чисел: {'пройдена' if turn.checked else 'не пройдена'}"
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
            "Подготовь справку для полиции", "Есть ли среди клиентов лица из перечня АФМ?", f"Заблокируй {first}"]


def follow_up_questions(session: ChatSession, data: CaseData) -> list[str]:
    """Подсказки для следующего вопроса: стартовые или по последнему ответу.

    Счёт в подсказке — тот, что обсуждается сейчас, поэтому вопрос можно задать одним нажатием.
    Уже вызванные функции не предлагаются второй раз.
    """
    if not session.turns:
        return demo_questions() if data.name == KZ else GENERIC_STARTERS
    focus, called = session.focus, {call["name"] for turn in session.turns for call in turn.calls}
    questions = []
    if session.turns[-1].graph and not session.turns[-1].show_graph:
        questions.append("Покажи на графе")
    if focus != "пока нет":
        if "trace_money" not in called:
            questions.append(f"Куда ушли деньги с {focus}?")
        if "trace_sources" not in called:
            questions.append(f"Кто ещё переводил на {focus}?")
        if "account_profile" not in called:
            questions.append(f"Почему {focus} подозрителен?")
        if "find_network" not in called:
            questions.append(f"Покажи сеть вокруг {focus}")
        if "find_similar" not in called:
            questions.append(f"Есть ли счета с таким же поведением, как {focus}?")
    if "search_knowledge" not in called:
        questions.append("Какая статья к этому относится?")
    if session.case_accounts and "make_report" not in called:
        questions.append("Подготовь справку для полиции")
    return questions[:MAX_SUGGESTIONS]


def render_suggestions(questions: list[str], label: str, key: str) -> str | None:
    """Чипы-подсказки по три в ряд. Возвращает выбранный вопрос."""
    if not questions:
        return None
    st.caption(label)
    chosen = None
    for start in range(0, len(questions), 3):
        for column, text in zip(st.columns(3), questions[start:start + 3]):
            if column.button(text, key=f"{key}_{text}", help=text, width="stretch"):
                chosen = text
    return chosen


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
    if ready:
        label = "Что можно спросить дальше" if session.turns else "С чего начать"
        question = render_suggestions(follow_up_questions(session, data), label, f"quick_{data.name}")
    typed = st.chat_input("Спросите, например: куда ушли деньги клиентки?", disabled=not ready)
    question = typed or question
    if question:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=str(MARK)):
            status = st.status("Модель выбирает функцию…", expanded=True)
            preview = status.empty()

        def show_results(calls: list[dict]) -> None:
            # Результат функции посчитан кодом — его можно показать сразу. Текст модели появится
            # только после проверки чисел, непроверенные числа на экран не попадают.
            with preview.container():
                st.markdown(template_answer(calls))
                for call in calls:
                    render_details(call, data)
            status.update(label="Результат функции готов. Модель пишет пересказ, код проверяет числа…")

        try:
            session.ask(question, on_tools=show_results)
        except LLMError as error:
            st.error(f"Модель не ответила: {error}. Используйте режим кнопок.")
        st.rerun()


def render_lists(result: dict) -> None:
    """Результат сверки с перечнями АФМ: совпадения, контрагенты и источник перечней."""
    similar = result.get("similar")
    hits = result.get("hits", []) + (similar if isinstance(similar, list) and similar and "list" in similar[0] else [])
    for hit in hits:
        text = (f"{hit['list_title']}: «{hit['listed_name']}» · совпадение по: {hit['match']}"
                + (f" · включён {hit['added_to_list']}" if hit.get("added_to_list") else ""))
        if hit.get("strict", True):
            st.error(text)
        else:
            st.warning(text.replace("совпадение по: ", ""))
        if hit.get("is_demo"):
            st.caption("Это вымышленная тестовая запись для демо, в настоящих перечнях её нет.")
    # Обзор без счёта отдаёт список в listed и similar, проверка счёта — контрагентов из перечней.
    listed = result.get("listed")
    rows = (listed + (similar or [])) if isinstance(listed, list) else result.get("counterparties")
    if rows:
        st.dataframe(pd.DataFrame([{"Счёт": r["account"], "Владелец": r.get("owner") or "—",
                                    "Совпадение": "точное" if r["hits"][0].get("strict", True)
                                    else f"похоже ({r['hits'][0]['similarity']}%), проверить",
                                    "Перечень": r["hits"][0]["list_short"], "Входящих": r["incoming_count"],
                                    "Сумма входящих": r["incoming_total_text"], "Исходящих": r["outgoing_count"],
                                    "Последний перевод": r["last_time_text"]} for r in rows]),
                     hide_index=True, width="stretch")
        if any(h.get("is_demo") for r in rows for h in r["hits"]):
            st.caption("Тестовая запись демо вымышлена, в настоящих перечнях её нет.")
    st.caption(f"Перечни АФМ РК: {result['list_records_text']} записей, загружены {result['lists_downloaded_text']}, "
               f"источник {result['lists_source']}. Сверка по ИИН/БИН, названию или ФИО с датой рождения; "
               "похожее написание (опечатка, латиница) — только «проверить вручную».")


def render_queue(data: CaseData, analyst: str) -> None:
    """Очередь: счета по числу флажков и оценке модели, под ней профиль выбранного."""
    listed = data.flags.index.isin(set(data.list_hits) | set(data.list_similar))
    table = data.flags[listed | ~data.flags.index.isin(data.businesses | data.victims)].copy()
    table["risk"] = data.scores["risk_score"].reindex(table.index).fillna(0)
    table["listed"] = table.index.isin(set(data.list_hits))
    table["similar"] = table.index.isin(set(data.list_similar))
    table = table[table["listed"] | table["similar"] | (table["flag_count"] > 0)
                  | (table["risk"] >= data.high_risk * data.config["colors"]["medium_risk_share"])]
    # Лица из перечней АФМ — в начале очереди независимо от флажков, следом похожие на них.
    table = table.sort_values(["listed", "similar", "flag_count", "risk"],
                              ascending=False).head(data.config["app"]["queue_rows"])
    decisions = all_latest_decisions(FEEDBACK_PATH)
    view = pd.DataFrame({
        "Счёт": table.index, "Оценка": [mark(data.color(a)) for a in table.index], "Флажков": table["flag_count"].to_numpy(),
        "Риск модели": table["risk"].to_numpy(),
        "Главная причина": [(F.list_phrases(data, a) + F.flag_phrases(data, a) or data.model_reasons(a) or ["—"])[0]
                            for a in table.index],
        "Решение": [DECISION_TITLES.get(decisions.get(a), "") for a in table.index]})
    st.caption(f"Счетов с флажками или заметным риском: {len(view)} из {len(data.flags)}. Выберите строку.")
    selection = st.dataframe(styled(view), hide_index=True, width="stretch", height=360, on_select="rerun",
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
    st.set_page_config(page_title="BankRAG", page_icon=str(MARK), layout="wide")
    st.logo(str(MARK), size="large")
    config = load_config()
    with st.sidebar:
        st.header("BankRAG")
        analyst = st.text_input("Аналитик", value=st.session_state.get("analyst", "Аналитик"), key="analyst")
        dataset = st.radio("Набор данных", [KZ, IBM], format_func=DATASET_TITLES.get)
        model_name = st.selectbox("Модель", [config["llm"]["model"], config["llm"]["fallback_model"]])
        client, ready = model_status(config, model_name)
        st.caption(f"Модель {model_name}: {'готова' if ready else 'недоступна'}")
        st.caption(GUARDRAIL)
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
