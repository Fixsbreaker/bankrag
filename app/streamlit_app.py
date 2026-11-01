"""Интерфейс аналитика: очередь, карточка счёта, похожие счета, справка, решение.

LightGBM здесь не загружается: оценки и причины читаются из data/processed/scores.parquet.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bankrag.faiss_safe import faiss  # noqa: E402, F401  # должен загружаться раньше scikit-learn

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from bankrag.data_loader import load_config  # noqa: E402
from bankrag.evaluate import render_markdown  # noqa: E402
from bankrag.feedback import CONFIRMED, REJECTED, all_latest_decisions, latest_decision, save_decision  # noqa: E402
from bankrag.graph_view import network_accounts, network_html  # noqa: E402
from bankrag.kb_index import KnowledgeBase, e5_embedder  # noqa: E402
from bankrag.labels import DROP, OTHER_LABELED  # noqa: E402
from bankrag.model import describe_reason  # noqa: E402
from bankrag.report import build_report  # noqa: E402
from bankrag.similarity import SimilarityIndex  # noqa: E402

WINDOW = "test"
FEEDBACK_PATH = ROOT / "data" / "feedback.sqlite"
LABEL_TITLES = {DROP: "дроп", OTHER_LABELED: "прочий помеченный", "clean": "чистый"}
DECISION_TITLES = {CONFIRMED: "подтвердил", REJECTED: "отклонил"}
CARD_FEATURES = ["unique_senders", "unique_receivers", "in_count", "out_count", "transit_ratio", "forward_minutes_median"]

@st.cache_data(show_spinner="Загружаю очередь…")
def load_queue() -> pd.DataFrame:
    """Счета тестового окна с оценкой риска и признаками, по убыванию риска."""
    processed = ROOT / "data" / "processed"
    scores = pd.read_parquet(processed / "scores.parquet")
    scores = scores[scores["window"] == WINDOW]
    features = pd.read_parquet(processed / "features.parquet")
    features = features[features["window"] == WINDOW].drop(columns=["label", "window", "period"])
    table = scores.merge(features, on="account_id", validate="one_to_one")
    return table.sort_values("risk_score", ascending=False, ignore_index=True)


@st.cache_data(show_spinner="Загружаю переводы…")
def load_edges() -> pd.DataFrame:
    """Переводы тестового окна для отрисовки сети."""
    return pd.read_parquet(ROOT / "data" / "processed" / "edges_test.parquet")


@st.cache_resource(show_spinner="Загружаю индекс похожих счетов…")
def load_similarity() -> SimilarityIndex:
    """Индекс FAISS HNSW тестового окна."""
    return SimilarityIndex.load(ROOT / "models", "similarity_test")


@st.cache_resource(show_spinner="Загружаю базу знаний и модель эмбеддингов…")
def load_kb() -> KnowledgeBase:
    """База знаний. Модель работает локально на CPU."""
    return KnowledgeBase.load(ROOT / "models", e5_embedder())


def render_card(row: pd.Series, threshold: float) -> None:
    """Карточка счёта: оценка риска, причины, основные числа."""
    st.subheader(f"Счёт {row['account_id']}")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Оценка риска", f"{row['risk_score']:.3f}".replace(".", ","))
        if row["risk_score"] >= threshold:
            st.warning("Выше порога тревоги")
        else:
            st.info("Ниже порога тревоги")
    with right:
        st.markdown("**Почему так оценено** (признаки, сильнее всего повысившие риск, за 48 часов)")
        reasons = list(row["reasons"])
        if reasons:
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.markdown("Признаков, повышающих риск, нет.")
    with st.expander("Основные числа за 48 часов", expanded=True):
        for feature in CARD_FEATURES:
            st.markdown(f"- {describe_reason(feature, row[feature])}")


def render_similar(account_id: str, queue: pd.DataFrame, show_labels: bool, k: int) -> None:
    """Похожие счета и граф переводов."""
    st.subheader("Похожие счета")
    if st.button("Найти похожие", key=f"similar_{account_id}"):
        st.session_state["similar_for"] = account_id
    if st.session_state.get("similar_for") != account_id:
        return
    found = load_similarity().find_similar(account_id, k=k).neighbors
    table = found.merge(queue[["account_id", "risk_score", "label"]], on="account_id", how="left")
    table = table.rename(columns={"account_id": "Счёт", "distance": "Близость", "risk_score": "Риск", "label": "Разметка"})
    if show_labels:
        table["Разметка"] = table["Разметка"].map(LABEL_TITLES)
    else:
        table = table.drop(columns="Разметка")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Близость": st.column_config.NumberColumn(format="%.3f"), "Риск": st.column_config.NumberColumn(format="%.3f")})
    similar = found["account_id"].tolist()
    accounts, grouped = network_accounts(account_id, similar, load_edges())
    st.caption("Красный — выбранный счёт, оранжевый — похожие по поведению, серый — его прямые контрагенты. "
               "Похожие счета не обязательно связаны переводами. Показаны переводы за 48 часов.")
    components.html(network_html(account_id, similar, accounts, grouped), height=540)


def render_report(row: pd.Series, config: dict) -> None:
    """Справка: дословные фрагменты базы знаний с источником, без генерации текста."""
    st.subheader("Справка")
    st.caption("Фрагменты базы знаний, близкие по тексту к признакам счёта. Текст выводится дословно. "
               "Это не определение типа схемы: на тесте такой подбор не лучше случайного, см. «Метрики».")
    account_id = row["account_id"]
    if st.button("Подготовить справку", key=f"report_{account_id}"):
        st.session_state["report_for"] = account_id
    if st.session_state.get("report_for") != account_id:
        return
    settings = config["report"]
    report = build_report(account_id, list(row["reasons"]), load_kb(), settings["top_k"], settings["min_similarity"])
    st.markdown("**Сработавшие признаки**")
    for reason in report.reasons:
        st.markdown(f"- {reason}")
    st.markdown("**Близкие описания схем** (пересказ статьи IBM авторами проекта, не цитата)")
    if report.no_typologies_message:
        st.write(report.no_typologies_message)
    for fragment in report.typologies:
        st.markdown(f"> {fragment['text']}")
        st.caption(f"Источник: {fragment['source']} · близость {fragment['score']:.2f}")
    st.markdown("**Нормы**")
    if report.no_norms_message:
        st.write(report.no_norms_message)
    for fragment in report.norms:
        st.markdown(f"> {fragment['text']}")
        st.caption(f"{fragment['title']}. Источник: {fragment['source']} · близость {fragment['score']:.2f}")


def render_decision(row: pd.Series) -> None:
    """Кнопки решения аналитика. Система сама ничего не блокирует."""
    st.subheader("Решение аналитика")
    account_id = row["account_id"]
    confirm, reject, _ = st.columns([1, 1, 4])
    if confirm.button("Подтвердил", key=f"confirm_{account_id}", type="primary"):
        save_decision(FEEDBACK_PATH, account_id, CONFIRMED, float(row["risk_score"]))
    if reject.button("Отклонил", key=f"reject_{account_id}"):
        save_decision(FEEDBACK_PATH, account_id, REJECTED, float(row["risk_score"]))
    decision = latest_decision(FEEDBACK_PATH, account_id)
    if decision:
        st.success(f"Сохранено локально: {DECISION_TITLES[decision]}")


def render_queue(queue: pd.DataFrame, threshold: float, config: dict) -> None:
    """Очередь, а под ней путь аналитика по выбранному счёту."""
    filters = st.columns([2, 2, 2])
    only_alerts = filters[0].checkbox("Только выше порога тревоги", value=True)
    search = filters[1].text_input("Поиск по счёту", placeholder="например, 3209_8000F4670")
    show_labels = filters[2].checkbox("Показать разметку набора (только для демо)", value=False)

    visible = queue
    if only_alerts:
        visible = visible[visible["risk_score"] >= threshold]
    if search.strip():
        visible = visible[visible["account_id"].str.contains(search.strip(), regex=False)]
    visible = visible.head(config["app"]["queue_rows"])

    decisions = all_latest_decisions(FEEDBACK_PATH)
    table = pd.DataFrame({
        "Счёт": visible["account_id"].astype(str),
        "Риск": visible["risk_score"],
        "Главная причина": [(list(r) or ["—"])[0] for r in visible["reasons"]],
        "Решение": [DECISION_TITLES.get(decisions.get(a), "") for a in visible["account_id"]],
    })
    if show_labels:
        table["Разметка"] = visible["label"].map(LABEL_TITLES).to_numpy()

    st.caption(f"Показано счетов: {len(table)} из {len(queue)}. Выберите строку, чтобы открыть карточку.")
    selection = st.dataframe(table, hide_index=True, width="stretch", height=320, on_select="rerun",
                             selection_mode="single-row", key="queue_table",
                             column_config={"Риск": st.column_config.NumberColumn(format="%.3f")})
    rows = selection.selection.rows
    if not rows:
        st.info("Выберите счёт в таблице.")
        return
    row = visible.iloc[rows[0]]
    st.divider()
    render_card(row, threshold)
    st.divider()
    render_similar(row["account_id"], queue, show_labels, config["similarity"]["k"])
    st.divider()
    render_report(row, config)
    st.divider()
    render_decision(row)


def render_metrics() -> None:
    """Вкладка метрик читает reports/metrics.json, числа здесь не считаются."""
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        st.warning("Файла reports/metrics.json нет: сначала запустите оценку.")
        return
    st.markdown(render_markdown(json.loads(path.read_text(encoding="utf-8"))))


def main() -> None:
    st.set_page_config(page_title="BankRAG", layout="wide")
    st.warning("Прототип. Данные синтетические (IBM AML, NeurIPS 2023). Решение принимает аналитик.")
    st.title("BankRAG: помощник аналитика антифрода")
    config = load_config()
    threshold = json.loads((ROOT / "reports" / "model.json").read_text(encoding="utf-8"))["threshold"]
    queue_tab, metrics_tab = st.tabs(["Очередь", "Метрики"])
    with queue_tab:
        render_queue(load_queue(), threshold, config)
    with metrics_tab:
        render_metrics()


if __name__ == "__main__":
    main()
