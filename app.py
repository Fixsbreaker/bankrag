"""Чат по банковским документам и таблица измерений.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bankrag.answer import answer
from bankrag.search import Searcher

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "results" / "benchmark.csv"

st.set_page_config(page_title="BankRAG", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Загружаю индекс и модель эмбеддингов...")
def get_searcher(index_dir: str) -> Searcher:
    return Searcher.load(index_dir)


def render_answer(question: str, k: int, index_dir: str) -> None:
    try:
        searcher = get_searcher(index_dir)
    except FileNotFoundError:
        st.error("Индекс не найден. Постройте его: `python scripts/build_index.py`")
        return

    with st.spinner("Ищу фрагменты и формулирую ответ..."):
        hits = searcher.search(question, k)
        try:
            result = answer(question, hits)
        except Exception as e:  # нет ключа, нет сети, не запущен ollama
            st.error(f"Не удалось получить ответ от языковой модели: {e}")
            return

    if result["refused"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
        if result["uncited"]:
            st.caption("⚠️ Модель не указала источники, ответ нельзя проверить.")
        if result["unverified"]:
            st.caption("⚠️ Этих чисел нет в источниках: " + ", ".join(result["unverified"]) + ". Проверьте по тексту ниже.")

    by_id = {h["chunk_id"]: h for h in hits}
    cited = [by_id[c] for c in result["sources"]]
    st.subheader("Источники" if cited else "Найденные фрагменты (в ответе не использованы)")
    for h in cited or hits:
        with st.expander(f"{h['source']} · близость {h['score']:.2f}"):
            if h.get("page"):
                st.caption(f"Страница {h['page']}")
            if h.get("source_url"):
                page_anchor = f"#page={h['page']}" if h.get("page") else ""
                st.markdown(f"[Открыть документ банка]({h['source_url']}{page_anchor})")
                st.caption("Ответ относится к сохранённой версии документа; действующие условия проверяйте у банка.")
            st.write(h["text"])


def render_metrics() -> None:
    st.subheader("Сравнение вариантов индекса")
    if not BENCHMARK.exists():
        st.info("Файл results/benchmark.csv ещё не создан. Таблицу замеров готовит Сырым.")
        return
    table = pd.read_csv(BENCHMARK)
    st.dataframe(table, hide_index=True, width="stretch")
    measured = table.drop(columns=["index", "bytes_per_vector"], errors="ignore")
    if measured.isna().all().all():
        st.caption("Заглушка: память на вектор посчитана по формуле, скорость и полнота ещё не измерены.")


chat_tab, metrics_tab = st.tabs(["Чат", "Метрики"])

with st.sidebar:
    st.header("Настройки")
    k = st.slider("Сколько фрагментов давать модели", 1, 10, 5)
    index_dir = st.text_input("Папка с индексом", str(ROOT / "index"))

with chat_tab:
    st.title("BankRAG")
    st.caption("Вопрос о банковских документах: ответ строится только по найденным фрагментам, со ссылкой на источник.")
    with st.form("ask"):
        question = st.text_input("Ваш вопрос", placeholder="Какая комиссия за перевод клиенту другого банка?")
        submitted = st.form_submit_button("Спросить")
    if submitted and question.strip():
        render_answer(question.strip(), k, index_dir)

with metrics_tab:
    render_metrics()
