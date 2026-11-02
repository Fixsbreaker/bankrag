"""Справка по делу в DOCX: движение денег, граф, флажки счетов и дословный текст нормы."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

from bankrag.data_loader import ROOT  # noqa: E402
from bankrag.flags import COLOR_HEX, COLOR_TITLES  # noqa: E402
from bankrag.kb_index import parse_document  # noqa: E402
from bankrag.tools import fmt  # noqa: E402
from bankrag.tools.data import CaseData  # noqa: E402

MAX_ROWS = 80
LAW_FILE = ROOT / "kb" / "law_232_1.md"
NOTE = ("Прототип BankRAG. Данные синтетические: IBM AML (NeurIPS 2023) и сценарии, сгенерированные командой. "
        "Справка собрана автоматически из данных и базы знаний, решение принимает аналитик. "
        "Правовую оценку даёт уполномоченный орган.")
CASH = "Наличные"


def case_transfers(data: CaseData, accounts: list[str]) -> pd.DataFrame:
    """Переводы дела: между счетами дела, поступления на них от пострадавших и снятие наличных."""
    edges = data.edges[~data.edges["is_self"]]
    chosen = set(accounts)
    inside = edges["from_account"].isin(chosen) & edges["to_account"].isin(chosen)
    cash = edges["from_account"].isin(chosen) & edges["is_withdrawal"]
    from_victims = edges["to_account"].isin(chosen) & edges["from_account"].isin(data.victims)
    return edges[inside | cash | from_victims].sort_values("timestamp")


def graph_png(data: CaseData, transfers: pd.DataFrame) -> bytes:
    """Картинка графа для документа: те же цвета, что в приложении."""
    graph = nx.DiGraph()
    for row in transfers.itertuples(index=False):
        target = CASH if row.is_withdrawal else row.to_account
        if graph.has_edge(row.from_account, target):
            graph[row.from_account][target]["amount"] += row.amount
        else:
            graph.add_edge(row.from_account, target, amount=row.amount)
    figure, axis = plt.subplots(figsize=(9, 6), dpi=150)
    if graph.number_of_nodes():
        layout = nx.spring_layout(graph, seed=42, k=1.2)
        colors = ["#ffffff" if node == CASH else COLOR_HEX[data.color(node)] for node in graph.nodes]
        nx.draw_networkx_nodes(graph, layout, node_color=colors, edgecolors="#333333", node_size=900, ax=axis)
        nx.draw_networkx_labels(graph, layout, font_size=7, ax=axis)
        nx.draw_networkx_edges(graph, layout, arrows=True, arrowsize=14, node_size=900, edge_color="#666666", ax=axis)
        labels = {(a, b): fmt.money(d["amount"], data.currency) for a, b, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=labels, font_size=6, ax=axis)
    axis.set_axis_off()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _table(document: Document, header: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for values in rows:
        for cell, text in zip(table.add_row().cells, values):
            cell.text = text
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8)


def build_case_report(data: CaseData, accounts: list[str], title: str, out_dir: Path | None = None) -> tuple[Path, int]:
    """Сохраняет справку и возвращает путь к файлу и число переводов в таблице."""
    from bankrag.tools.functions import flag_phrases

    out_dir = out_dir or ROOT / "data" / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    transfers = case_transfers(data, accounts)
    document = Document()
    document.add_heading(title, level=1)
    document.add_paragraph(f"Сформировано {datetime.now():%d.%m.%Y %H:%M}. Набор данных: {data.title}.")
    document.add_paragraph(NOTE).runs[0].italic = True

    document.add_heading("Движение денег", level=2)
    shown = transfers.head(MAX_ROWS)
    _table(document, ["Время", "Откуда", "Куда", "Сумма", "Формат"], [
        [fmt.when(r.timestamp), r.from_account, CASH + " (банкомат)" if r.is_withdrawal else r.to_account,
         fmt.money(r.amount, data.currency), str(r.payment_format)] for r in shown.itertuples(index=False)])
    if len(transfers) > MAX_ROWS:
        document.add_paragraph(f"Показаны первые {MAX_ROWS} переводов из {len(transfers)}.")

    document.add_heading("Граф связей", level=2)
    document.add_picture(io.BytesIO(graph_png(data, transfers)), width=Cm(16))
    document.add_paragraph("Красный — подозрительно (2 и больше флажков или высокий риск), оранжевый — стоит проверить, "
                           "серый — нейтрально, зелёный — чисто, синий — пострадавший, белый — снятие наличных.")

    document.add_heading("Признаки счетов", level=2)
    _table(document, ["Счёт", "Оценка", "Флажки", "Риск модели"], [
        [account, COLOR_TITLES[data.color(account)], "; ".join(flag_phrases(data, account)) or "нет",
         "нет оценки" if data.risk(account) is None else fmt.number(data.risk(account), 3)] for account in accounts])

    document.add_heading("Норма закона (дословно)", level=2)
    fragments = parse_document(LAW_FILE.read_text(encoding="utf-8"))
    for fragment in fragments:
        document.add_paragraph(fragment["text"])
    if fragments:
        document.add_paragraph(f"Источник: {fragments[0]['source']}. Правовую оценку даёт уполномоченный орган.")

    path = out_dir / f"case_{datetime.now():%Y%m%d_%H%M%S_%f}.docx"
    document.save(path)
    return path, len(transfers)
