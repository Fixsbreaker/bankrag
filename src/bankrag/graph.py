"""Граф переводов: вершины — счета, рёбра — переводы.

Полный граф из 5 млн рёбер хранится как таблица, это быстрее и экономнее объекта networkx.
Объект networkx строится только для небольшой сети, которую нужно показать на экране.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

BASE_CURRENCY = "US Dollar"


def estimate_usd_rates(transactions: pd.DataFrame) -> dict[str, float]:
    """Сколько единиц валюты дают за 1 доллар: медиана по переводам из долларов.

    Курсов в наборе нет, но у переводов с конвертацией известны обе суммы.
    Считать нужно только по обучающему периоду, чтобы не заглядывать в будущее.
    """
    payment = transactions["payment_currency"].astype(str)
    receiving = transactions["receiving_currency"].astype(str)
    from_usd = transactions[(payment == BASE_CURRENCY) & (receiving != BASE_CURRENCY)]
    ratio = from_usd["amount_received"] / from_usd["amount_paid"]
    rates = ratio.groupby(from_usd["receiving_currency"].astype(str)).median().to_dict()
    rates[BASE_CURRENCY] = 1.0
    missing = (set(payment.unique()) | set(receiving.unique())) - set(rates)
    if missing:
        raise ValueError("нет курса для валют: " + ", ".join(sorted(missing)))
    return rates


def build_edges(transactions: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Таблица рёбер: суммы приводятся к долларам, иначе их нельзя складывать между валютами."""
    edges = transactions.copy()
    edges["received_usd"] = edges["amount_received"] / edges["receiving_currency"].astype(str).map(rates).astype(float)
    edges["paid_usd"] = edges["amount_paid"] / edges["payment_currency"].astype(str).map(rates).astype(float)
    edges["is_self"] = edges["from_account"] == edges["to_account"]
    return edges


def subgraph_between(edges: pd.DataFrame, accounts: list[str]) -> nx.MultiDiGraph:
    """Граф переводов только между указанными счетами, для отрисовки сети."""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(accounts)
    inside = edges[edges["from_account"].isin(accounts) & edges["to_account"].isin(accounts) & ~edges["is_self"]]
    for row in inside.itertuples(index=False):
        graph.add_edge(row.from_account, row.to_account, timestamp=row.timestamp, amount_usd=row.paid_usd,
                       payment_format=str(row.payment_format), currency=str(row.payment_currency))
    return graph
