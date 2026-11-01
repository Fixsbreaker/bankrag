"""Отрисовка сети счетов для интерфейса."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyvis.network import Network

MAX_COUNTERPARTIES = 25
COLOR_SELECTED = "#d62728"
COLOR_SIMILAR = "#ff7f0e"
COLOR_COUNTERPARTY = "#b0b0b0"

# Шаблон pyvis подключает Bootstrap с CDN даже в режиме in_line, а интерфейс работает без внешних запросов.
EXTERNAL_TAG = re.compile(r"<(?:link|script)\b[^>]*https?://[^>]*>(?:\s*</script>)?", re.IGNORECASE)


def network_accounts(account_id: str, similar: list[str], edges: pd.DataFrame,
                     max_counterparties: int = MAX_COUNTERPARTIES) -> tuple[list[str], pd.DataFrame]:
    """Счета для графа: выбранный, похожие и его крупнейшие прямые контрагенты; переводы между ними.

    Похожие счета редко переводят деньги друг другу, поэтому без контрагентов граф был бы пуст.
    """
    between = edges[~edges["is_self"]]
    touching = between[(between["from_account"] == account_id) | (between["to_account"] == account_id)]
    other = touching["from_account"].where(touching["from_account"] != account_id, touching["to_account"])
    largest = touching.groupby(other)["paid_usd"].sum().sort_values(ascending=False).head(max_counterparties)
    counterparties = [a for a in largest.index if a not in similar]
    accounts = [account_id, *similar, *counterparties]
    inside = between[between["from_account"].isin(accounts) & between["to_account"].isin(accounts)]
    grouped = inside.groupby(["from_account", "to_account"], as_index=False).agg(
        transfers=("paid_usd", "size"), amount_usd=("paid_usd", "sum"))
    return accounts, grouped


def remove_external_tags(html: str) -> str:
    """Убирает теги, загружающие что-либо из интернета."""
    return EXTERNAL_TAG.sub("", html)


def network_html(account_id: str, similar: list[str], accounts: list[str], grouped: pd.DataFrame) -> str:
    """HTML графа. Скрипты встроены в страницу, внешних запросов нет."""
    network = Network(height="520px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    for account in accounts:
        if account == account_id:
            color, size, group = COLOR_SELECTED, 28, "выбранный счёт"
        elif account in similar:
            color, size, group = COLOR_SIMILAR, 20, "похожий счёт"
        else:
            color, size, group = COLOR_COUNTERPARTY, 12, "контрагент"
        network.add_node(account, label=account, color=color, size=size, title=f"{account} ({group})", font={"size": 11})
    for row in grouped.itertuples(index=False):
        title = f"переводов: {row.transfers}, сумма около {row.amount_usd:,.0f} USD".replace(",", " ")
        network.add_edge(row.from_account, row.to_account, title=title, width=float(1 + np.log1p(row.transfers)))
    network.toggle_physics(True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "graph.html"
        network.write_html(str(path), notebook=False, open_browser=False)
        return remove_external_tags(path.read_text(encoding="utf-8"))
