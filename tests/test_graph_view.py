"""Тесты подготовки сети для графа."""
from __future__ import annotations

import re

import pandas as pd

from bankrag.graph_view import network_accounts, network_html, remove_external_tags


def _edges() -> pd.DataFrame:
    return pd.DataFrame({
        "from_account": ["A", "X", "A", "S", "Q", "A"], "to_account": ["X", "A", "Y", "T", "R", "A"],
        "paid_usd": [10.0, 5.0, 100.0, 1.0, 1.0, 7.0], "is_self": [False, False, False, False, False, True]})


def test_network_has_selected_similar_and_counterparties() -> None:
    accounts, grouped = network_accounts("A", ["S"], _edges())
    assert accounts == ["A", "S", "Y", "X"]
    assert set(zip(grouped["from_account"], grouped["to_account"])) == {("A", "X"), ("X", "A"), ("A", "Y")}


def test_counterparties_are_limited_to_the_largest() -> None:
    accounts, _ = network_accounts("A", [], _edges(), max_counterparties=1)
    assert accounts == ["A", "Y"]


def test_html_makes_no_external_requests() -> None:
    accounts, grouped = network_accounts("A", ["S"], _edges())
    html = network_html("A", ["S"], accounts, grouped)
    assert "vis.Network" in html or "vis-network" in html
    assert not re.search(r"<(link|script)\b[^>]*https?://", html, re.IGNORECASE)


def test_remove_external_tags_keeps_inline_scripts() -> None:
    html = '<link href="https://x/a.css" rel="stylesheet"/><script src="https://x/a.js"></script><script>var a=1;</script>'
    assert remove_external_tags(html) == "<script>var a=1;</script>"
