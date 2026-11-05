"""Вклад признаков в оценку модели: доли, скрытые признаки IBM и вывод в профиле счёта."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bankrag.features import FEATURE_COLUMNS
from bankrag.model import explain_rows
from bankrag.tools import functions as F
from bankrag.tools.data import CaseData


def _contributions(**values: float) -> np.ndarray:
    row = np.zeros(len(FEATURE_COLUMNS))
    for name, value in values.items():
        row[FEATURE_COLUMNS.index(name)] = value
    return row[None, :]


def test_shares_split_explainable_and_hidden_features() -> None:
    contrib = _contributions(transit_ratio=3.0, unique_senders=1.0, currency_count=4.0, in_count=-2.0, format_count=-2.0)
    values = np.ones((1, len(FEATURE_COLUMNS)))
    row = explain_rows(contrib, values)[0]
    assert row["up_share"] == [3 / 8, 1 / 8] and row["up_hidden"] == 4 / 8
    assert row["up_text"][0].startswith("ушло дальше")
    assert row["down_share"] == [0.5] and row["down_hidden"] == 0.5


def test_no_contributions_give_empty_explanation() -> None:
    row = explain_rows(np.zeros((1, len(FEATURE_COLUMNS))), np.ones((1, len(FEATURE_COLUMNS))))[0]
    assert row["up_text"] == [] and row["up_hidden"] == 0.0


def test_profile_shows_factors_with_shares(data: CaseData) -> None:
    account = data.scores.index[0]
    scores = data.scores.assign(up_text=[["разных отправителей: 6"], []], up_share=[[0.6], []], up_hidden=[0.4, 0.0],
                                down_text=[["входящих переводов нет"], []], down_share=[[0.004], []],
                                down_hidden=[1.0, 0.0])
    with_factors = CaseData(data.name, data.currency, data.edges, scores, data.kinds, data.complaints, data.config,
                            data.high_risk)
    profile = F.account_profile(with_factors, account)
    assert profile["model_up"][0]["share_text"] == "60%"
    assert profile["model_down"] == []  # вклад меньше 1% не показывается
    assert "повысили оценку: разных отправителей: 6 (60%)" in profile["summary"]
    assert "особенности данных IBM" in profile["summary"]


def test_profile_without_explanations_still_works(data: CaseData) -> None:
    profile = F.account_profile(data, data.scores.index[0])
    assert profile["model_up"] == [] and "Вклад признаков" not in profile["summary"]
