"""Тесты правила и модели на маленьких фикстурах."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bankrag.baseline_rule import Rule, evaluable, fast_transit_reasons, fast_transit_rule, tune_rule
from bankrag.features import FEATURE_COLUMNS
from bankrag.labels import CLEAN, DROP, OTHER_LABELED
from bankrag.model import best_f1_threshold, describe_reason, top_reasons


def _rule_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "unique_senders": [5, 5, 1, 5, 5],
        "transit_ratio": [0.95, 0.1, 0.95, 50.0, np.nan],
        "forward_minutes_median": [10.0, 10.0, 10.0, 10.0, np.nan],
        "label": [DROP, CLEAN, CLEAN, CLEAN, OTHER_LABELED],
    })


def test_rule_needs_all_conditions_and_treats_missing_as_false() -> None:
    flags = Rule(min_senders=3, min_transit=0.8, max_forward_minutes=60).flags(_rule_frame())
    assert flags.tolist() == [True, False, False, False, False]


def test_fast_transit_rule_uses_its_own_upper_bound() -> None:
    rule = fast_transit_rule({"rule": {"fast_transit": {"min_senders": 3, "min_transit": 0.8, "max_transit": 1.2,
                                                        "max_forward_minutes": 60}}})
    frame = pd.DataFrame({"unique_senders": [3, 3, 3, 2], "transit_ratio": [1.2, 1.21, 0.8, 1.0],
                          "forward_minutes_median": [60.0, 10.0, 61.0, 10.0]})
    assert rule.flags(frame).tolist() == [True, False, False, False]


def test_fast_transit_reasons_show_values_and_thresholds() -> None:
    rule = Rule(min_senders=3, min_transit=0.8, max_forward_minutes=60, max_transit=1.2)
    reasons = fast_transit_reasons(rule, 12, 0.95, 15)
    assert reasons == ["разных отправителей: 12 (порог 3)",
                       "ушло дальше 95% от полученной суммы (порог 80–120%)",
                       "деньги уходят дальше через 15 мин. (порог 60 мин.)"]


def test_evaluable_removes_other_labeled() -> None:
    assert OTHER_LABELED not in set(evaluable(_rule_frame())["label"])


def test_tune_rule_finds_separating_thresholds() -> None:
    grid = {"min_senders": [1, 3], "min_transit": [0.05, 0.8], "max_forward_minutes": [60]}
    rule, score = tune_rule(_rule_frame(), grid)
    assert score == pytest.approx(1.0)
    assert rule.min_senders == 3 and rule.min_transit == 0.8


def test_describe_reason_is_readable() -> None:
    assert describe_reason("unique_senders", 12) == "разных отправителей: 12"
    assert describe_reason("transit_ratio", 0.95) == "ушло дальше 95% от полученной суммы"
    assert describe_reason("transit_ratio", 283.0) == "отправлено в 283 раза больше, чем получено"
    assert describe_reason("transit_ratio", 5.0) == "отправлено в 5 раз больше, чем получено"
    assert describe_reason("forward_minutes_median", float("nan")) == "после входящих переводов исходящих не было"


def test_every_feature_has_a_phrase() -> None:
    for feature in FEATURE_COLUMNS:
        assert describe_reason(feature, 1.0)


def test_top_reasons_keeps_only_risk_raising_features() -> None:
    contributions = np.zeros((1, len(FEATURE_COLUMNS)))
    contributions[0, FEATURE_COLUMNS.index("unique_senders")] = 2.0
    contributions[0, FEATURE_COLUMNS.index("in_count")] = 1.0
    contributions[0, FEATURE_COLUMNS.index("night_share")] = -5.0
    values = np.full((1, len(FEATURE_COLUMNS)), 3.0)
    assert top_reasons(contributions, values) == [["разных отправителей: 3", "входящих переводов: 3"]]


def test_best_threshold_separates_classes() -> None:
    threshold = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < threshold <= 0.8


# LightGBM падает в одном процессе с FAISS и torch, поэтому модель проверяется в отдельном процессе.
MODEL_CHECK = """
import numpy as np, pandas as pd
from bankrag.features import FEATURE_COLUMNS
from bankrag.labels import CLEAN, DROP
from bankrag.model import score_table, train_model

def frame(seed):
    random = np.random.default_rng(seed)
    data = pd.DataFrame(random.random((300, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    data["label"] = np.where(data["unique_senders"] > 0.8, DROP, CLEAN)
    data["account_id"] = [f"a{i}" for i in range(len(data))]
    data["window"] = "w"
    data["period"] = "p"
    return data

params = {"objective": "binary", "metric": "average_precision", "verbose": -1, "seed": 1,
          "min_data_in_leaf": 5, "num_threads": 1}
model = train_model(frame(1), frame(2), params, num_boost_round=50, early_stopping_rounds=20)
scores = score_table(model, frame(3))
assert scores.loc[scores["label"] == DROP, "risk_score"].mean() > scores.loc[scores["label"] == CLEAN, "risk_score"].mean()
top = scores.sort_values("risk_score", ascending=False).iloc[0]
assert any("отправителей" in reason for reason in top["reasons"])
"""


def test_model_learns_and_explains_a_clear_signal() -> None:
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    done = subprocess.run([sys.executable, "-c", MODEL_CHECK], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
