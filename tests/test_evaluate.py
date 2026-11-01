"""Тесты расчёта метрик на маленьких фикстурах."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bankrag.evaluate import (
    bootstrap_pr_auc,
    classification_metrics,
    drop_pattern_types,
    extra_drops_at_same_alerts,
    queue_composition,
)
from bankrag.labels import CLEAN, DROP, OTHER_LABELED


def test_classification_metrics_counts() -> None:
    truth = np.array([True, True, False, False])
    flagged = np.array([True, False, True, False])
    result = classification_metrics(truth, flagged)
    assert result["alerts"] == 2
    assert result["drops_found"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_extra_drops_use_the_same_number_of_alerts_as_the_rule() -> None:
    truth = np.array([True, True, True, False, False])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    rule_flags = np.array([False, False, True, True, False])
    assert extra_drops_at_same_alerts(truth, scores, rule_flags) == {
        "alerts": 2, "rule_drops": 1, "model_drops": 2, "extra_drops": 1}


def test_queue_counts_clean_accounts_as_false_alarms() -> None:
    labels = pd.Series([DROP, CLEAN, OTHER_LABELED, CLEAN, DROP])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
    assert queue_composition(labels, scores, size=3) == {
        "size": 3, DROP: 1, OTHER_LABELED: 1, "false_alarms": 1}


def test_bootstrap_interval_contains_a_perfect_score() -> None:
    truth = np.array([True] * 5 + [False] * 95)
    scores = truth.astype(float)
    low, high = bootstrap_pr_auc(truth, scores, rounds=20, seed=1)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_pattern_types_keep_only_given_drops() -> None:
    patterns = pd.DataFrame({"from_account": ["A", "M", "X"], "to_account": ["M", "B", "Y"],
                             "pattern_type": ["STACK", "CYCLE", "CYCLE"]})
    assert drop_pattern_types(patterns, {"M"}) == {"CYCLE": {"M"}, "STACK": {"M"}}
