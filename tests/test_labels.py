"""Тесты разметки счетов и разбиения по времени."""
from __future__ import annotations

import pandas as pd
import pytest

from bankrag.labels import CLEAN, DROP, OTHER_LABELED, assign_period, label_accounts, split_cutoffs


def _transactions(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["from_account", "to_account", "is_laundering"])


def test_intermediary_is_drop_and_chain_ends_are_not() -> None:
    labels = label_accounts(_transactions([("A", "B", 1), ("B", "C", 1), ("D", "E", 0)]))
    assert labels["B"] == DROP
    assert labels["A"] == OTHER_LABELED
    assert labels["C"] == OTHER_LABELED
    assert labels["D"] == CLEAN
    assert labels["E"] == CLEAN


def test_clean_transfers_do_not_make_a_drop() -> None:
    labels = label_accounts(_transactions([("A", "B", 1), ("B", "C", 0)]))
    assert labels["B"] == OTHER_LABELED
    assert labels["C"] == CLEAN


def test_labeled_self_transfer_is_not_a_drop() -> None:
    labels = label_accounts(_transactions([("A", "A", 1)]))
    assert labels["A"] == OTHER_LABELED


def test_split_is_by_transaction_count_and_keeps_order() -> None:
    # Девять транзакций в первый день и одна спустя месяц: календарь дал бы другие границы.
    timestamps = pd.Series(pd.date_range("2022-09-01", periods=9, freq="h").tolist()
                           + [pd.Timestamp("2022-10-01")])
    train_end, val_end = split_cutoffs(timestamps, [0.6, 0.2, 0.2])
    periods = assign_period(timestamps, train_end, val_end)
    assert periods.value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}
    assert timestamps[periods == "train"].max() < timestamps[periods == "val"].min()
    assert timestamps[periods == "val"].max() < timestamps[periods == "test"].min()


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split_cutoffs(pd.Series(pd.date_range("2022-09-01", periods=3)), [0.5, 0.2, 0.2])
