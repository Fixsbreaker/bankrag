"""Тесты генератора казахстанских сценариев."""
from __future__ import annotations

import pandas as pd
import pytest

from bankrag.kz_scenarios import (
    CARD_FORMAT,
    CASH_FORMAT,
    CURRENCY,
    ROLE_CASH_OUT,
    ROLE_MULE,
    ROLE_SENDER,
    build_scenario_features,
    generate_scenarios,
)
from bankrag.labels import DROP, OTHER_LABELED

SETTINGS = {
    "networks": 4, "senders": [5, 10], "depth": [1, 3], "max_layer_width": 2, "inflow_hours": 6,
    "forward_minutes": [5, 30], "forward_share": [0.9, 0.98], "amount_tenge": [20000, 300000],
}
START = pd.Timestamp("2022-09-09")
END = pd.Timestamp("2022-09-11")


def _scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_scenarios(SETTINGS, START, END, seed=7)


def test_same_seed_gives_same_networks() -> None:
    pd.testing.assert_frame_equal(_scenarios()[0], _scenarios()[0])
    assert not _scenarios()[0].equals(generate_scenarios(SETTINGS, START, END, seed=8)[0])


def test_all_transfers_stay_inside_the_window() -> None:
    transfers, _ = _scenarios()
    assert transfers["timestamp"].min() >= START
    assert transfers["timestamp"].max() < END


def test_money_is_in_tenge_with_card_transfers_and_cash_out() -> None:
    transfers, accounts = _scenarios()
    assert set(transfers["payment_currency"]) == {CURRENCY}
    assert set(transfers["payment_format"]) == {CARD_FORMAT, CASH_FORMAT}
    cash_targets = set(transfers.loc[transfers["payment_format"] == CASH_FORMAT, "to_account"])
    assert cash_targets == set(accounts.loc[accounts["role"] == ROLE_CASH_OUT, "account_id"])


def test_mules_forward_less_than_they_received_and_later() -> None:
    transfers, accounts = _scenarios()
    for mule in accounts.loc[accounts["role"] == ROLE_MULE, "account_id"]:
        incoming = transfers[transfers["to_account"] == mule]
        outgoing = transfers[transfers["from_account"] == mule]
        assert len(incoming) and len(outgoing)
        assert outgoing["amount_paid"].sum() <= incoming["amount_received"].sum()
        assert outgoing["timestamp"].min() > incoming["timestamp"].min()


def test_labels_mark_mules_as_drops_and_chain_ends_separately() -> None:
    transfers, accounts = _scenarios()
    usd = "US Dollar"
    background = pd.DataFrame({
        "timestamp": [START], "from_account": ["BG1"], "to_account": ["BG2"], "amount_received": [5.0],
        "receiving_currency": pd.Categorical([usd]), "amount_paid": [5.0], "payment_currency": pd.Categorical([usd]),
        "payment_format": pd.Categorical(["ACH"]), "is_laundering": [0]})
    features = build_scenario_features(background, transfers, {usd: 1.0, CURRENCY: 500.0}, set(), (0, 6)).set_index("account_id")
    labels = features["label"]
    for row in accounts.itertuples(index=False):
        expected = DROP if row.role == ROLE_MULE else OTHER_LABELED
        assert labels[row.account_id] == expected, row
    assert labels["BG1"] == "clean"
    # Суммы в тенге переводятся по заданному курсу, транзит дропа близок к единице.
    mule = accounts.loc[accounts["role"] == ROLE_MULE, "account_id"].iloc[0]
    assert features.loc[mule, "transit_ratio"] == pytest.approx(0.94, abs=0.1)
    sender = accounts.loc[accounts["role"] == ROLE_SENDER, "account_id"].iloc[0]
    assert features.loc[sender, "in_count"] == 0
