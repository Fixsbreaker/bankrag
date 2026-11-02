"""Тесты генератора казахстанского набора на маленьких настройках."""
from __future__ import annotations

import pandas as pd

from bankrag.kz_scenarios import (
    CASH_FORMAT,
    DEMO_NETWORK,
    KIND_ATM,
    ROLE_MULE,
    ROLE_VICTIM,
    build_world_edges,
    build_world_features,
    generate_world,
)
from bankrag.labels import DROP, OTHER_LABELED

SETTINGS = {
    "start": "2026-11-01 00:00", "end": "2026-11-04 00:00", "features_hours": 48, "clients": 300, "employers": 3,
    "merchants": 10, "atms": 3, "salary_share": 0.3, "salary_tenge": [250000, 700000], "purchases_per_client": 2,
    "purchase_tenge": [2000, 40000], "family_share": 0.2, "family_tenge": [10000, 100000], "cash_share": 0.1,
    "cash_tenge": [20000, 100000], "networks": 3, "victims": [5, 8], "victim_tenge": [50000, 600000],
    "complaint_share": 0.4, "depth": [2, 4], "split_width": [2, 3], "forward_minutes": [5, 30],
    "forward_share": [0.92, 0.98], "split_noise": 0.03, "cash_out_minutes": [60, 180], "mule_with_history_share": 0.5,
    "tenge_per_usd": 500,
    "demo": {"victim_time": "2026-11-03 14:05", "victim_tenge": 500000, "depth": 3, "split_width": 2,
             "other_victims": 8, "other_complaints": 2},
}


def test_same_seed_gives_same_world_and_demo_case() -> None:
    first, second = generate_world(SETTINGS, 7), generate_world(SETTINGS, 7)
    pd.testing.assert_frame_equal(first.transfers, second.transfers)
    assert first.demo == second.demo
    assert not first.transfers.equals(generate_world(SETTINGS, 8).transfers)


def test_demo_case_matches_settings() -> None:
    world = generate_world(SETTINGS, 7)
    demo = world.demo
    assert [len(layer) for layer in demo["layers"]] == [1, 1, 2]
    first = world.transfers[(world.transfers["from_account"] == demo["victim"])
                            & (world.transfers["to_account"] == demo["first_drop"])]
    assert first["timestamp"].tolist() == [pd.Timestamp("2026-11-03 14:05")]
    assert first["amount_tenge"].tolist() == [500000.0]
    assert len(demo["other_senders"]) == 8
    demo_complaints = world.complaints[world.complaints["network_id"] == DEMO_NETWORK]
    assert len(demo_complaints) == 3 and demo["victim"] in set(demo_complaints["account_id"])


def test_chains_end_with_cash_and_stay_inside_period() -> None:
    world = generate_world(SETTINGS, 7)
    transfers = world.transfers
    assert transfers["timestamp"].min() >= pd.Timestamp(SETTINGS["start"])
    assert transfers["timestamp"].max() < pd.Timestamp(SETTINGS["end"])
    cash = transfers[(transfers["is_laundering"] == 1) & (transfers["payment_format"] == CASH_FORMAT)]
    atms = set(world.clients.loc[world.clients["kind"] == KIND_ATM, "account_id"])
    assert len(cash) and set(cash["to_account"]) <= atms


def test_labels_mark_mules_as_drops_and_victims_separately() -> None:
    world = generate_world(SETTINGS, 7)
    edges = build_world_edges(world.transfers, SETTINGS["tenge_per_usd"])
    assert edges["is_withdrawal"].any()
    features = build_world_features(edges, world.clients, SETTINGS["features_hours"], (0, 6)).set_index("account_id")
    assert not features.index.str.startswith("ATM-").any()
    mules = world.accounts.loc[world.accounts["role"] == ROLE_MULE, "account_id"]
    victims = world.accounts.loc[world.accounts["role"] == ROLE_VICTIM, "account_id"]
    present = [m for m in mules if m in features.index]
    assert present and (features.loc[present, "label"] == DROP).all()
    assert (features.loc[[v for v in victims if v in features.index], "label"] == OTHER_LABELED).all()
