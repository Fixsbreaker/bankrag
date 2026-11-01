"""Тесты графа и признаков на маленькой фикстуре."""
from __future__ import annotations

import pandas as pd
import pytest

from bankrag.features import compute_features, make_windows
from bankrag.graph import build_edges, estimate_usd_rates, subgraph_between


def _transactions(rows: list[tuple]) -> pd.DataFrame:
    """Строка фикстуры: время, откуда, куда, получено, валюта, оплачено, валюта, формат, метка."""
    df = pd.DataFrame(rows, columns=["timestamp", "from_account", "to_account", "amount_received",
                                     "receiving_currency", "amount_paid", "payment_currency",
                                     "payment_format", "is_laundering"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def edges() -> pd.DataFrame:
    usd = "US Dollar"
    transactions = _transactions([
        ("2022-09-01 10:00", "A", "M", 100.0, usd, 100.0, usd, "ACH", 1),
        ("2022-09-01 10:10", "B", "M", 100.0, usd, 100.0, usd, "Cash", 1),
        ("2022-09-01 10:20", "M", "C", 170.0, "Euro", 200.0, usd, "ACH", 1),
        ("2022-09-01 02:00", "C", "D", 50.0, "Euro", 50.0, "Euro", "Wire", 0),
        ("2022-09-01 12:00", "D", "D", 7.0, usd, 7.0, usd, "Reinvestment", 0),
    ])
    return build_edges(transactions, {usd: 1.0, "Euro": 0.85})


def test_usd_rate_comes_from_conversions() -> None:
    usd = "US Dollar"
    rates = estimate_usd_rates(_transactions([
        ("2022-09-01", "A", "B", 85.0, "Euro", 100.0, usd, "ACH", 0),
        ("2022-09-01", "A", "B", 8.5, "Euro", 10.0, usd, "ACH", 0),
    ]))
    assert rates == pytest.approx({usd: 1.0, "Euro": 0.85})


def test_missing_rate_is_an_error() -> None:
    with pytest.raises(ValueError):
        estimate_usd_rates(_transactions([("2022-09-01", "A", "B", 1.0, "Yen", 1.0, "Yen", "ACH", 0)]))


def test_mule_features(edges: pd.DataFrame) -> None:
    mule = compute_features(edges, known_drops=set()).loc["M"]
    assert mule["unique_senders"] == 2
    assert mule["unique_receivers"] == 1
    assert mule["in_count"] == 2
    assert mule["out_count"] == 1
    assert mule["transit_ratio"] == pytest.approx(1.0)
    assert mule["forward_minutes_median"] == pytest.approx(15.0)
    assert mule["forward_minutes_min"] == pytest.approx(10.0)
    assert mule["format_count"] == 2
    assert mule["currency_count"] == 1


def test_transit_ratio_converts_currencies(edges: pd.DataFrame) -> None:
    # C получил 170 евро (200 долларов), а отправил 50 евро.
    account = compute_features(edges, known_drops=set()).loc["C"]
    assert account["transit_ratio"] == pytest.approx(50 / 170)
    assert account["night_share"] == pytest.approx(0.5)


def test_self_transfers_are_not_counterparties(edges: pd.DataFrame) -> None:
    account = compute_features(edges, known_drops=set()).loc["D"]
    assert account["self_transfer_count"] == 1
    assert account["unique_receivers"] == 0
    assert account["unique_senders"] == 1


def test_features_ignore_transaction_labels(edges: pd.DataFrame) -> None:
    flipped = edges.assign(is_laundering=1 - edges["is_laundering"])
    pd.testing.assert_frame_equal(compute_features(edges, set()), compute_features(flipped, set()))


def test_known_drop_neighbors_counts_only_given_accounts(edges: pd.DataFrame) -> None:
    features = compute_features(edges, known_drops={"M"})
    assert features.loc["A", "known_drop_neighbors"] == 1
    assert features.loc["C", "known_drop_neighbors"] == 1
    assert features.loc["D", "known_drop_neighbors"] == 0
    assert features.loc["M", "known_drop_neighbors"] == 0


def test_windows_do_not_see_the_future() -> None:
    windows = make_windows(pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-06"), pd.Timestamp("2022-09-08 12:00"),
                           pd.Timestamp("2022-09-11"), pd.Timestamp("2022-09-18"), hours=48)
    train_end = pd.Timestamp("2022-09-06")
    names = [w.name for w in windows]
    assert names == ["train_1", "train_2", "val", "test", "test_with_tail"]
    for window in windows:
        assert window.known_until <= window.start or window.known_until <= train_end
        if window.period == "train":
            assert window.known_until == window.start
            assert window.end <= train_end
        else:
            assert window.known_until == train_end
            assert window.start >= train_end
    assert windows[2].end - windows[2].start == pd.Timedelta(hours=48)
    assert windows[3].end - windows[3].start == pd.Timedelta(hours=48)


def test_overlapping_train_windows_stay_inside_train_period() -> None:
    windows = make_windows(pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-06"), pd.Timestamp("2022-09-08 12:00"),
                           pd.Timestamp("2022-09-11"), pd.Timestamp("2022-09-18"), hours=48, train_step_hours=24)
    train = [w for w in windows if w.period == "train"]
    assert len(train) == 4
    assert all(w.start >= pd.Timestamp("2022-09-01") and w.end <= pd.Timestamp("2022-09-06") for w in train)
    assert all(w.end - w.start == pd.Timedelta(hours=48) for w in train)
    assert all(w.known_until == w.start for w in train)


def test_subgraph_keeps_only_transfers_inside_the_network(edges: pd.DataFrame) -> None:
    graph = subgraph_between(edges, ["A", "M", "C"])
    assert set(graph.nodes) == {"A", "M", "C"}
    assert graph.number_of_edges() == 2
    assert graph.has_edge("A", "M") and graph.has_edge("M", "C")
