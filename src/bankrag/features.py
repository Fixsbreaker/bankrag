"""Признаки счёта за окно фиксированной длины.

Окно одной длины нужно потому, что периоды разбиения разные: обучение длится 5,6 дня, а
валидация 2,1. Счётчики за разное время несравнимы, модель училась бы на другом масштабе.

    PYTHONPATH=src python -m bankrag.features
    PYTHONPATH=src python -m bankrag.features --sample 0.1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bankrag.data_loader import ROOT, load_config
from bankrag.graph import build_edges, estimate_usd_rates
from bankrag.labels import CLEAN, DROP, OTHER_LABELED, label_accounts, split_cutoffs

FEATURE_COLUMNS = [
    "unique_senders", "unique_receivers", "in_count", "out_count", "self_transfer_count",
    "transit_ratio", "forward_minutes_median", "forward_minutes_min", "night_share",
    "format_count", "currency_count", "amount_log_mean", "amount_log_std",
    "neighbor_degree_mean", "neighbor_degree_max", "known_drop_neighbors",
]


@dataclass(frozen=True)
class Window:
    """Окно признаков. Известными считаются дропы из переводов до known_until."""

    name: str
    period: str
    start: pd.Timestamp
    end: pd.Timestamp
    known_until: pd.Timestamp


def make_windows(first: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp,
                 tail_start: pd.Timestamp, last: pd.Timestamp, hours: int,
                 train_step_hours: int | None = None) -> list[Window]:
    """Окна обучения идут назад от конца периода; валидация и тест берут одно окно той же длины.

    Шаг короче окна даёт перекрывающиеся окна обучения: один счёт встречается в них
    несколько раз, зато цепочки, разрезанные одной границей, целиком попадают в другое окно.

    Для окон обучения известны только дропы из более раннего времени, для валидации и
    теста — все дропы обучающего периода. Так признак ведёт себя одинаково во всех окнах.
    """
    length = pd.Timedelta(hours=hours)
    step = pd.Timedelta(hours=train_step_hours or hours)
    windows = []
    end = train_end
    number = 0
    while end - length >= first:
        number += 1
        windows.append(Window(f"train_{number}", "train", end - length, end, end - length))
        end -= step
    if not windows:
        raise ValueError("окно длиннее обучающего периода")
    windows.append(Window("val", "val", val_end - length, val_end, train_end))
    windows.append(Window("test", "test", tail_start - length, tail_start, train_end))
    # Вариант с хвостом длиннее остальных окон: он нужен только для сравнения.
    windows.append(Window("test_with_tail", "test_with_tail", val_end, last + pd.Timedelta(minutes=1), train_end))
    return windows


def _forward_minutes(edges: pd.DataFrame) -> pd.DataFrame:
    """Минуты от каждого входящего перевода до следующего исходящего с того же счёта."""
    incoming = edges[["to_account", "timestamp"]].rename(columns={"to_account": "account_id"})
    outgoing = edges[["from_account", "timestamp"]].rename(
        columns={"from_account": "account_id", "timestamp": "out_time"})
    paired = pd.merge_asof(incoming.sort_values("timestamp"), outgoing.sort_values("out_time"),
                           left_on="timestamp", right_on="out_time", by="account_id", direction="forward")
    minutes = (paired["out_time"] - paired["timestamp"]).dt.total_seconds() / 60
    return minutes.groupby(paired["account_id"]).agg(forward_minutes_median="median", forward_minutes_min="min")


def compute_features(edges: pd.DataFrame, known_drops: set[str], night_hours: tuple[int, int] = (0, 6)) -> pd.DataFrame:
    """Признаки каждого счёта по рёбрам одного окна. Метки переводов здесь не используются."""
    between = edges[~edges["is_self"]]
    accounts = pd.Index(pd.concat([edges["from_account"], edges["to_account"]]).unique(), name="account_id")
    features = pd.DataFrame(index=accounts)

    incoming = between.groupby("to_account", observed=True)
    outgoing = between.groupby("from_account", observed=True)
    features["unique_senders"] = incoming["from_account"].nunique()
    features["unique_receivers"] = outgoing["to_account"].nunique()
    features["in_count"] = incoming.size()
    features["out_count"] = outgoing.size()
    features["self_transfer_count"] = edges[edges["is_self"]].groupby("from_account").size()
    for column in ["unique_senders", "unique_receivers", "in_count", "out_count", "self_transfer_count"]:
        features[column] = features[column].fillna(0)

    # Без входящих доля транзита не определена; пропуск LightGBM обрабатывает сам.
    received = incoming["received_usd"].sum()
    paid = outgoing["paid_usd"].sum()
    features["transit_ratio"] = paid.reindex(accounts).fillna(0) / received.reindex(accounts)

    features = features.join(_forward_minutes(between))

    # Длинная таблица: одна строка на участие счёта в переводе, с обеих сторон.
    sides = pd.concat([
        pd.DataFrame({"account_id": between["from_account"], "timestamp": between["timestamp"],
                      "amount_usd": between["paid_usd"], "payment_format": between["payment_format"].astype(str),
                      "currency": between["payment_currency"].astype(str)}),
        pd.DataFrame({"account_id": between["to_account"], "timestamp": between["timestamp"],
                      "amount_usd": between["received_usd"], "payment_format": between["payment_format"].astype(str),
                      "currency": between["receiving_currency"].astype(str)}),
    ], ignore_index=True)
    night_start, night_end = night_hours
    sides["is_night"] = sides["timestamp"].dt.hour.between(night_start, night_end - 1)
    sides["amount_log"] = np.log1p(sides["amount_usd"])
    grouped = sides.groupby("account_id")
    features["night_share"] = grouped["is_night"].mean()
    features["format_count"] = grouped["payment_format"].nunique()
    features["currency_count"] = grouped["currency"].nunique()
    features["amount_log_mean"] = grouped["amount_log"].mean()
    features["amount_log_std"] = grouped["amount_log"].std().fillna(0)

    # Соседи: уникальные контрагенты в обе стороны.
    pairs = pd.concat([
        between[["from_account", "to_account"]].set_axis(["account_id", "neighbor"], axis=1),
        between[["to_account", "from_account"]].set_axis(["account_id", "neighbor"], axis=1),
    ], ignore_index=True).drop_duplicates()
    degree = pairs.groupby("account_id").size().rename("neighbor_degree")
    pairs = pairs.join(degree, on="neighbor")
    by_account = pairs.groupby("account_id")["neighbor_degree"]
    features["neighbor_degree_mean"] = by_account.mean()
    features["neighbor_degree_max"] = by_account.max()

    known = pairs[pairs["neighbor"].isin(known_drops)]
    features["known_drop_neighbors"] = known.groupby("account_id").size()
    features["known_drop_neighbors"] = features["known_drop_neighbors"].fillna(0)
    return features[FEATURE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=float, default=None, metavar="SHARE",
                        help="считать по случайной доле счетов (для быстрой проверки)")
    args = parser.parse_args()

    config = load_config()
    processed = ROOT / config["paths"]["processed_dir"]
    transactions = pd.read_parquet(processed / "transactions.parquet")
    train_end, val_end = split_cutoffs(transactions["timestamp"], config["split"]["fractions"])
    rates = estimate_usd_rates(transactions[transactions["timestamp"] < train_end])
    edges = build_edges(transactions, rates)

    windows = make_windows(transactions["timestamp"].min(), train_end, val_end,
                           pd.Timestamp(config["split"]["tail_start"]), transactions["timestamp"].max(),
                           config["features"]["window_hours"], config["features"]["train_step_hours"])
    night = tuple(config["features"]["night_hours"])

    frames = []
    summary = {"window_hours": config["features"]["window_hours"], "sample": args.sample, "usd_rates": rates, "windows": {}}
    for window in windows:
        current = edges[(edges["timestamp"] >= window.start) & (edges["timestamp"] < window.end)]
        before = edges[edges["timestamp"] < window.known_until]
        labels_before = label_accounts(before) if len(before) else pd.Series(dtype="object")
        known = set(labels_before[labels_before == DROP].index)
        features = compute_features(current, known, night)
        features["label"] = label_accounts(current).reindex(features.index)
        features["window"] = window.name
        features["period"] = window.period
        if args.sample is not None:
            features = features.sample(frac=args.sample, random_state=config["seed"])
        frames.append(features.reset_index())
        summary["windows"][window.name] = {
            "start": str(window.start), "end": str(window.end), "transactions": len(current),
            "accounts": len(features), "known_drops": len(known),
            **{k: int((features["label"] == k).sum()) for k in (DROP, OTHER_LABELED, CLEAN)},
        }
        print(window.name, summary["windows"][window.name], flush=True)

    table = pd.concat(frames, ignore_index=True)
    # Метка важнее id: выборку нельзя случайно принять за полный расчёт.
    name = "features_sample.parquet" if args.sample is not None else "features.parquet"
    table.to_parquet(processed / name, index=False)
    if args.sample is None:
        (ROOT / config["paths"]["reports_dir"] / "features_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
