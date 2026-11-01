"""Разметка счетов (вариант A) и разбиение по времени.

Метка считается по любому набору транзакций, поэтому одна и та же функция
годится и для всего периода, и для отдельного периода разбиения.

    PYTHONPATH=src python -m bankrag.labels
"""
from __future__ import annotations

import json

import pandas as pd

from bankrag.data_loader import ROOT, load_config

DROP = "drop"
OTHER_LABELED = "other_labeled"
CLEAN = "clean"
PERIODS = ("train", "val", "test")


def label_accounts(transactions: pd.DataFrame) -> pd.Series:
    """Возвращает метку каждого счёта: drop, other_labeled или clean.

    drop — счёт и получал, и отправлял помеченные переводы (посредник в цепочке).
    other_labeled — остальные счета с помеченными переводами; в метриках не считаются.
    Помеченный перевод самому себе не делает счёт посредником.
    """
    accounts = pd.Index(pd.concat([transactions["from_account"], transactions["to_account"]]).unique())
    labeled = transactions[transactions["is_laundering"] == 1]
    between_accounts = labeled[labeled["from_account"] != labeled["to_account"]]
    senders = pd.Index(between_accounts["from_account"].unique())
    receivers = pd.Index(between_accounts["to_account"].unique())
    touched = pd.Index(pd.concat([labeled["from_account"], labeled["to_account"]]).unique())

    labels = pd.Series(CLEAN, index=accounts, name="label", dtype="object")
    labels.loc[touched] = OTHER_LABELED
    labels.loc[senders.intersection(receivers)] = DROP
    labels.index.name = "account_id"
    return labels


def split_cutoffs(timestamps: pd.Series, fractions: list[float]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Границы разбиения по числу транзакций, а не по календарю.

    В конце периода IBM почти нет обычных переводов, поэтому календарные доли
    оставили бы в тесте 123 перевода, 61% которых помечены.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("нужны три доли с суммой 1")
    ordered = timestamps.sort_values(ignore_index=True)
    train_end = ordered.iloc[int(len(ordered) * fractions[0])]
    val_end = ordered.iloc[int(len(ordered) * (fractions[0] + fractions[1]))]
    return train_end, val_end


def assign_period(timestamps: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.Series:
    """Номер периода каждой транзакции: train до train_end, val до val_end, затем test."""
    period = pd.Series("test", index=timestamps.index, dtype="object")
    period[timestamps < val_end] = "val"
    period[timestamps < train_end] = "train"
    return period


def main() -> None:
    config = load_config()
    processed = ROOT / config["paths"]["processed_dir"]
    transactions = pd.read_parquet(processed / "transactions.parquet", columns=[
        "timestamp", "from_account", "to_account", "is_laundering"])

    train_end, val_end = split_cutoffs(transactions["timestamp"], config["split"]["fractions"])
    periods = assign_period(transactions["timestamp"], train_end, val_end)
    tail_start = pd.Timestamp(config["split"]["tail_start"])

    def counts(subset: pd.DataFrame) -> dict:
        labels = label_accounts(subset)
        return {"transactions": len(subset), "laundering_transactions": int(subset["is_laundering"].sum()),
                "accounts": len(labels), **{k: int((labels == k).sum()) for k in (DROP, OTHER_LABELED, CLEAN)}}

    overall = label_accounts(transactions)
    overall.to_frame().reset_index().to_parquet(processed / "labels.parquet", index=False)

    drops_by_period = {}
    summary = {"variant": "A", "train_end": str(train_end), "val_end": str(val_end), "all": counts(transactions)}
    for period in PERIODS:
        subset = transactions[periods == period]
        summary[period] = counts(subset)
        labels = label_accounts(subset)
        drops_by_period[period] = set(labels[labels == DROP].index)
    summary["test_without_tail"] = counts(transactions[(periods == "test") & (transactions["timestamp"] < tail_start)])
    summary["test_drops_also_in_train"] = len(drops_by_period["test"] & drops_by_period["train"])
    summary["all_drops_not_drop_in_any_single_period"] = len(
        set(overall[overall == DROP].index) - set().union(*drops_by_period.values()))

    reports = ROOT / config["paths"]["reports_dir"]
    (reports / "labels_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
