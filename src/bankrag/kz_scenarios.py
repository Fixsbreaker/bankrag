"""Синтетические сети в стиле описанных в Казахстане схем: отдельный тестовый набор.

Много физлиц переводят деньги с карты на карту на один счёт, а дропы за минуты пересылают
их дальше по цепочке и выводят в наличные. Параметры заданы командой в config.yaml и не
откалиброваны на реальных данных. Модель на этих сетях не обучается.

    PYTHONPATH=src python -m bankrag.kz_scenarios
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bankrag.data_loader import ROOT, load_config
from bankrag.features import compute_features
from bankrag.graph import build_edges, estimate_usd_rates
from bankrag.labels import DROP, label_accounts, split_cutoffs

CURRENCY = "Tenge"
CARD_FORMAT = "Credit Card"
CASH_FORMAT = "Cash"
ACCOUNT_PREFIX = "KZ"
ROLE_SENDER = "sender"
ROLE_MULE = "mule"
ROLE_CASH_OUT = "cash_out"


def _transfer(when: pd.Timestamp, source: str, target: str, amount: float, payment_format: str) -> dict:
    return {"timestamp": when, "from_account": source, "to_account": target, "amount_received": amount,
            "receiving_currency": CURRENCY, "amount_paid": amount, "payment_currency": CURRENCY,
            "payment_format": payment_format, "is_laundering": 1}


def generate_network(network_id: int, settings: dict, start: pd.Timestamp, end: pd.Timestamp,
                     random: np.random.Generator) -> tuple[list[dict], dict[str, str]]:
    """Одна сеть: отправители → слои дропов → счёт вывода. Возвращает переводы и роли счетов."""
    account = lambda role, number: f"{ACCOUNT_PREFIX}{network_id:03d}_{role}{number:03d}"  # noqa: E731
    roles: dict[str, str] = {}
    transfers: list[dict] = []

    depth = int(random.integers(settings["depth"][0], settings["depth"][1] + 1))
    layers = [[account("M", layer * 10 + i) for i in range(int(random.integers(1, settings["max_layer_width"] + 1)))]
              for layer in range(depth)]
    for layer in layers:
        roles.update({mule: ROLE_MULE for mule in layer})
    cash_out = account("C", 0)
    roles[cash_out] = ROLE_CASH_OUT

    senders = int(random.integers(settings["senders"][0], settings["senders"][1] + 1))
    # Запас времени в конце окна, чтобы вся цепочка успела пройти внутри него.
    latest = end - pd.Timedelta(minutes=settings["forward_minutes"][1] * (depth + 2) + settings["inflow_hours"] * 60)
    first_time = start + (latest - start) * float(random.random())
    for number in range(senders):
        sender = account("S", number)
        roles[sender] = ROLE_SENDER
        when = first_time + pd.Timedelta(minutes=float(random.uniform(0, settings["inflow_hours"] * 60)))
        amount = float(np.round(random.uniform(*settings["amount_tenge"]), -2))
        target = layers[0][int(random.integers(len(layers[0])))]
        transfers.append(_transfer(when, sender, target, amount, CARD_FORMAT))

        # Дальше деньги идут по слоям: каждый дроп оставляет себе долю и быстро пересылает остаток.
        current, current_time, current_amount = target, when, amount
        for layer_index in range(1, depth + 1):
            current_time += pd.Timedelta(minutes=float(random.uniform(*settings["forward_minutes"])))
            current_amount = float(np.round(current_amount * random.uniform(*settings["forward_share"]), -2))
            if layer_index < depth:
                next_layer = layers[layer_index]
                following = next_layer[int(random.integers(len(next_layer)))]
                payment_format = CARD_FORMAT
            else:
                following, payment_format = cash_out, CASH_FORMAT
            transfers.append(_transfer(current_time, current, following, current_amount, payment_format))
            current = following
    # Случайный выбор мог обойти дропа в широком слое: счёт без переводов в сети не участвует.
    used = {t["from_account"] for t in transfers} | {t["to_account"] for t in transfers}
    return transfers, {account_id: role for account_id, role in roles.items() if account_id in used}


def generate_scenarios(settings: dict, start: pd.Timestamp, end: pd.Timestamp, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Все сети: таблица переводов и таблица счетов с номером сети и ролью."""
    random = np.random.default_rng(seed)
    transfers: list[dict] = []
    accounts: list[dict] = []
    for network_id in range(settings["networks"]):
        network_transfers, roles = generate_network(network_id, settings, start, end, random)
        transfers.extend(network_transfers)
        accounts.extend({"account_id": a, "network_id": network_id, "role": r} for a, r in roles.items())
    return pd.DataFrame(transfers), pd.DataFrame(accounts)


def build_scenario_features(background: pd.DataFrame, scenarios: pd.DataFrame, rates: dict[str, float],
                            known_drops: set[str], night_hours: tuple[int, int]) -> pd.DataFrame:
    """Встраивает сети в фон и считает признаки тем же кодом, что и для набора IBM."""
    columns = list(scenarios.columns)
    combined = pd.concat([background[columns].astype({"receiving_currency": "object", "payment_currency": "object",
                                                      "payment_format": "object"}), scenarios], ignore_index=True)
    combined = combined.sort_values("timestamp", kind="stable", ignore_index=True)
    edges = build_edges(combined, rates)
    features = compute_features(edges, known_drops, night_hours)
    features["label"] = label_accounts(combined).reindex(features.index)
    features["window"] = "kz"
    features["period"] = "kz"
    return features.reset_index()


def main() -> None:
    config = load_config()
    settings = config["kz_scenarios"]
    processed = ROOT / config["paths"]["processed_dir"]
    transactions = pd.read_parquet(processed / "transactions.parquet")
    train_end, _ = split_cutoffs(transactions["timestamp"], config["split"]["fractions"])
    train = transactions[transactions["timestamp"] < train_end]

    rates = estimate_usd_rates(train)
    # Тенге в наборе IBM нет, поэтому курс задан в настройках как допущение.
    rates[CURRENCY] = float(settings["tenge_per_usd"])

    end = pd.Timestamp(config["split"]["tail_start"])
    start = end - pd.Timedelta(hours=config["features"]["window_hours"])
    window = transactions[(transactions["timestamp"] >= start) & (transactions["timestamp"] < end)]
    # В фоне только обычные переводы: все помеченные счета здесь — сгенерированные.
    background = window[window["is_laundering"] == 0]

    scenarios, accounts = generate_scenarios(settings, start, end, config["seed"])
    train_labels = label_accounts(train)
    known = set(train_labels[train_labels == DROP].index)
    features = build_scenario_features(background, scenarios, rates, known, tuple(config["features"]["night_hours"]))

    features.to_parquet(processed / "kz_features.parquet", index=False)
    accounts.to_parquet(processed / "kz_accounts.parquet", index=False)
    drops = set(features.loc[features["label"] == DROP, "account_id"])
    mules = set(accounts.loc[accounts["role"] == ROLE_MULE, "account_id"])
    print(json.dumps({"networks": settings["networks"], "transfers": len(scenarios), "accounts": len(accounts),
                      "mules": len(mules), "drops_by_label": len(drops), "mules_equal_drops": mules == drops,
                      "background_transfers": len(background)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
