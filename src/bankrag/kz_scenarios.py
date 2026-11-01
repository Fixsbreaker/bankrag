"""Синтетические сети в стиле описанных в Казахстане схем: отдельный тестовый набор.

Много физлиц переводят деньги с карты на карту на один счёт, а дропы за минуты пересылают
их дальше по цепочке и выводят в наличные. Параметры заданы командой в config.yaml и не
откалиброваны на реальных данных. Модель на этих сетях не обучается.

    PYTHONPATH=src python -m bankrag.kz_scenarios
    PYTHONPATH=src python -m bankrag.kz_scenarios --seeds 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from bankrag.baseline_rule import fast_transit_rule
from bankrag.data_loader import ROOT, load_config
from bankrag.features import compute_features
from bankrag.graph import build_edges, estimate_usd_rates
from bankrag.labels import DROP, label_accounts, split_cutoffs
from bankrag.similarity import SimilarityIndex, VectorScaler, build_hnsw

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
    # Номер не выдаёт роль счёта: аналитик и модель не должны видеть её в названии.
    numbers = iter(random.permutation(settings["senders"][1] + settings["depth"][1] * settings["max_layer_width"] + 1))

    def account() -> str:
        return f"{ACCOUNT_PREFIX}{network_id:03d}_{next(numbers):03d}"

    roles: dict[str, str] = {}
    transfers: list[dict] = []

    depth = int(random.integers(settings["depth"][0], settings["depth"][1] + 1))
    layers = [[account() for _ in range(int(random.integers(1, settings["max_layer_width"] + 1)))]
              for _ in range(depth)]
    for layer in layers:
        roles.update({mule: ROLE_MULE for mule in layer})
    cash_out = account()
    roles[cash_out] = ROLE_CASH_OUT

    senders = int(random.integers(settings["senders"][0], settings["senders"][1] + 1))
    # Запас времени в конце окна, чтобы вся цепочка успела пройти внутри него.
    latest = end - pd.Timedelta(minutes=settings["forward_minutes"][1] * (depth + 2) + settings["inflow_hours"] * 60)
    first_time = start + (latest - start) * float(random.random())
    for _ in range(senders):
        sender = account()
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


def build_scenario_edges(background: pd.DataFrame, scenarios: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Встраивает сети в фон и приводит суммы к долларам."""
    columns = list(scenarios.columns)
    combined = pd.concat([background[columns].astype({"receiving_currency": "object", "payment_currency": "object",
                                                      "payment_format": "object"}), scenarios], ignore_index=True)
    combined = combined.sort_values("timestamp", kind="stable", ignore_index=True)
    return build_edges(combined, rates)


def build_scenario_features(edges: pd.DataFrame, known_drops: set[str], night_hours: tuple[int, int]) -> pd.DataFrame:
    """Считает признаки тем же кодом, что и для набора IBM."""
    features = compute_features(edges, known_drops, night_hours)
    features["label"] = label_accounts(edges).reindex(features.index)
    features["window"] = "kz"
    features["period"] = "kz"
    return features.reset_index()


def seed_spread(config: dict, background: pd.DataFrame, rates: dict[str, float], known: set[str],
                start: pd.Timestamp, end: pd.Timestamp, seeds: list[int]) -> dict:
    """Результат правила быстрого транзита на сетях с разными seed: одна генерация даёт шумную цифру."""
    rule = fast_transit_rule(config)
    night = tuple(config["features"]["night_hours"])
    rows = []
    for seed in seeds:
        scenarios, accounts = generate_scenarios(config["kz_scenarios"], start, end, seed)
        features = build_scenario_features(build_scenario_edges(background, scenarios, rates), known, night)
        flags = rule.flags(features)
        drops = features["label"] == DROP
        network_of = accounts.set_index("account_id")["network_id"]
        found = features.loc[flags & drops, "account_id"]
        rows.append({"seed": seed, "drops": int(drops.sum()), "drops_found": len(found),
                     "networks_found": int(network_of.reindex(found).nunique()),
                     "networks": int(accounts["network_id"].nunique()),
                     "background_alerts": int((flags & ~features["account_id"].isin(accounts["account_id"])).sum())})
        print(rows[-1], flush=True)
    shares = [row["drops_found"] / row["drops"] for row in rows]
    return {"seeds": rows, "share_min": min(shares), "share_median": float(np.median(shares)), "share_max": max(shares)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=None, help="только посчитать разброс правила на N других seed")
    args = parser.parse_args()
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

    train_labels = label_accounts(train)
    known = set(train_labels[train_labels == DROP].index)
    if args.seeds is not None:
        seeds = [settings["seed"] + 1 + i for i in range(args.seeds)]
        spread = seed_spread(config, background, rates, known, start, end, seeds)
        (ROOT / config["paths"]["reports_dir"] / "kz_seeds.json").write_text(
            json.dumps(spread, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    scenarios, accounts = generate_scenarios(settings, start, end, settings["seed"])
    edges = build_scenario_edges(background, scenarios, rates)
    features = build_scenario_features(edges, known, tuple(config["features"]["night_hours"]))

    features.to_parquet(processed / "kz_features.parquet", index=False)
    accounts.to_parquet(processed / "kz_accounts.parquet", index=False)
    edges[["timestamp", "from_account", "to_account", "paid_usd", "payment_format", "payment_currency",
           "is_self"]].astype({"payment_format": "object", "payment_currency": "object"}).to_parquet(
        processed / "edges_kz.parquet", index=False)

    # Индекс похожих счетов для интерфейса: масштаб обучен на периоде обучения IBM.
    models = ROOT / "models"
    scaler = VectorScaler.load(models / "similarity_scaler.npz")
    metric = json.loads((ROOT / config["paths"]["reports_dir"] / "similarity.json").read_text(encoding="utf-8"))["chosen_metric"]
    vectors = scaler.transform(features)
    search = config["similarity"]
    index = build_hnsw(vectors, metric, search["m"], search["ef_construction"], search["ef_search"])
    SimilarityIndex(index, features["account_id"].tolist(), vectors, metric).save(models, "similarity_kz")

    drops = set(features.loc[features["label"] == DROP, "account_id"])
    mules = set(accounts.loc[accounts["role"] == ROLE_MULE, "account_id"])
    print(json.dumps({"seed": settings["seed"], "networks": settings["networks"], "transfers": len(scenarios), "accounts": len(accounts),
                      "mules": len(mules), "drops_by_label": len(drops), "mules_equal_drops": mules == drops,
                      "background_transfers": len(background)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
