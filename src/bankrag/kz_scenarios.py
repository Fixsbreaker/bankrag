"""Казахстанский набор: клиенты банка, обычная жизнь и сети дропов с жалобами пострадавших.

Весь набор сгенерирован командой: фон (зарплаты, покупки, переводы родственникам, снятие
наличных) и сети, где пострадавшие переводят деньги первому дропу, а дропы за минуты
пересылают их дальше, дробят на несколько счетов и снимают в банкомате. Параметры заданы
в config.yaml и не откалиброваны на реальных данных. Модель на этом наборе не обучается.

Сеть 0 — главный пример демо с жалобой клиентки, её счета пишутся в reports/demo_case.json.

    PYTHONPATH=src python -m bankrag.kz_scenarios
    PYTHONPATH=src python -m bankrag.kz_scenarios --seeds 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bankrag.baseline_rule import fast_transit_rule
from bankrag.data_loader import ROOT, load_config
from bankrag.features import compute_features
from bankrag.graph import build_edges
from bankrag.labels import DROP, label_accounts
from bankrag.similarity import SimilarityIndex, VectorScaler, build_hnsw

CURRENCY = "Tenge"
CARD_FORMAT = "Credit Card"
SALARY_FORMAT = "ACH"
CASH_FORMAT = "Cash"
KIND_CLIENT = "client"
KIND_EMPLOYER = "employer"
KIND_MERCHANT = "merchant"
KIND_ATM = "atm"
ROLE_VICTIM = "victim"
ROLE_MULE = "mule"
DEMO_NETWORK = 0
COMPLAINT_TEXTS = [
    "Позвонили из «службы безопасности банка» и попросили перевести деньги на «безопасный счёт».",
    "Продавец на сайте объявлений попросил предоплату и перестал отвечать.",
    "Предложили вложить деньги под высокий процент, после перевода связь пропала.",
    "Написали от имени родственника и попросили срочно перевести деньги.",
]
TRANSFER_COLUMNS = ["timestamp", "from_account", "to_account", "amount_tenge", "payment_format", "is_laundering"]


@dataclass
class World:
    """Сгенерированный набор: переводы, клиенты, участники сетей, жалобы и главный пример."""

    transfers: pd.DataFrame
    clients: pd.DataFrame
    accounts: pd.DataFrame
    complaints: pd.DataFrame
    demo: dict


class _AccountNumbers:
    """Номера вида KZ-1234 в случайном порядке: номер не выдаёт роль счёта."""

    def __init__(self, random: np.random.Generator):
        self._numbers = iter(random.permutation(np.arange(1000, 10000)))

    def next(self) -> str:
        return f"KZ-{next(self._numbers)}"


def _daytime(random: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """Случайный момент с 8:00 до 22:00 в один из дней периода."""
    days = max(int((end - start) / pd.Timedelta(days=1)), 1)
    day = start.normalize() + pd.Timedelta(days=int(random.integers(days)))
    moment = day + pd.Timedelta(minutes=int(random.integers(8 * 60, 22 * 60)))
    return min(max(moment, start), end - pd.Timedelta(minutes=1))


def _transfer(when: pd.Timestamp, source: str, target: str, amount: float, payment_format: str,
              laundering: int = 0) -> dict:
    # Время с точностью до минуты, как в выписке.
    return {"timestamp": when.floor("min"), "from_account": source, "to_account": target, "amount_tenge": float(amount),
            "payment_format": payment_format, "is_laundering": laundering}


def _round(amount: float, digits: int = -3) -> float:
    return float(max(np.round(amount, digits), 10 ** -digits))


def generate_background(settings: dict, start: pd.Timestamp, end: pd.Timestamp, random: np.random.Generator,
                        numbers: _AccountNumbers) -> tuple[list[dict], pd.DataFrame]:
    """Обычная жизнь клиентов: зарплата, покупки, переводы родственникам, снятие наличных."""
    clients = [numbers.next() for _ in range(settings["clients"])]
    employers = [numbers.next() for _ in range(settings["employers"])]
    merchants = [numbers.next() for _ in range(settings["merchants"])]
    atms = [f"ATM-{i + 1:02d}" for i in range(settings["atms"])]
    transfers: list[dict] = []

    salary_day = start.normalize() + pd.Timedelta(days=1)
    for client in random.choice(clients, size=int(settings["salary_share"] * len(clients)), replace=False):
        when = salary_day + pd.Timedelta(minutes=int(random.integers(9 * 60, 12 * 60)))
        transfers.append(_transfer(when, str(random.choice(employers)), str(client),
                                   _round(random.uniform(*settings["salary_tenge"])), SALARY_FORMAT))
    for client in clients:
        for _ in range(int(random.poisson(settings["purchases_per_client"]))):
            transfers.append(_transfer(_daytime(random, start, end), client, str(random.choice(merchants)),
                                       _round(random.uniform(*settings["purchase_tenge"]), -2), CARD_FORMAT))
    for client in random.choice(clients, size=int(settings["family_share"] * len(clients)), replace=False):
        relative = str(random.choice(clients))
        if relative != client:
            transfers.append(_transfer(_daytime(random, start, end), str(client), relative,
                                       _round(random.uniform(*settings["family_tenge"])), CARD_FORMAT))
    for client in random.choice(clients, size=int(settings["cash_share"] * len(clients)), replace=False):
        transfers.append(_transfer(_daytime(random, start, end), str(client), str(random.choice(atms)),
                                   _round(random.uniform(*settings["cash_tenge"])), CASH_FORMAT))

    kinds = ([(c, KIND_CLIENT) for c in clients] + [(e, KIND_EMPLOYER) for e in employers]
             + [(m, KIND_MERCHANT) for m in merchants] + [(a, KIND_ATM) for a in atms])
    return transfers, pd.DataFrame(kinds, columns=["account_id", "kind"])


def _layers(depth: int, split_layer: int, width: int, take_mule) -> list[list[str]]:
    """Слои дропов: до слоя дробления по одному счёту, начиная с него — по width."""
    return [[take_mule() for _ in range(width if index >= split_layer else 1)] for index in range(depth)]


def send_through_chain(layers: list[list[str]], when: pd.Timestamp, amount: float, settings: dict,
                       atms: list[str], random: np.random.Generator) -> list[dict]:
    """Деньги одного пострадавшего проходят слои дропов и снимаются в банкомате."""
    transfers = []
    holders = [(0, when, amount)]
    for layer in range(1, len(layers)):
        following = []
        for position, received_at, received in holders:
            sent_at = received_at + pd.Timedelta(minutes=float(random.uniform(*settings["forward_minutes"])))
            total = received * float(random.uniform(*settings["forward_share"]))
            source = layers[layer - 1][position]
            if len(layers[layer]) > len(layers[layer - 1]):
                # Дробление: примерно равные части на все счета следующего слоя.
                for part_index, target in enumerate(layers[layer]):
                    noise = float(random.uniform(-settings["split_noise"], settings["split_noise"]))
                    part = _round(total / len(layers[layer]) * (1 + noise))
                    part_at = sent_at + pd.Timedelta(minutes=part_index * int(random.integers(1, 4)))
                    transfers.append(_transfer(part_at, source, target, part, CARD_FORMAT, 1))
                    following.append((part_index, part_at, part))
            else:
                part = _round(total)
                transfers.append(_transfer(sent_at, source, layers[layer][position], part, CARD_FORMAT, 1))
                following.append((position, sent_at, part))
        holders = following
    for position, received_at, received in holders:
        cash_at = received_at + pd.Timedelta(minutes=float(random.uniform(*settings["cash_out_minutes"])))
        transfers.append(_transfer(cash_at, layers[-1][position], str(random.choice(atms)),
                                   _round(received * float(random.uniform(0.97, 1.0))), CASH_FORMAT, 1))
    return transfers


def generate_world(settings: dict, seed: int) -> World:
    """Весь набор с одним seed: один и тот же seed даёт один и тот же демо-кейс."""
    random = np.random.default_rng(seed)
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    numbers = _AccountNumbers(random)
    transfers, clients = generate_background(settings, start, end, random, numbers)
    atms = clients.loc[clients["kind"] == KIND_ATM, "account_id"].tolist()
    free_clients = list(random.permutation(clients.loc[clients["kind"] == KIND_CLIENT, "account_id"].to_numpy()))
    new_clients: list[str] = []

    def take_client() -> str:
        return str(free_clients.pop())

    def take_mule() -> str:
        if random.random() < settings["mule_with_history_share"]:
            return take_client()
        account = numbers.next()
        new_clients.append(account)
        return account

    # Цепочка должна успеть закончиться внутри периода.
    latest = end - pd.Timedelta(hours=6)
    features_start = end - pd.Timedelta(hours=settings["features_hours"])
    demo = settings["demo"]
    roster, complaints = [], []
    demo_case: dict = {}
    for network in range(settings["networks"] + 1):
        if network == DEMO_NETWORK:
            depth, width, split_layer = demo["depth"], demo["split_width"], demo["depth"] - 1
            others = demo["other_victims"]
        else:
            depth = int(random.integers(settings["depth"][0], settings["depth"][1] + 1))
            width = int(random.integers(settings["split_width"][0], settings["split_width"][1] + 1))
            split_layer = int(random.integers(1, depth))
            others = int(random.integers(settings["victims"][0], settings["victims"][1] + 1))
        layers = _layers(depth, split_layer, width, take_mule)
        roster += [{"account_id": a, "network_id": network, "role": ROLE_MULE} for layer in layers for a in layer]

        victims = []
        for index in range(others + (1 if network == DEMO_NETWORK else 0)):
            victim = take_client()
            if network == DEMO_NETWORK and index == 0:
                when, amount, complained = pd.Timestamp(demo["victim_time"]), float(demo["victim_tenge"]), True
                text = COMPLAINT_TEXTS[0]
            else:
                # Сети демо собирают деньги три дня, остальные — в окне признаков.
                window_start = start if network == DEMO_NETWORK else features_start
                when = _daytime(random, window_start, latest)
                amount = _round(random.uniform(*settings["victim_tenge"]))
                limit = demo["other_complaints"] if network == DEMO_NETWORK else None
                complained = (index <= limit) if limit is not None else bool(random.random() < settings["complaint_share"])
                text = COMPLAINT_TEXTS[int(random.integers(len(COMPLAINT_TEXTS)))]
            transfers.append(_transfer(when, victim, layers[0][0], amount, CARD_FORMAT, 1))
            transfers += send_through_chain(layers, when, amount, settings, atms, random)
            roster.append({"account_id": victim, "network_id": network, "role": ROLE_VICTIM})
            victims.append(victim)
            if complained:
                filed = min(when + pd.Timedelta(minutes=int(random.integers(120, 480))), end - pd.Timedelta(minutes=1))
                complaints.append({"account_id": victim, "target_account": layers[0][0], "transfer_time": when,
                                   "filed_at": filed, "amount_tenge": amount, "text": text, "network_id": network})
        if network == DEMO_NETWORK:
            demo_case = {"network_id": network, "victim": victims[0], "victim_time": str(pd.Timestamp(demo["victim_time"])),
                         "victim_tenge": float(demo["victim_tenge"]), "first_drop": layers[0][0], "layers": layers,
                         "other_senders": victims[1:]}

    clients = pd.concat([clients, pd.DataFrame({"account_id": new_clients, "kind": KIND_CLIENT})], ignore_index=True)
    table = pd.DataFrame(transfers, columns=TRANSFER_COLUMNS).sort_values("timestamp", kind="stable", ignore_index=True)
    return World(table, clients, pd.DataFrame(roster), pd.DataFrame(complaints), demo_case)


def build_world_edges(transfers: pd.DataFrame, tenge_per_usd: float) -> pd.DataFrame:
    """Рёбра в том же виде, что у IBM: суммы приведены к долларам по заданному курсу."""
    frame = pd.DataFrame({
        "timestamp": transfers["timestamp"], "from_account": transfers["from_account"],
        "to_account": transfers["to_account"], "amount_received": transfers["amount_tenge"],
        "receiving_currency": CURRENCY, "amount_paid": transfers["amount_tenge"], "payment_currency": CURRENCY,
        "payment_format": transfers["payment_format"], "is_laundering": transfers["is_laundering"]})
    edges = build_edges(frame, {CURRENCY: float(tenge_per_usd)})
    edges["amount"] = transfers["amount_tenge"].to_numpy()
    edges["is_withdrawal"] = edges["to_account"].str.startswith("ATM-") & (edges["payment_format"] == CASH_FORMAT)
    return edges


def build_world_features(edges: pd.DataFrame, clients: pd.DataFrame, features_hours: int,
                         night_hours: tuple[int, int]) -> pd.DataFrame:
    """Признаки тем же кодом, что для IBM, за последние features_hours. Банкоматы не оцениваются."""
    end = edges["timestamp"].max().ceil("D")
    window = edges[edges["timestamp"] >= end - pd.Timedelta(hours=features_hours)]
    features = compute_features(window, set(), night_hours)
    features["label"] = label_accounts(edges).reindex(features.index)
    atms = set(clients.loc[clients["kind"] == KIND_ATM, "account_id"])
    features = features[~features.index.isin(atms)]
    features["window"] = "kz"
    features["period"] = "kz"
    return features.reset_index()


def seed_spread(config: dict, seeds: list[int]) -> dict:
    """Результат правила быстрого транзита на наборах с разными seed: одна генерация даёт шумную цифру."""
    settings = config["kz_scenarios"]
    rule = fast_transit_rule(config)
    rows = []
    for seed in seeds:
        world = generate_world(settings, seed)
        edges = build_world_edges(world.transfers, settings["tenge_per_usd"])
        features = build_world_features(edges, world.clients, settings["features_hours"],
                                        tuple(config["features"]["night_hours"]))
        flags = rule.flags(features)
        drops = features["label"] == DROP
        network_of = world.accounts.drop_duplicates("account_id").set_index("account_id")["network_id"]
        found = features.loc[flags & drops, "account_id"]
        rows.append({"seed": seed, "drops": int(drops.sum()), "drops_found": len(found),
                     "networks_found": int(network_of.reindex(found).nunique()),
                     "networks": int(world.accounts["network_id"].nunique()),
                     "background_alerts": int((flags & ~features["account_id"].isin(world.accounts["account_id"])).sum())})
        print(rows[-1], flush=True)
    shares = [row["drops_found"] / row["drops"] for row in rows]
    return {"seeds": rows, "share_min": min(shares), "share_median": float(np.median(shares)), "share_max": max(shares)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=None, help="только посчитать разброс правила на N других seed")
    parser.add_argument("--seed", type=int, default=None, help="seed набора, по умолчанию из config.yaml")
    args = parser.parse_args()
    config = load_config()
    settings = config["kz_scenarios"]
    processed = ROOT / config["paths"]["processed_dir"]
    reports = ROOT / config["paths"]["reports_dir"]
    if args.seeds is not None:
        seeds = [settings["seed"] + 1 + i for i in range(args.seeds)]
        (reports / "kz_seeds.json").write_text(json.dumps(seed_spread(config, seeds), ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        return

    seed = settings["seed"] if args.seed is None else args.seed
    world = generate_world(settings, seed)
    edges = build_world_edges(world.transfers, settings["tenge_per_usd"])
    features = build_world_features(edges, world.clients, settings["features_hours"],
                                    tuple(config["features"]["night_hours"]))

    features.to_parquet(processed / "kz_features.parquet", index=False)
    world.accounts.to_parquet(processed / "kz_accounts.parquet", index=False)
    world.clients.to_parquet(processed / "kz_clients.parquet", index=False)
    world.complaints.to_parquet(processed / "kz_complaints.parquet", index=False)
    edges[["timestamp", "from_account", "to_account", "amount", "paid_usd", "payment_format", "payment_currency",
           "is_self", "is_withdrawal"]].astype({"payment_format": "object", "payment_currency": "object"}).to_parquet(
        processed / "edges_kz.parquet", index=False)
    if seed == settings["seed"]:
        (reports / "demo_case.json").write_text(json.dumps({"seed": seed, **world.demo}, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    # Индекс похожих счетов для интерфейса: масштаб обучен на периоде обучения IBM.
    models = ROOT / "models"
    scaler = VectorScaler.load(models / "similarity_scaler.npz")
    metric = json.loads((reports / "similarity.json").read_text(encoding="utf-8"))["chosen_metric"]
    vectors = scaler.transform(features)
    search = config["similarity"]
    index = build_hnsw(vectors, metric, search["m"], search["ef_construction"], search["ef_search"])
    SimilarityIndex(index, features["account_id"].tolist(), vectors, metric).save(models, "similarity_kz")

    drops = set(features.loc[features["label"] == DROP, "account_id"])
    mules = set(world.accounts.loc[world.accounts["role"] == ROLE_MULE, "account_id"])
    print(json.dumps({"seed": seed, "networks": int(world.accounts["network_id"].nunique()),
                      "transfers": len(world.transfers), "clients": len(world.clients),
                      "mules": len(mules), "drops_in_features": len(drops), "mules_not_drops": len(mules - drops),
                      "complaints": len(world.complaints), "demo": world.demo}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
