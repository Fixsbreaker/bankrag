"""Данные, с которыми работают функции чата: переводы, флажки, оценки модели, жалобы.

Набор загружается один раз. Интерфейс держит его в кеше, тесты собирают из маленькой фикстуры.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bankrag.flags import account_color, compute_flags
from bankrag.watchlist import Watchlist, load_watchlist

KZ = "kz"
IBM = "ibm"
DATASET_TITLES = {KZ: "Казахстанский сценарий (сгенерирован командой)", IBM: "IBM AML, тестовое окно (синтетика)"}
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
          "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
BUSINESS_KINDS = {"employer", "merchant", "atm"}
# Вклады признаков из bankrag.model.explain_rows (есть после `python -m bankrag.model --rescore`).
EXPLAIN_COLUMNS = ["up_text", "up_share", "up_hidden", "down_text", "down_share", "down_hidden"]
KIND_TITLES = {"client": "клиент", "employer": "работодатель", "merchant": "магазин", "atm": "банкомат"}


@dataclass
class CaseData:
    """Один набор данных для функций чата."""

    name: str
    currency: str
    edges: pd.DataFrame
    scores: pd.DataFrame
    kinds: pd.Series
    complaints: pd.DataFrame
    config: dict
    high_risk: float
    confirmed: set[str] = field(default_factory=set)
    similarity: Any = None
    knowledge: Any = None
    owners: pd.DataFrame = field(default_factory=pd.DataFrame)
    watchlist: Watchlist = field(default_factory=Watchlist)
    flags: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.edges = self.edges.sort_values("timestamp", kind="stable").reset_index(drop=True)
        self.flags = compute_flags(self.edges, self.config["flags"], self.businesses, self.confirmed)
        self.victims = set(self.complaints["account_id"]) if len(self.complaints) else set()
        if len(self.owners):
            self.owners = self.owners.set_index("account_id") if "account_id" in self.owners.columns else self.owners
        # Сверка всех владельцев с перечнями один раз при загрузке: счёт → совпадения.
        # list_hits — точные (красный), list_similar — похожее написание, «проверить вручную» (оранжевый).
        self.list_hits: dict[str, list[dict]] = {}
        self.list_similar: dict[str, list[dict]] = {}
        if len(self.owners):
            similar = self.watchlist.screen_similar_many(list(zip(
                self.owners["owner_name"], self.owners["owner_kind"], self.owners.get("birth_date"))))
            for (account, row), fuzzy in zip(self.owners.iterrows(), similar):
                hits = self.watchlist.screen(row["owner_name"], row["owner_kind"], row.get("birth_date"),
                                             row.get("iin_bin"))
                if hits:
                    self.list_hits[str(account)] = [h.as_dict() for h in hits]
                elif fuzzy:
                    self.list_similar[str(account)] = [h.as_dict() for h in fuzzy]

    def owner(self, account: str) -> dict | None:
        """Владелец счёта: ФИО или название, ИИН/БИН, дата рождения. В наборе IBM владельцев нет."""
        if account not in self.owners.index:
            return None
        row = self.owners.loc[account]
        birth = row.get("birth_date")
        return {"kind": row["owner_kind"], "name": row["owner_name"], "iin_bin": row["iin_bin"],
                "birth_date": None if birth is None or pd.isna(birth) else str(birth)}

    def is_listed(self, account: str) -> bool:
        return account in self.list_hits

    def is_list_similar(self, account: str) -> bool:
        """Владелец похож на запись перечня (не точное совпадение): проверить вручную."""
        return account in self.list_similar

    @property
    def businesses(self) -> set[str]:
        return set(self.kinds[self.kinds.isin(BUSINESS_KINDS)].index)

    @property
    def title(self) -> str:
        return DATASET_TITLES.get(self.name, self.name)

    @property
    def start(self) -> pd.Timestamp:
        return self.edges["timestamp"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.edges["timestamp"].max()

    def set_confirmed(self, confirmed: set[str]) -> None:
        """Пересчитывает флажок связи с подтверждённым дропом после решения аналитика."""
        if confirmed != self.confirmed:
            self.confirmed = set(confirmed)
            self.flags = compute_flags(self.edges, self.config["flags"], self.businesses, self.confirmed)

    def has(self, account: str) -> bool:
        return account in self.flags.index

    def kind(self, account: str) -> str:
        return str(self.kinds.get(account, "client"))

    def risk(self, account: str) -> float | None:
        return float(self.scores.at[account, "risk_score"]) if account in self.scores.index else None

    def model_reasons(self, account: str) -> list[str]:
        return list(self.scores.at[account, "reasons"]) if account in self.scores.index else []

    def model_explanation(self, account: str) -> dict | None:
        """Что повысило и что понизило оценку модели: фразы, доли и доля скрытых признаков IBM."""
        if account not in self.scores.index or "up_text" not in self.scores.columns:
            return None
        row = self.scores.loc[account]
        return {side: {"factors": [{"text": str(t), "share": float(v)} for t, v in zip(row[f"{side}_text"], row[f"{side}_share"])],
                       "hidden": float(row[f"{side}_hidden"])} for side in ("up", "down")}

    def color(self, account: str) -> str:
        row = self.flags.loc[account] if account in self.flags.index else None
        colors = self.config["colors"]
        return account_color(int(row["flag_count"]) if row is not None else 0, self.risk(account),
                             account in self.victims, float(row["transfers"]) if row is not None else 0,
                             self.high_risk, self.high_risk * colors["medium_risk_share"],
                             colors["green_min_transfers"], self.is_listed(account), self.is_list_similar(account))

    def normalize_account(self, value: str) -> str:
        """Номер счёта в виде из данных: «kz 1654», «KZ1654» и «1654» дают KZ-1654."""
        text = str(value).strip().strip("«»\"'").upper().replace(" ", "")
        if self.name == KZ:
            match = re.fullmatch(r"(?:KZ-?)?(\d{4})", text)
            if match:
                return f"KZ-{match.group(1)}"
        return text

    def parse_time(self, value: Any) -> pd.Timestamp | None:
        """Время из вопроса: «2026-11-03 14:05», «3 ноября 14:05» или «3 ноября в 14:05»."""
        if value is None or value == "":
            return None
        text = str(value).strip().lower()
        match = re.search(r"(\d{1,2})\s+([а-я]+)\D*?(\d{1,2})[:.](\d{2})", text)
        if match and match.group(2) in MONTHS:
            return pd.Timestamp(self.end.year, MONTHS[match.group(2)], int(match.group(1)),
                                int(match.group(3)), int(match.group(4)))
        try:
            return pd.Timestamp(text.replace("t", " "))
        except ValueError:
            return None


def parse_amount(value: Any) -> float | None:
    """Сумма из вопроса: 500000, «500 000», «500 000 ₸»."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    digits = re.sub(r"[^\d,.]", "", str(value)).replace(",", ".")
    try:
        return float(digits)
    except ValueError:
        return None


def load_case_data(name: str, config: dict, root: Path, confirmed: set[str] | None = None) -> CaseData:
    """Загружает казахстанский набор или тестовое окно IBM из data/processed."""
    processed = root / config["paths"]["processed_dir"]
    high_risk = json.loads((root / config["paths"]["reports_dir"] / "model.json").read_text(encoding="utf-8"))["threshold"]
    if name == KZ:
        edges = pd.read_parquet(processed / "edges_kz.parquet")
        scores = pd.read_parquet(processed / "kz_scores.parquet")
        kinds = pd.read_parquet(processed / "kz_clients.parquet").set_index("account_id")["kind"]
        complaints = pd.read_parquet(processed / "kz_complaints.parquet")
        owners_path = processed / "kz_owners.parquet"
        owners = pd.read_parquet(owners_path) if owners_path.exists() else pd.DataFrame()
        currency = "KZT"
    else:
        edges = pd.read_parquet(processed / "edges_test.parquet")
        edges["amount"] = edges["paid_usd"]
        # В IBM формат Cash — платёж между счетами, а не снятие в банкомате (проверено на данных).
        edges["is_withdrawal"] = False
        scores = pd.read_parquet(processed / "scores.parquet")
        scores = scores[scores["window"] == "test"]
        kinds = pd.Series(dtype="object")
        complaints = pd.DataFrame(columns=["account_id", "target_account", "transfer_time", "filed_at",
                                           "amount_tenge", "text"])
        owners = pd.DataFrame()
        currency = "USD"
    edges = edges.astype({"from_account": "object", "to_account": "object"})
    explained = [c for c in EXPLAIN_COLUMNS if c in scores.columns]
    scores = scores.set_index("account_id")[["risk_score", "reasons", *explained]]
    return CaseData(name, currency, edges, scores, kinds, complaints, config, high_risk, set(confirmed or set()),
                    owners=owners,
                    watchlist=load_watchlist(root, fuzzy_threshold=config.get("watchlist", {}).get("fuzzy_threshold", 88)))
