"""Владельцы счетов казахстанского сценария и их сверка с перечнями АФМ.

В сгенерированном наборе у счетов не было владельцев. Здесь каждому клиенту даётся вымышленное ФИО,
дата рождения и ИИН, магазину и работодателю — название ТОО и БИН. ИИН и БИН нарочно недействительны
(неверная контрольная цифра), поэтому не могут совпасть с настоящими.

Тестовые записи configs/watchlist_demo.json получают по одному счёту:
- организация (DEMO-1) — магазин, которому платит больше всего разных клиентов: точное совпадение, красный;
- человек (DEMO-2, ФИО латиницей) — чистый клиент, чаще всех плативший этому магазину; у владельца то же ФИО
  кириллицей: нечёткое совпадение, «проверить вручную».
ФИО, которые совпали или похожи на запись настоящего перечня, пропускаются: реальным людям синтетические
переводы не приписываются.

    PYTHONPATH=src python -m bankrag.owners
"""
from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from bankrag.data_loader import ROOT, load_config
from bankrag.watchlist import ORGANIZATION, PERSON, Watchlist, load_watchlist, normalize_name

SURNAMES = ["Ахметов", "Беков", "Жумабаев", "Искаков", "Касымов", "Нурланов", "Оспанов", "Сапаров", "Тлеубаев",
            "Утепов", "Есенов", "Кенжебаев", "Мухамедов", "Садыков", "Ергалиев", "Аманов", "Бекмуратов", "Жаксылыков",
            "Иванов", "Ким", "Петров", "Смирнов", "Цой", "Шевченко"]
MALE_NAMES = ["Алибек", "Арман", "Ерлан", "Данияр", "Нурлан", "Тимур", "Азамат", "Руслан", "Санжар", "Олжас",
              "Сергей", "Андрей", "Дмитрий", "Максим"]
FEMALE_NAMES = ["Айгерим", "Асель", "Динара", "Жанна", "Камила", "Мадина", "Сауле", "Алия", "Дана", "Гульнара",
                "Елена", "Ольга", "Анна", "Наталья"]
FATHER_NAMES = ["Серик", "Болат", "Кайрат", "Марат", "Ержан", "Бауыржан", "Асхат", "Талгат", "Игорь", "Виктор"]
OWNER_COLUMNS = ["account_id", "owner_kind", "owner_name", "iin_bin", "birth_date"]


def _feminine(surname: str) -> str:
    return surname + "а" if surname.endswith(("ов", "ев", "ин")) else surname


def _patronymic(father: str, female: bool) -> str:
    base = father[:-1] + "ь" if father.endswith("й") else father
    if base.endswith("ь"):
        return base[:-1] + ("евна" if female else "евич")
    return base + ("овна" if female else "ович")


def _control_digit(digits: str) -> int:
    """Контрольная цифра ИИН/БИН по открытому алгоритму. Результат 10 значит, что номер невозможен."""
    weights = [list(range(1, 12)), [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]]
    for w in weights:
        value = sum(int(d) * k for d, k in zip(digits, w)) % 11
        if value != 10:
            return value
    return 10


def invalid_number(first_eleven: str) -> str:
    """12-значный номер с заведомо неверной контрольной цифрой."""
    control = _control_digit(first_eleven)
    return first_eleven + str((control + 1) % 10)


def is_valid_number(number: str) -> bool:
    """Проверка контрольной цифры ИИН/БИН."""
    if len(number) != 12 or not number.isdigit():
        return False
    control = _control_digit(number[:11])
    return control != 10 and control == int(number[11])


# Владелец демо-клиента: то же ФИО, что у записи DEMO-2, но кириллицей.
DEMO_PERSON_OWNER = "Тестов Кайрат Демоевич"


def generate_owners(clients: pd.DataFrame, edges: pd.DataFrame, real_names: set[str], demo: list[dict] | None,
                    seed: int, labels: pd.Series | None = None,
                    similar_to_real: Callable[[str, str, str | None], bool] | None = None,
                    ) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Владельцы всех счетов, кроме банкоматов.

    Возвращает таблицу, счета демо (id записи → счёт) и число пропущенных ФИО, совпавших или похожих на
    записи настоящих перечней.
    """
    random = np.random.default_rng(seed)
    rows: list[dict] = []
    skipped = 0
    used: set[str] = set()
    for row in clients.sort_values("account_id").itertuples(index=False):
        if row.kind == "atm":
            continue
        if row.kind in ("merchant", "employer"):
            word = "Магазин" if row.kind == "merchant" else "Компания"
            number = row.account_id.split("-")[-1]
            bin_head = f"{random.integers(5, 25):02d}{random.integers(1, 13):02d}40{random.integers(0, 10**5):05d}"
            rows.append({"account_id": row.account_id, "owner_kind": ORGANIZATION,
                         "owner_name": f"ТОО «{word} №{number}»", "iin_bin": invalid_number(bin_head), "birth_date": None})
            continue
        while True:
            female = bool(random.integers(0, 2))
            surname = str(random.choice(SURNAMES))
            name = f"{_feminine(surname) if female else surname} {random.choice(FEMALE_NAMES if female else MALE_NAMES)} " \
                   f"{_patronymic(str(random.choice(FATHER_NAMES)), female)}"
            birth = pd.Timestamp(1960, 1, 1) + pd.Timedelta(days=int(random.integers(0, 44 * 365)))
            key = (normalize_name(name), birth.date().isoformat())
            if normalize_name(name) in real_names or (
                    similar_to_real is not None and similar_to_real(name, PERSON, birth.date().isoformat())):
                skipped += 1
                continue
            if key not in used:
                used.add(key)
                break
        century = (3 if birth.year < 2000 else 5) + (1 if female else 0)
        head = f"{birth:%y%m%d}{century}{random.integers(0, 10**4):04d}"
        rows.append({"account_id": row.account_id, "owner_kind": PERSON, "owner_name": name,
                     "iin_bin": invalid_number(head), "birth_date": birth.date().isoformat()})
    owners = pd.DataFrame(rows, columns=OWNER_COLUMNS)

    demo_accounts: dict[str, str] = {}
    by_kind = {str(r.get("type", "")).startswith(ORGANIZATION): r for r in (demo or [])}
    org, person = by_kind.get(True), by_kind.get(False)
    merchant = None
    if org is not None:
        merchants = set(clients.loc[clients["kind"] == "merchant", "account_id"])
        incoming = edges[edges["to_account"].isin(merchants)]
        # Магазин, которому платит больше всего разных клиентов: в демо видно, что переводов к лицу из перечня много.
        merchant = str(incoming.groupby("to_account")["from_account"].nunique().sort_values(
            ascending=False, kind="stable").index[0])
        owners.loc[owners["account_id"] == merchant, ["owner_name", "iin_bin"]] = [org["full_name"], org["iin"]]
        demo_accounts[str(org["id"])] = merchant
    if person is not None:
        ordinary = set(clients.loc[clients["kind"] == "client", "account_id"])
        if labels is not None:
            ordinary &= set(labels[labels == "clean"].index)
        candidates = edges[edges["from_account"].isin(ordinary)]
        if merchant is not None and (candidates["to_account"] == merchant).any():
            candidates = candidates[candidates["to_account"] == merchant]
        client = str(candidates.groupby("from_account").size().sort_values(ascending=False, kind="stable").index[0])
        birth = person["date_of_birth"]
        head = f"{pd.Timestamp(birth):%y%m%d}3{random.integers(0, 10**4):04d}"
        owners.loc[owners["account_id"] == client, ["owner_name", "iin_bin", "birth_date"]] = [
            DEMO_PERSON_OWNER, invalid_number(head), birth]
        demo_accounts[str(person["id"])] = client
    return owners, demo_accounts, skipped


def screen_owners(owners: pd.DataFrame, watchlist: Watchlist) -> pd.DataFrame:
    """Совпадения владельцев с перечнями, точные и похожие: одна строка на совпадение."""
    rows = []
    similar = watchlist.screen_similar_many(list(zip(owners["owner_name"], owners["owner_kind"], owners["birth_date"])))
    for row, fuzzy in zip(owners.itertuples(index=False), similar):
        for hit in watchlist.screen(row.owner_name, row.owner_kind, row.birth_date, row.iin_bin) + fuzzy:
            rows.append({"account_id": row.account_id, **hit.as_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    processed = ROOT / config["paths"]["processed_dir"]
    reports = ROOT / config["paths"]["reports_dir"]
    clients = pd.read_parquet(processed / "kz_clients.parquet")
    edges = pd.read_parquet(processed / "edges_kz.parquet")
    threshold = config["watchlist"]["fuzzy_threshold"]
    watchlist = load_watchlist(ROOT, fuzzy_threshold=threshold)
    real = load_watchlist(ROOT, include_demo=False, fuzzy_threshold=threshold)
    real_names = {normalize_name(r.full_name) for r in real.records}
    demo_path = ROOT / "configs" / "watchlist_demo.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.exists() else None
    labels = pd.read_parquet(processed / "kz_scores.parquet").set_index("account_id")["label"]
    owners, demo_accounts, skipped = generate_owners(
        clients, edges, real_names, demo, config["kz_scenarios"]["seed"], labels,
        lambda name, kind, birth: bool(real.screen_similar(name, kind, birth)))
    owners.to_parquet(processed / "kz_owners.parquet", index=False)
    hits = screen_owners(owners, watchlist)
    summary = {
        "lists_downloaded_at": watchlist.downloaded_at,
        "list_records": watchlist.counts,
        "real_records": watchlist.real_count,
        "owners": len(owners),
        "persons": int((owners["owner_kind"] == PERSON).sum()),
        "organizations": int((owners["owner_kind"] == ORGANIZATION).sum()),
        "generated_names_skipped_as_real": skipped,
        "fuzzy_threshold": threshold,
        "demo_accounts": demo_accounts,
        "hits": hits.to_dict("records") if len(hits) else [],
        "hits_real_lists": int((~hits["is_demo"]).sum()) if len(hits) else 0,
        "similar_hits_real_lists": int((~hits["is_demo"] & ~hits["strict"]).sum()) if len(hits) else 0,
    }
    (reports / "watchlist.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "hits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
