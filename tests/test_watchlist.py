"""Сверка с перечнями АФМ: нормализация, правила совпадения, владельцы, функция чата и справка."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from docx import Document

from bankrag.flags import ORANGE, RED
from bankrag.owners import generate_owners, invalid_number, is_valid_number
from bankrag.tools import functions as F
from bankrag.tools.data import CaseData
from bankrag.watchlist import (
    MATCH_IIN,
    MATCH_NAME_DOB,
    MATCH_NAME_ONLY,
    MATCH_ORG,
    ORGANIZATION,
    latin_key,
    PERSON,
    Watchlist,
    normalize_name,
    parse_record,
)

PERSON_RAW = {"id": 1, "full_name": "СЫЗДЫКОВ\tБАХТИЯР АДИЛОВИЧ", "type": "person", "status": "acting",
              "added_to_list": "2026-10-29", "date_of_birth": "1989-06-09"}
NO_DOB_RAW = {"id": 2, "full_name": "Abuliezi Abuduhamiti", "type": "person", "date_of_birth": None}
ORG_RAW = {"id": 3, "full_name": "«Sweet Money»  ", "type": "organizationcis\n", "iin": None}
WEAPON_RAW = {"id": "4220", "name": "САРВАТ  САЛАХ ", "dob": "1960-06-29", "listed_at": "2001-10-06", "other": "…"}
DEMO_RAW = {"id": "DEMO-1", "full_name": "ТОО «Тестовая запись BankRAG»", "iin": "990000000001",
            "type": "organization", "added_to_list": "2026-11-02"}
DEMO_PERSON_RAW = {"id": "DEMO-2", "full_name": "Testov Qairat Demoevich", "type": "person",
                   "date_of_birth": "1991-03-14", "added_to_list": "2026-11-02"}
OTHER_SURNAME_RAW = {"id": 5, "full_name": "ШАКИРОВ РУСЛАН МАРАТОВИЧ", "type": "person", "date_of_birth": None}


@pytest.fixture
def watchlist() -> Watchlist:
    records = [parse_record("sanction_category_1", PERSON_RAW), parse_record("sanction_category_2", NO_DOB_RAW),
               parse_record("sanction_category_1", ORG_RAW), parse_record("sanction_category_4", WEAPON_RAW),
               parse_record("demo", DEMO_RAW), parse_record("demo", DEMO_PERSON_RAW),
               parse_record("sanction_category_2", OTHER_SURNAME_RAW)]
    return Watchlist(records, "2026-11-05T08:00:00", {"sanction_category_1": 2, "sanction_category_2": 2,
                                                      "sanction_category_4": 1, "demo": 2})


def test_normalize_ignores_case_quotes_order_and_kazakh_letters() -> None:
    assert normalize_name("ТОО «Sweet Money»") == normalize_name("sweet money")
    assert normalize_name("Смаил Әнуарбек") == normalize_name("АНУАРБЕК СМАИЛ")
    assert normalize_name("СЫЗДЫКОВ\tБАХТИЯР") == normalize_name("Бахтияр Сыздыков")


def test_parse_both_api_formats() -> None:
    org = parse_record("sanction_category_1", ORG_RAW)
    weapon = parse_record("sanction_category_4", WEAPON_RAW)
    assert org.kind == ORGANIZATION and org.full_name == "«Sweet Money»"
    assert weapon.kind == PERSON and weapon.date_of_birth == "1960-06-29" and weapon.added == "2001-10-06"


def test_person_needs_same_birth_date(watchlist: Watchlist) -> None:
    assert [h.match for h in watchlist.screen("Сыздыков Бахтияр Адилович", PERSON, "1989-06-09")] == [MATCH_NAME_DOB]
    # Однофамилец с другой датой рождения — не совпадение.
    assert watchlist.screen("Сыздыков Бахтияр Адилович", PERSON, "1990-01-01") == []


def test_person_without_birth_date_in_list_is_flagged_for_check(watchlist: Watchlist) -> None:
    assert [h.match for h in watchlist.screen("Abuduhamiti Abuliezi", PERSON, "1982-10-04")] == [MATCH_NAME_ONLY]


def test_organization_by_name_and_by_bin(watchlist: Watchlist) -> None:
    assert [h.match for h in watchlist.screen("ТОО Sweet Money", ORGANIZATION)] == [MATCH_ORG]
    assert watchlist.screen("Sweet Money", PERSON) == []
    hits = watchlist.screen("Другое название", ORGANIZATION, iin="990000000001")
    assert [h.match for h in hits] == [MATCH_IIN] and hits[0].as_dict()["is_demo"]


def test_lookup_by_name_without_birth_date(watchlist: Watchlist) -> None:
    assert len(watchlist.lookup("Бахтияр Сыздыков Адилович")) == 1
    assert watchlist.lookup("Иванов Иван") == []
    assert watchlist.real_count == 5


def test_latin_key_joins_scripts_and_kazakh_letters() -> None:
    assert len({latin_key(n) for n in ("Қайрат", "Кайрат", "Kairat", "Qairat")}) == 1
    assert latin_key("Жумабаев Ержан") == latin_key("Zhumabayev Yerzhan")


def test_similar_spelling_is_only_a_manual_check(watchlist: Watchlist) -> None:
    hits = watchlist.screen_similar("Тестов Кайрат Демоевич", PERSON, "1991-03-14")
    assert [h.record.record_id for h in hits] == ["DEMO-2"] and not hits[0].strict
    assert watchlist.screen("Тестов Кайрат Демоевич", PERSON, "1991-03-14") == []
    # Опечатка в имени.
    assert [h.record.record_id for h in watchlist.screen_similar("Сыздыков Бахтиар Адилович", PERSON, "1989-06-09")] == ["1"]


def test_similar_spelling_respects_birth_date_kind_and_surname(watchlist: Watchlist) -> None:
    assert watchlist.screen_similar("Тестов Кайрат Демоевич", PERSON, "1970-01-01") == []
    assert watchlist.screen_similar("Тестов Кайрат Демоевич", ORGANIZATION) == []
    # Одинаковые имя и отчество при другой фамилии не считаются похожими.
    assert watchlist.screen_similar("Садыков Руслан Маратович", PERSON, "1999-08-16") == []
    # Точное совпадение не дублируется похожим.
    assert watchlist.screen_similar("Сыздыков Бахтияр Адилович", PERSON, "1989-06-09") == []


def test_lookup_returns_similar_after_exact(watchlist: Watchlist) -> None:
    hits = watchlist.lookup("Sweet Mony")
    assert [h.record.record_id for h in hits] == ["3"] and hits[0].similarity is not None


def test_generated_numbers_are_never_valid() -> None:
    for head in ("89060930012", "05011450001", "20034012345"):
        number = invalid_number(head)
        assert len(number) == 12 and not is_valid_number(number)


def test_generated_owners_skip_real_names_and_get_demo() -> None:
    clients = pd.DataFrame({"account_id": [f"KZ-{i}" for i in range(1000, 1040)] + ["KZ-7000", "KZ-7001", "ATM-01"],
                            "kind": ["client"] * 40 + ["merchant", "merchant", "atm"]})
    edges = pd.DataFrame({"from_account": ["KZ-1000", "KZ-1001", "KZ-1002"], "to_account": ["KZ-7001"] * 2 + ["KZ-7000"]})
    real = {normalize_name("Ахметов Алибек Серикович")}
    owners, demo_accounts, _ = generate_owners(clients, edges, real, [DEMO_RAW, DEMO_PERSON_RAW], seed=1)
    assert demo_accounts == {"DEMO-1": "KZ-7001", "DEMO-2": "KZ-1000"}
    table = owners.set_index("account_id")
    assert table.at["KZ-7001", "owner_name"] == DEMO_RAW["full_name"]
    assert table.at["KZ-1000", "birth_date"] == DEMO_PERSON_RAW["date_of_birth"]
    assert not is_valid_number(table.at["KZ-1000", "iin_bin"])
    assert "ATM-01" not in set(owners["account_id"])
    assert not owners["owner_name"].map(normalize_name).isin(real).any()


@pytest.fixture
def listed_data(data: CaseData, watchlist: Watchlist) -> CaseData:
    """Фикстура conftest, где магазин KZ-7000 принадлежит тестовой записи перечня."""
    owners = pd.DataFrame([
        {"account_id": "KZ-7000", "owner_kind": ORGANIZATION, "owner_name": DEMO_RAW["full_name"],
         "iin_bin": DEMO_RAW["iin"], "birth_date": None},
        {"account_id": "KZ-1001", "owner_kind": PERSON, "owner_name": "Беков Арман Маратович",
         "iin_bin": invalid_number("90010130001"), "birth_date": "1990-01-01"},
        {"account_id": "KZ-1002", "owner_kind": PERSON, "owner_name": "Тестов Кайрат Демоевич",
         "iin_bin": invalid_number("91031430001"), "birth_date": "1991-03-14"},
    ])
    return CaseData(data.name, data.currency, data.edges, data.scores, data.kinds, data.complaints, data.config,
                    data.high_risk, owners=owners, watchlist=watchlist)


def test_listed_account_is_red_and_first_in_queue(listed_data: CaseData) -> None:
    assert listed_data.color("KZ-7000") == RED
    assert F.list_suspicious(listed_data)["rows"][0]["account"] == "KZ-7000"
    assert F.account_profile(listed_data, "KZ-7000")["listed"]


def test_check_lists_for_listed_account_and_its_senders(listed_data: CaseData) -> None:
    listed = F.check_lists(listed_data, account="KZ-7000")
    assert listed["listed"] and listed["incoming_count"] == 7 and listed["incoming_senders"] == 2
    sender = F.check_lists(listed_data, account="KZ-1001")
    assert not sender["listed"] and [c["account"] for c in sender["counterparties"] if c["strict"]] == ["KZ-7000"]
    assert [c["account"] for c in sender["counterparties"] if not c["strict"]] == ["KZ-1002"]
    assert sender["counterparties"][1]["incoming_count"] == 1
    overview = F.check_lists(listed_data)
    assert overview["listed_count"] == 1 and "KZ-7000" in overview["summary"]
    assert overview["similar_count"] == 1 and "проверить вручную" in overview["summary"]


def test_similar_owner_is_orange_and_checked_manually(listed_data: CaseData) -> None:
    assert not listed_data.is_listed("KZ-1002") and listed_data.is_list_similar("KZ-1002")
    assert listed_data.color("KZ-1002") in (ORANGE, RED)
    result = F.check_lists(listed_data, account="KZ-1002")
    assert result["list_similar"] and not result["listed"] and "сверьте документы" in result["summary"]
    profile = F.account_profile(listed_data, "KZ-1002")
    assert profile["list_similar"] and not profile["listed"]
    rows = [r["account"] for r in F.list_suspicious(listed_data)["rows"]]
    assert rows[:2] == ["KZ-7000", "KZ-1002"]


def test_check_lists_without_owners_says_so(data: CaseData) -> None:
    result = F.check_lists(data, account="KZ-1001")
    assert result["owner"] is None and "нет данных о владельце" in result["summary"]
    assert result["summary"].count("Сверка с перечнями") == 1


def test_check_lists_by_name(listed_data: CaseData) -> None:
    assert F.check_lists(listed_data, name="Sweet Money")["hit_count"] == 1
    assert F.check_lists(listed_data, name="Нет Такого")["hit_count"] == 0
    latin = F.check_lists(listed_data, name="Syzdykov Bakhtiyar Adilovich")
    assert latin["hit_count"] == 0 and latin["similar_count"] == 1 and "проверить вручную" in latin["summary"]


def test_report_has_list_section_and_rows_for_single_account(listed_data: CaseData, tmp_path: Path) -> None:
    result = F.make_report(listed_data, ["KZ-7000"], out_dir=tmp_path)
    assert result["transfer_count"] == 7
    text = "\n".join(p.text for p in Document(result["path"]).paragraphs)
    assert "Проверка по перечням АФМ" in text and "тестовая запись демо" in text
