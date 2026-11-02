"""Тесты функций чата на маленькой фикстуре: одна сеть дропов, магазин и банкоматы."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from docx import Document

from bankrag.data_loader import load_config
from bankrag.flags import BLUE, FLAG_CASH, FLAG_FAST, FLAG_SENDERS, FLAG_SPLIT, FLAG_TRANSIT, GREY, RED
from bankrag.tools import fmt
from bankrag.tools import functions as F
from bankrag.tools.data import KZ, CaseData

DAY = pd.Timestamp("2026-11-03")


def _t(hours: float) -> pd.Timestamp:
    return DAY + pd.Timedelta(hours=hours)


def _edge(hours: float, source: str, target: str, amount: float, cash: bool = False) -> dict:
    return {"timestamp": _t(hours), "from_account": source, "to_account": target, "amount": amount,
            "payment_format": "Cash" if cash else "Credit Card", "is_self": source == target, "is_withdrawal": cash}


@pytest.fixture
def data() -> CaseData:
    edges = [
        _edge(10, "KZ-1000", "KZ-1001", 100_000),             # пострадавшая → первый дроп
        _edge(10.2, "KZ-1001", "KZ-1002", 96_000),            # через 12 минут дальше
        _edge(10.4, "KZ-1002", "KZ-1003", 47_000),            # дробление на два счёта
        _edge(10.42, "KZ-1002", "KZ-1004", 46_000),
        _edge(12, "KZ-1003", "ATM-01", 46_000, cash=True),    # снятие наличных
        _edge(11, "KZ-1004", "ATM-02", 45_000, cash=True),
        _edge(11, "KZ-1001", "KZ-7000", 3_000),               # покупка в магазине
        _edge(10 + 90, "KZ-1002", "KZ-1005", 1_000),          # позже окна в 72 часа
    ]
    # Ещё пять отправителей за сутки на первый дроп.
    edges += [_edge(1 + i, f"KZ-20{i:02d}", "KZ-1001", 20_000 + i) for i in range(5)]
    edges += [_edge(1 + i, "KZ-1000", "KZ-7000", 1_000 + i) for i in range(6)]
    kinds = pd.Series({"KZ-7000": "merchant", "ATM-01": "atm", "ATM-02": "atm"})
    complaints = pd.DataFrame([
        {"account_id": "KZ-1000", "target_account": "KZ-1001", "transfer_time": _t(10), "filed_at": _t(14),
         "amount_tenge": 100_000.0, "text": "Позвонили из «службы безопасности банка»."},
        {"account_id": "KZ-2001", "target_account": "KZ-1001", "transfer_time": _t(2), "filed_at": _t(5),
         "amount_tenge": 20_001.0, "text": "Предоплата продавцу."},
    ])
    scores = pd.DataFrame({"risk_score": [0.9, 0.01], "reasons": [["входящих переводов: 6"], []]},
                          index=["KZ-1003", "KZ-1002"])
    return CaseData(KZ, "KZT", pd.DataFrame(edges), scores, kinds, complaints, load_config(), high_risk=0.5)


def test_flags_mark_each_role(data: CaseData) -> None:
    flags = data.flags
    assert flags.at["KZ-1001", FLAG_SENDERS] and flags.at["KZ-1001", FLAG_FAST]
    # Деньги пяти других отправителей остались на счёте: дальше ушло меньше 80% всего полученного.
    assert not flags.at["KZ-1001", FLAG_TRANSIT]
    assert flags.at["KZ-1002", FLAG_SPLIT] and flags.at["KZ-1002", FLAG_TRANSIT]
    assert flags.at["KZ-1003", FLAG_CASH] and flags.at["KZ-1004", FLAG_CASH]
    # Покупка в магазине — не пересылка денег.
    assert flags.at["KZ-1000", "flag_count"] == 0


def test_colors(data: CaseData) -> None:
    assert data.color("KZ-1000") == BLUE
    assert data.color("KZ-1001") == RED
    assert data.color("KZ-1005") == GREY


def test_trace_money_follows_branches_and_stops_on_cash(data: CaseData) -> None:
    result = F.trace_money(data, "kz 1001", start_time="3 ноября 10:00", amount="100 000")
    steps = result["steps"]
    assert steps[0]["from"] == "KZ-1000" and steps[0]["amount"] == 100_000
    assert [(s["from"], s["to"]) for s in steps[1:]] == [
        ("KZ-1001", "KZ-1002"), ("KZ-1002", "KZ-1003"), ("KZ-1002", "KZ-1004"),
        ("KZ-1004", F.CASH_NODE), ("KZ-1003", F.CASH_NODE)]
    assert steps[1]["delay_minutes"] == 12 and steps[1]["share_text"] == "96%"
    assert result["cash_total"] == 91_000
    assert result["retained"] == pytest.approx(100_000 - 91_000)
    assert "KZ-7000" not in result["accounts"] and "KZ-1005" not in result["accounts"]
    assert "96 000 ₸" in result["summary"]


def test_trace_money_respects_depth_and_window(data: CaseData) -> None:
    short = F.trace_money(data, "KZ-1001", start_time="2026-11-03 10:00", max_depth=1)
    assert [s["to"] for s in short["steps"][1:]] == ["KZ-1002"]
    narrow = F.trace_money(data, "KZ-1002", start_time="2026-11-03 10:12", window_hours=0.1)
    assert len(narrow["steps"]) == 1


def test_trace_money_errors(data: CaseData) -> None:
    assert "не найден" in F.trace_money(data, "KZ-9999")["summary"]
    assert "нет" in F.trace_money(data, "KZ-1001", start_time="1 ноября 09:00")["summary"]


def test_trace_sources_counts_senders_and_complaints(data: CaseData) -> None:
    # Последний перевод фикстуры через 90 часов, поэтому окно шире 72 часов.
    result = F.trace_sources(data, "KZ-1001", window_hours=120)
    assert result["sender_count"] == 6 and result["transfer_count"] == 6
    assert result["complaint_count"] == 2
    assert {s["account"] for s in result["senders"] if s["complaint"]} == {"KZ-1000", "KZ-2001"}


def test_account_profile(data: CaseData) -> None:
    profile = F.account_profile(data, "KZ-1002")
    assert profile["flag_count"] >= 2 and "дробление" in profile["flags"]
    assert profile["risk_text"] == "0,010"
    assert F.account_profile(data, "KZ-9999")["error"] == "Счёт KZ-9999 не найден в данных."


def test_find_network_skips_shops_and_marks_colors(data: CaseData) -> None:
    result = F.find_network(data, ["KZ-1002"], depth=2)
    assert {"KZ-1001", "KZ-1003", "KZ-1004", "KZ-1000"} <= set(result["accounts"])
    assert "KZ-7000" not in result["accounts"]
    assert result["victim_count"] >= 1
    assert any(edge["to"] == F.CASH_NODE for edge in result["graph"]["edges"])


def test_list_suspicious_puts_flags_first(data: CaseData) -> None:
    rows = F.list_suspicious(data, limit=3)["rows"]
    assert rows[0]["account"] in {"KZ-1001", "KZ-1002"}
    assert all(r["account"] not in {"KZ-1000", "KZ-7000"} for r in rows)


def test_make_report_writes_docx(data: CaseData, tmp_path: Path) -> None:
    result = F.make_report(data, ["KZ-1001", "KZ-1002", "KZ-1003", "KZ-1004"], "Тестовое дело", out_dir=tmp_path)
    document = Document(result["path"])
    text = "\n".join(p.text for p in document.paragraphs)
    assert "Статья 232-1" in text and "уполномоченный орган" in text
    # Три перевода между счетами дела, два снятия наличных и два поступления от пострадавших.
    assert result["transfer_count"] == 7
    assert len(document.tables) == 2 and len(document.inline_shapes) == 1


def test_formatting() -> None:
    assert fmt.money(480_000, "KZT") == "480 000 ₸"
    assert fmt.money_short(2_100_000, "KZT") == "2,1 млн ₸"
    assert fmt.duration(160) == "2 ч 40 мин"
    assert [fmt.plural(n, "счёт", "счёта", "счетов") for n in (1, 4, 11, 22)] == ["1 счёт", "4 счёта", "11 счетов", "22 счёта"]
