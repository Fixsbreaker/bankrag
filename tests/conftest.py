"""Общая фикстура: маленький набор с одной сетью дропов, магазином и банкоматами."""
from __future__ import annotations

import pandas as pd
import pytest

from bankrag.data_loader import load_config
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
