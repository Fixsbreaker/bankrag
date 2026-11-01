"""Тесты загрузки на маленьких фикстурах."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankrag.data_loader import load_patterns, load_transactions, make_account_id, summarize

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering\n")
ROWS = [
    "2022/09/02 10:00,010,AAA,0020,BBB,5.00,Euro,5.00,Euro,ACH,1\n",
    "2022/09/01 09:30,0010,AAA,020,CCC,1.50,US Dollar,1.50,US Dollar,Cheque,0\n",
    "2022/09/05 00:00,020,BBB,020,BBB,2.00,Euro,2.00,Euro,Reinvestment,0\n",
]
PATTERNS = """
BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out
2022/09/02 10:00,010,AAA,0020,BBB,5.00,Euro,5.00,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - STACK
2022/09/03 11:00,020,BBB,030,DDD,4.00,Euro,4.00,Euro,ACH,1
END LAUNDERING ATTEMPT - STACK
"""


@pytest.fixture
def transactions_path(tmp_path: Path) -> Path:
    path = tmp_path / "trans.csv"
    path.write_text(HEADER + "".join(ROWS), encoding="utf-8")
    return path


def test_account_id_ignores_leading_zeros_and_keeps_banks_apart() -> None:
    ids = make_account_id(pd.Series(["010", "0010", "020"], dtype="string"),
                          pd.Series(["AAA", "AAA", "AAA"], dtype="string"))
    assert ids.tolist() == ["10_AAA", "10_AAA", "20_AAA"]


def test_load_transactions_reads_both_account_columns_and_sorts(transactions_path: Path) -> None:
    df = load_transactions(transactions_path)
    assert df["timestamp"].is_monotonic_increasing
    assert df.loc[0, "from_account"] == "10_AAA"
    assert df.loc[0, "to_account"] == "20_CCC"
    assert df["is_laundering"].sum() == 1


def test_sample_keeps_only_first_days(transactions_path: Path) -> None:
    assert len(load_transactions(transactions_path, sample_days=2)) == 2


def test_load_patterns_assigns_attempts_and_types(tmp_path: Path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERNS, encoding="utf-8")
    df = load_patterns(path)
    assert df["attempt_id"].tolist() == [0, 1]
    assert df["pattern_type"].tolist() == ["FAN-OUT", "STACK"]
    assert df["pattern_detail"].tolist() == ["Max 2-degree Fan-Out", ""]


def test_summarize_counts(transactions_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERNS, encoding="utf-8")
    summary = summarize(load_transactions(transactions_path), load_patterns(path), listed_accounts=3)
    assert summary["transactions"] == 3
    assert summary["accounts_in_transactions"] == 3
    assert summary["laundering_transactions"] == 1
    assert summary["self_transfers"] == 1
    assert summary["laundering_transactions_found_in_patterns"] == 1
    assert summary["attempts_by_type"] == {"FAN-OUT": 1, "STACK": 1}
