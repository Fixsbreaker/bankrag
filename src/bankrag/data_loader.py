"""Загрузка набора IBM AML: приведение типов, id счёта, разбор файла паттернов.

    PYTHONPATH=src python -m bankrag.data_loader
    PYTHONPATH=src python -m bankrag.data_loader --sample 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# В CSV две колонки называются Account, поэтому имена задаются явно по порядку.
TRANSACTION_COLUMNS = [
    "timestamp", "from_bank", "from_number", "to_bank", "to_number",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]
TRANSACTION_DTYPES = {
    "from_bank": "string", "from_number": "string", "to_bank": "string", "to_number": "string",
    "amount_received": "float64", "receiving_currency": "category",
    "amount_paid": "float64", "payment_currency": "category",
    "payment_format": "category", "is_laundering": "int8",
}
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
ATTEMPT_HEADER = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)(?::\s*(.*))?$")


def load_config(path: Path = ROOT / "configs" / "config.yaml") -> dict:
    """Читает настройки проекта из YAML."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_account_id(bank: pd.Series, number: pd.Series) -> pd.Series:
    """Id счёта = банк + номер: одинаковые номера в разных банках — разные счета.

    В транзакциях банк записан с ведущими нулями в разном виде (010 и 0010),
    в списке счетов нули отсутствуют, поэтому код банка нормализуется.
    """
    return bank.str.lstrip("0") + "_" + number


def add_account_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонки from_account и to_account."""
    df["from_account"] = make_account_id(df["from_bank"], df["from_number"])
    df["to_account"] = make_account_id(df["to_bank"], df["to_number"])
    return df


def load_transactions(path: Path, sample_days: int | None = None) -> pd.DataFrame:
    """Читает CSV транзакций и сортирует по времени; sample_days оставляет первые N дней."""
    df = pd.read_csv(path, header=0, names=TRANSACTION_COLUMNS, dtype=TRANSACTION_DTYPES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    if sample_days is not None:
        df = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=sample_days)]
    df = add_account_ids(df)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_patterns(path: Path) -> pd.DataFrame:
    """Разбирает файл паттернов: одна строка на перевод с номером попытки и её типом."""
    rows: list[list[str]] = []
    attempts: list[tuple[int, str, str]] = []
    attempt_id = -1
    pattern_type = pattern_detail = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("END LAUNDERING ATTEMPT"):
            continue
        header = ATTEMPT_HEADER.match(line)
        if header:
            attempt_id += 1
            pattern_type, pattern_detail = header.group(1), (header.group(2) or "").strip()
            continue
        if line.startswith("BEGIN"):
            raise ValueError(f"неизвестный заголовок попытки: {line}")
        rows.append(line.split(","))
        attempts.append((attempt_id, pattern_type, pattern_detail))
    df = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    df["amount_paid"] = df["amount_paid"].astype("float64")
    df["amount_received"] = df["amount_received"].astype("float64")
    df["is_laundering"] = df["is_laundering"].astype("int8")
    df[["attempt_id", "pattern_type", "pattern_detail"]] = pd.DataFrame(attempts, index=df.index)
    return add_account_ids(df)


def summarize(transactions: pd.DataFrame, patterns: pd.DataFrame, listed_accounts: int) -> dict:
    """Считает сводку набора для отчёта. Все числа получаются из данных."""
    accounts = pd.concat([transactions["from_account"], transactions["to_account"]]).nunique()
    laundering = transactions[transactions["is_laundering"] == 1]
    laundering_accounts = pd.concat([laundering["from_account"], laundering["to_account"]]).nunique()
    days = transactions["timestamp"].dt.floor("D")
    per_day = pd.DataFrame({
        "transactions": transactions.groupby(days).size(),
        "laundering": laundering.groupby(laundering["timestamp"].dt.floor("D")).size(),
    }).fillna(0).astype(int)
    key = ["timestamp", "from_account", "to_account", "amount_paid", "payment_format"]
    matched = laundering[key].merge(patterns[key].drop_duplicates(), on=key, how="inner")
    return {
        "transactions": len(transactions),
        "accounts_in_transactions": int(accounts),
        "accounts_in_listing": listed_accounts,
        "laundering_transactions": len(laundering),
        "laundering_share": len(laundering) / len(transactions),
        "accounts_with_laundering": int(laundering_accounts),
        "self_transfers": int((transactions["from_account"] == transactions["to_account"]).sum()),
        "first_timestamp": str(transactions["timestamp"].min()),
        "last_timestamp": str(transactions["timestamp"].max()),
        "pattern_attempts": int(patterns["attempt_id"].nunique()),
        "pattern_transactions": len(patterns),
        "attempts_by_type": {k: int(v) for k, v in
                             patterns.groupby("pattern_type")["attempt_id"].nunique().items()},
        "laundering_transactions_found_in_patterns": len(matched),
        "per_day": {str(day.date()): {"transactions": int(row.transactions), "laundering": int(row.laundering)}
                    for day, row in per_day.iterrows()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, metavar="DAYS",
                        help="оставить только первые DAYS дней")
    args = parser.parse_args()

    config = load_config()
    prefix = config["dataset"]["prefix"]
    raw = ROOT / config["paths"]["raw_dir"]
    processed = ROOT / config["paths"]["processed_dir"]
    reports = ROOT / config["paths"]["reports_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(raw / f"{prefix}_Trans.csv", args.sample)
    patterns = load_patterns(raw / f"{prefix}_Patterns.txt")
    listed = len(pd.read_csv(raw / f"{prefix}_accounts.csv", usecols=["Account Number"]))

    transactions.to_parquet(processed / "transactions.parquet", index=False)
    patterns.to_parquet(processed / "patterns.parquet", index=False)

    summary = summarize(transactions, patterns, listed)
    summary["sample_days"] = args.sample
    (reports / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
