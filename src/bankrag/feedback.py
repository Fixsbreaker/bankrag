"""Решения аналитика: хранятся локально в SQLite."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CONFIRMED = "confirmed"
REJECTED = "rejected"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        risk_score REAL,
        created_at TEXT NOT NULL)""")
    # Базы старой версии без имени аналитика и набора данных дополняются столбцами.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(decisions)")}
    for column in ("analyst", "dataset"):
        if column not in columns:
            connection.execute(f"ALTER TABLE decisions ADD COLUMN {column} TEXT")
    return connection


def save_decision(path: Path, account_id: str, decision: str, risk_score: float | None = None,
                  analyst: str = "", dataset: str = "") -> None:
    """Добавляет решение. История не перезаписывается: действующим считается последнее."""
    if decision not in (CONFIRMED, REJECTED):
        raise ValueError(f"неизвестное решение {decision!r}")
    with _connect(path) as connection:
        connection.execute("INSERT INTO decisions (account_id, decision, risk_score, created_at, analyst, dataset)"
                           " VALUES (?, ?, ?, ?, ?, ?)",
                           (account_id, decision, risk_score, datetime.now(timezone.utc).isoformat(), analyst, dataset))
    connection.close()


def read_decisions(path: Path, limit: int = 500) -> pd.DataFrame:
    """История решений для журнала, новые сверху."""
    with _connect(path) as connection:
        frame = pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", connection, params=(limit,))
    connection.close()
    return frame


def latest_decision(path: Path, account_id: str) -> str | None:
    """Последнее решение по счёту или None."""
    with _connect(path) as connection:
        row = connection.execute("SELECT decision FROM decisions WHERE account_id = ? ORDER BY id DESC LIMIT 1",
                                 (account_id,)).fetchone()
    connection.close()
    return row[0] if row else None


def all_latest_decisions(path: Path) -> dict[str, str]:
    """Последние решения по всем счетам."""
    with _connect(path) as connection:
        rows = connection.execute("SELECT account_id, decision FROM decisions ORDER BY id").fetchall()
    connection.close()
    return dict(rows)
