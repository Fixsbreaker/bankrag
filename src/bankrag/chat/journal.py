"""Журнал запросов: кто спросил, когда, что, какие функции вызваны, что ответил помощник.

Хранится локально в SQLite. Решения аналитика пишет feedback.py, журнал показывает их рядом.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        analyst TEXT,
        dataset TEXT,
        mode TEXT,
        question TEXT NOT NULL,
        calls TEXT,
        answer TEXT,
        checked INTEGER,
        fallback INTEGER,
        seconds REAL)""")
    return connection


def log_query(path: Path, analyst: str, dataset: str, mode: str, question: str, calls: list[dict], answer: str,
              checked: bool, fallback: bool, seconds: float) -> int:
    """Добавляет запись и возвращает её номер."""
    brief = [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
    with _connect(path) as connection:
        cursor = connection.execute(
            "INSERT INTO queries (created_at, analyst, dataset, mode, question, calls, answer, checked, fallback, seconds)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), analyst, dataset, mode, question,
             json.dumps(brief, ensure_ascii=False), answer, int(checked), int(fallback), float(seconds)))
        number = int(cursor.lastrowid)
    connection.close()
    return number


def read_queries(path: Path, limit: int = 200) -> pd.DataFrame:
    """Последние записи журнала, новые сверху."""
    with _connect(path) as connection:
        frame = pd.read_sql_query("SELECT * FROM queries ORDER BY id DESC LIMIT ?", connection, params=(limit,))
    connection.close()
    return frame
