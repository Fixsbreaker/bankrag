"""Проверка чисел: каждое число в ответе модели должно быть в результатах функций.

Учитываются пробелы в тысячах, запятая в дробях, «млн», проценты, время 14:05 и номера счетов.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SPACES = "    "
ACCOUNT = re.compile(r"\b(?:KZ-\d{4}|ATM-\d{2}|\d+_[0-9A-F]{6,})\b")
TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
NUMBER = re.compile(rf"(?<![\w.,])\d{{1,3}}(?:[{SPACES}]\d{{3}})+(?:,\d+)?(?![\w])|(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
MILLION = re.compile(r"^\s*млн")
THOUSAND = re.compile(r"^\s*тыс")


@dataclass
class Check:
    """Итог проверки: прошёл ли ответ и какие числа или счета не найдены."""

    ok: bool
    unknown: list[str] = field(default_factory=list)


def _to_float(text: str) -> float:
    return float(re.sub(rf"[{SPACES}]", "", text).replace(",", "."))


def extract(text: str, skip_list_markers: bool = True) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    """Числа (как написаны и значение), время и номера счетов из текста.

    Номера пунктов списка в ответе не проверяются, а в результатах функций они нужны: «часть 7».
    """
    accounts = ACCOUNT.findall(text)
    clean = ACCOUNT.sub(" ", text)
    times = [f"{int(h):02d}:{m}" for h, m in TIME.findall(clean)]
    clean = TIME.sub(" ", clean)
    if skip_list_markers:
        clean = LIST_MARKER.sub(" ", clean)
    numbers = []
    for match in NUMBER.finditer(clean):
        value = _to_float(match.group())
        tail = clean[match.end():match.end() + 6]
        if MILLION.match(tail):
            value *= 1_000_000
        elif THOUSAND.match(tail):
            value *= 1_000
        numbers.append((match.group(), value))
    return numbers, times, accounts


def _walk(value: Any, out: list) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _walk(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, out)
    elif isinstance(value, bool) or value is None:
        return
    else:
        out.append(value)


@dataclass
class Allowed:
    """Числа, время и счета, которые можно упоминать в ответе."""

    numbers: set[float] = field(default_factory=set)
    times: set[str] = field(default_factory=set)
    accounts: set[str] = field(default_factory=set)

    def add_number(self, value: float) -> None:
        # Допустимые записи одного значения: целое, десятые, доля как процент, миллионы и тысячи.
        for form in (value, round(value), round(value, 1), round(value, 2), round(value * 100),
                     round(value / 1_000_000, 1) * 1_000_000, round(value / 1_000) * 1_000):
            self.numbers.add(round(float(form), 2))

    def add_text(self, text: str) -> None:
        numbers, times, accounts = extract(text, skip_list_markers=False)
        for _, value in numbers:
            self.add_number(value)
        self.times.update(times)
        self.accounts.update(accounts)

    def add(self, payload: Any) -> None:
        values: list = []
        _walk(payload, values)
        for value in values:
            if isinstance(value, (int, float)):
                self.add_number(float(value))
            else:
                self.add_text(str(value))


def check_answer(answer: str, allowed: Allowed) -> Check:
    """Сверяет числа, время и счета ответа с разрешёнными."""
    numbers, times, accounts = extract(answer)
    unknown = [raw for raw, value in numbers if round(value, 2) not in allowed.numbers]
    unknown += [t for t in times if t not in allowed.times]
    unknown += [a for a in accounts if a not in allowed.accounts]
    return Check(not unknown, unknown)
