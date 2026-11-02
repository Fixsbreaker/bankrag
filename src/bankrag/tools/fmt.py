"""Готовые строки для ответов: суммы, время, длительность.

Модель переписывает эти строки как есть, поэтому числа в ответе совпадают с данными.
"""
from __future__ import annotations

import math

import pandas as pd

CURRENCY_SIGNS = {"KZT": "₸", "USD": "USD"}
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


def number(value: float, digits: int = 0) -> str:
    """Число с пробелами между тысячами и запятой: 1 234 567 или 0,96."""
    text = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return text


def money(amount: float, currency: str) -> str:
    """Сумма с валютой без сокращений: «480 000 ₸»."""
    return f"{number(round(amount))} {CURRENCY_SIGNS.get(currency, currency)}"


def money_short(amount: float, currency: str) -> str:
    """Короткая запись крупной суммы: «2,1 млн ₸»."""
    sign = CURRENCY_SIGNS.get(currency, currency)
    if abs(amount) >= 1_000_000:
        return f"{number(amount / 1_000_000, 1)} млн {sign}"
    return money(amount, currency)


def percent(share: float) -> str:
    """Доля в процентах без дробной части: «96%»."""
    return f"{share * 100:.0f}%"


def when(moment: pd.Timestamp) -> str:
    """Дата и время: «3 ноября 14:05»."""
    moment = pd.Timestamp(moment)
    return f"{moment.day} {MONTHS[moment.month - 1]} {moment:%H:%M}"


def clock(moment: pd.Timestamp) -> str:
    """Только время: «14:05»."""
    return f"{pd.Timestamp(moment):%H:%M}"


def duration(minutes: float) -> str:
    """Длительность: «12 мин», «2 ч 40 мин», «1 д 3 ч»."""
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "—"
    total = int(round(minutes))
    if total < 60:
        return f"{total} мин"
    hours, rest = divmod(total, 60)
    if hours < 24:
        return f"{hours} ч {rest} мин" if rest else f"{hours} ч"
    days, hours = divmod(hours, 24)
    return f"{days} д {hours} ч" if hours else f"{days} д"


def plural(count: int, one: str, few: str, many: str) -> str:
    """Число со словом в нужной форме: «1 счёт», «4 счёта», «10 счетов»."""
    tail, tens = count % 10, count % 100
    word = one if tail == 1 and tens != 11 else few if 2 <= tail <= 4 and not 12 <= tens <= 14 else many
    return f"{count} {word}"


def hours(value: float) -> str:
    """Окно в часах: «72 ч»."""
    return f"{value:.0f} ч"
