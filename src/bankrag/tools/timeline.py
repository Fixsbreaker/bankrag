"""Таймлайн движения денег по результату trace_money: сколько минут деньги лежали на каждом счёте.

Граф показывает, кто кому переводил, а таймлайн — как быстро. По оси X — минуты от первого
перевода пострадавшего, по оси Y — счета цепочки в порядке появления. Полоса — время, пока
деньги лежали на счёте до следующего перевода; точка — перевод, ромб — снятие наличных.
Все числа берутся из шагов trace_money, здесь ничего не пересчитывается.
"""
from __future__ import annotations

import pandas as pd

CASH_ROW = "Наличные (банкомат)"


def timeline_rows(steps: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Строки для графика и порядок счетов сверху вниз."""
    if not steps:
        return pd.DataFrame(), []
    start = pd.Timestamp(steps[0]["time"])
    order: list[str] = []
    rows = []
    for step in steps:
        minute = (pd.Timestamp(step["time"]) - start).total_seconds() / 60
        target = CASH_ROW if step.get("cash") else step["to"]
        for account in (step["from"], target):
            if account not in order:
                order.append(account)
        held = step.get("delay_minutes")
        rows.append({"step": step["step"], "from": step["from"], "to": target, "minute": round(minute, 1),
                     "held_from": round(minute - held, 1) if held is not None else None,
                     "amount_text": step["amount_text"], "time_text": step["time_text"],
                     "delay_text": step.get("delay_text") or "—", "cash": bool(step.get("cash"))})
    return pd.DataFrame(rows), order


def timeline_chart(steps: list[dict]):
    """График Altair (входит в Streamlit). None, если шагов меньше двух."""
    import altair as alt

    rows, order = timeline_rows(steps)
    if len(rows) < 2:
        return None
    # Порядок счетов задаётся явно: иначе слои графика сортируют их по алфавиту.
    scale = alt.Scale(domain=order)
    y = alt.Y("account:N", scale=scale, sort=None, title=None)
    tooltip = [alt.Tooltip("step:Q", title="Шаг"), alt.Tooltip("time_text:N", title="Время"),
               alt.Tooltip("from:N", title="Откуда"), alt.Tooltip("to:N", title="Куда"),
               alt.Tooltip("amount_text:N", title="Сумма"), alt.Tooltip("delay_text:N", title="Лежали на счёте")]
    held = rows.dropna(subset=["held_from"]).assign(account=lambda f: f["from"])
    bars = alt.Chart(held).mark_bar(height=14, color="#c77700", opacity=0.55).encode(
        x=alt.X("held_from:Q", title="минут от первого перевода пострадавшего"), x2="minute:Q", y=y, tooltip=tooltip)
    # Точка у пострадавшего — момент первого перевода, у остальных счетов — поступление денег.
    start = rows.iloc[[0]].assign(account=rows["from"].iloc[0])
    arrivals = pd.concat([start, rows.assign(account=rows["to"])], ignore_index=True)
    points = alt.Chart(arrivals).mark_point(filled=True, size=90).encode(
        x="minute:Q", y=y, tooltip=tooltip,
        shape=alt.Shape("cash:N", scale=alt.Scale(domain=[False, True], range=["circle", "diamond"]), legend=None),
        color=alt.Color("cash:N", scale=alt.Scale(domain=[False, True], range=["#1f5fa8", "#b3261e"]), legend=None))
    links = alt.Chart(rows.dropna(subset=["held_from"])).mark_rule(color="#9a9ea6", strokeDash=[3, 3]).encode(
        x="minute:Q", y=alt.Y("from:N", scale=scale, sort=None), y2="to:N")
    return (bars + links + points).properties(height=max(160, 34 * len(order)))
