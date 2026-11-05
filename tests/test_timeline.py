"""Таймлайн движения денег строится из шагов trace_money без пересчёта чисел."""
from __future__ import annotations

from bankrag.tools import functions as F
from bankrag.tools.data import CaseData
from bankrag.tools.timeline import CASH_ROW, timeline_chart, timeline_rows


def test_rows_follow_trace_steps(data: CaseData) -> None:
    result = F.trace_money(data, "KZ-1001")
    rows, order = timeline_rows(result["steps"])
    assert len(rows) == len(result["steps"]) and rows["minute"].iloc[0] == 0
    assert order[0] == result["steps"][0]["from"]
    later = rows.dropna(subset=["held_from"])
    assert (later["minute"] - later["held_from"]).round().tolist() == [
        s["delay_minutes"] for s in result["steps"][1:]]
    if any(s["cash"] for s in result["steps"]):
        assert CASH_ROW in order


def test_chart_needs_two_steps() -> None:
    assert timeline_chart([]) is None
    assert timeline_chart([{"step": 1, "from": "A", "to": "B", "time": "2026-11-03 14:05", "amount_text": "1 ₸",
                            "time_text": "3 ноября 14:05", "delay_minutes": None, "cash": False}]) is None
