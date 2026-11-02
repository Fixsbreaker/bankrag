"""Флажки подозрительности и цвет счёта на графе.

Флажок — простое правило, понятное без объяснений: «пришло на карту, и в ближайшие
3 дня сняли наличными или отправили такую же сумму дальше». Пороги лежат в config.yaml.

Каждый исходящий перевод привязывается к последнему поступлению на счёт до него
(в пределах окна). Так видно, через сколько минут и какую долю поступления счёт отправил дальше.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAG_TRANSIT = "transit"
FLAG_CASH = "cash_out"
FLAG_FAST = "fast_forward"
FLAG_SENDERS = "many_senders"
FLAG_SPLIT = "split"
FLAG_LINKED = "linked_to_confirmed"
FLAGS = [FLAG_TRANSIT, FLAG_CASH, FLAG_FAST, FLAG_SENDERS, FLAG_SPLIT, FLAG_LINKED]
FLAG_TITLES = {
    FLAG_TRANSIT: "транзит", FLAG_CASH: "вывод наличными", FLAG_FAST: "быстрая пересылка",
    FLAG_SENDERS: "много отправителей", FLAG_SPLIT: "дробление", FLAG_LINKED: "связь с подтверждённым дропом",
}

RED, ORANGE, GREY, GREEN, BLUE = "red", "orange", "grey", "green", "blue"
# Цвета различаются не только оттенком, но и светлотой: на графе и в чёрно-белой распечатке
# справки уровни видно и без цвета. Рядом с цветом всегда идёт подпись из COLOR_TITLES.
COLOR_HEX = {RED: "#b3261e", ORANGE: "#c77700", GREY: "#9a9ea6", GREEN: "#3e8e7e", BLUE: "#1f5fa8"}
COLOR_TITLES = {RED: "подозрительно", ORANGE: "стоит проверить", GREY: "нейтрально, данных мало",
                GREEN: "чисто", BLUE: "пострадавший (есть жалоба)"}


def pair_outflows(edges: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    """Исходящие переводы с последним поступлением на тот же счёт за window_hours до них.

    В edges нужны столбцы timestamp, from_account, to_account, amount, is_self, is_withdrawal.
    """
    between = edges[~edges["is_self"]]
    inflows = between[~between["is_withdrawal"]]
    incoming = pd.DataFrame({"account_id": inflows["to_account"].astype(str), "in_time": inflows["timestamp"],
                             "in_amount": inflows["amount"].astype(float), "in_id": inflows.index,
                             "sender": inflows["from_account"].astype(str)}).sort_values("in_time")
    outgoing = pd.DataFrame({"account_id": between["from_account"].astype(str), "out_time": between["timestamp"],
                             "out_amount": between["amount"].astype(float), "receiver": between["to_account"].astype(str),
                             "is_withdrawal": between["is_withdrawal"].astype(bool)}).sort_values("out_time")
    paired = pd.merge_asof(outgoing, incoming, left_on="out_time", right_on="in_time", by="account_id",
                           direction="backward", tolerance=pd.Timedelta(hours=window_hours))
    paired = paired.dropna(subset=["in_id"])
    paired["delay_minutes"] = (paired["out_time"] - paired["in_time"]).dt.total_seconds() / 60
    return paired


def _max_senders(inflows: pd.DataFrame, hours: float, threshold: int) -> pd.Series:
    """Наибольшее число разных отправителей за hours часов, начиная с любого поступления."""
    result = {}
    span = np.timedelta64(int(hours * 3600), "s")
    for account, group in inflows.groupby("account_id", sort=False):
        if group["sender"].nunique() < threshold:
            result[account] = group["sender"].nunique() if len(group) else 0
            continue
        times = group["in_time"].to_numpy()
        senders = group["sender"].to_numpy()
        ends = np.searchsorted(times, times + span, side="left")
        # У счетов с тысячами поступлений хватает части стартовых точек.
        step = max(1, len(times) // 300)
        result[account] = max(len(set(senders[i:ends[i]])) for i in range(0, len(times), step))
    return pd.Series(result, dtype="float64")


def compute_flags(edges: pd.DataFrame, settings: dict, businesses: set[str] | None = None,
                  confirmed: set[str] | None = None) -> pd.DataFrame:
    """Флажки и числа для фраз по каждому счёту из edges."""
    businesses = businesses or set()
    confirmed = confirmed or set()
    between = edges[~edges["is_self"]]
    accounts = pd.Index(pd.concat([between["from_account"], between["to_account"]]).astype(str).unique(),
                        name="account_id")
    table = pd.DataFrame(index=accounts)
    inflows = between[~between["is_withdrawal"]]
    table["received"] = inflows.groupby(inflows["to_account"].astype(str))["amount"].sum()
    table["sent"] = between.groupby(between["from_account"].astype(str))["amount"].sum()
    table["transfers"] = pd.concat([between["from_account"], between["to_account"]]).astype(str).value_counts()
    table[["received", "sent", "transfers"]] = table[["received", "sent", "transfers"]].fillna(0)

    paired = pair_outflows(edges, settings["window_hours"])
    # Покупки в магазинах — не пересылка денег дальше.
    paired = paired[~paired["receiver"].isin(businesses) | paired["is_withdrawal"]]
    # Дальше не может уйти больше, чем пришло: остальное — деньги, которые были на счёте раньше.
    table["forwarded"] = paired.groupby("account_id")["out_amount"].sum()
    table["forwarded"] = table["forwarded"].clip(upper=table["received"])
    table["forwarded"] = table["forwarded"].fillna(0)
    table["transit_share"] = (table["forwarded"] / table["received"]).where(table["received"] > 0)

    cash = paired[paired["is_withdrawal"] & (paired["out_amount"] >= settings["cash_min_share"] * paired["in_amount"])]
    table["cash_amount"] = cash.groupby("account_id")["out_amount"].sum()
    table["cash_delay_minutes"] = cash.groupby("account_id")["delay_minutes"].min()

    # Быстрая пересылка: через сколько минут после поступления ушла его заметная доля, даже частями.
    transfers_out = paired[~paired["is_withdrawal"]].sort_values("delay_minutes")
    cumulative = transfers_out.groupby(["account_id", "in_id"])["out_amount"].cumsum()
    reached = transfers_out[cumulative >= settings["min_forward_share"] * transfers_out["in_amount"]]
    table["fastest_minutes"] = reached.groupby("account_id")["delay_minutes"].min()

    parts = transfers_out.groupby(["account_id", "in_id"]).agg(
        parts=("receiver", "nunique"), total=("out_amount", "sum"), largest=("out_amount", "max"),
        smallest=("out_amount", "min"), in_amount=("in_amount", "first"))
    splits = parts[(parts["parts"] >= settings["split_min_parts"])
                   & (parts["total"] >= settings["split_min_share"] * parts["in_amount"])
                   & (parts["largest"] <= settings["split_max_ratio"] * parts["smallest"])].reset_index()
    first_split = splits.sort_values("total", ascending=False).drop_duplicates("account_id").set_index("account_id")
    table["split_parts"] = first_split["parts"]
    table["split_total"] = first_split["total"]
    table["split_count"] = splits.groupby("account_id").size()

    incoming = pd.DataFrame({"account_id": inflows["to_account"].astype(str), "in_time": inflows["timestamp"],
                             "sender": inflows["from_account"].astype(str)}).sort_values("in_time")
    table["max_senders_day"] = _max_senders(incoming, settings["many_senders_hours"], settings["many_senders"])
    table["max_senders_day"] = table["max_senders_day"].fillna(0)

    pairs = pd.concat([between[["from_account", "to_account"]].set_axis(["account_id", "other"], axis=1),
                       between[["to_account", "from_account"]].set_axis(["account_id", "other"], axis=1)]).astype(str)
    linked = pairs[pairs["other"].isin(confirmed) & (pairs["account_id"] != pairs["other"])].drop_duplicates()
    table["linked_to"] = linked.groupby("account_id")["other"].agg(sorted)

    table[FLAG_TRANSIT] = table["transit_share"].fillna(0) >= settings["transit_share"]
    table[FLAG_CASH] = table["cash_amount"].notna()
    table[FLAG_FAST] = table["fastest_minutes"].fillna(np.inf) <= settings["fast_forward_minutes"]
    table[FLAG_SENDERS] = (table["max_senders_day"] >= settings["many_senders"]) & ~table.index.isin(businesses)
    table[FLAG_SPLIT] = table["split_parts"].notna()
    table[FLAG_LINKED] = table["linked_to"].notna()
    table["flag_count"] = table[FLAGS].sum(axis=1).astype(int)
    return table


def account_color(flag_count: int, risk: float | None, is_victim: bool, transfers: float, high_risk: float,
                  medium_risk: float, green_min_transfers: int) -> str:
    """Цвет счёта: пострадавший, подозрительный, стоит проверить, чистый или нейтральный."""
    risk = 0.0 if risk is None or pd.isna(risk) else float(risk)
    if is_victim:
        return BLUE
    if flag_count >= 2 or risk >= high_risk:
        return RED
    if flag_count == 1 or risk >= medium_risk:
        return ORANGE
    if transfers >= green_min_transfers:
        return GREEN
    return GREY
