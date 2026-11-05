"""Функции, которые вызывает чат. Все числа считает этот код, модель только пересказывает.

Каждая функция возвращает словарь для JSON: сырые числа, готовые строки (*_text) и summary —
ответ по шаблону, который показывается, если модель недоступна или ошиблась в числах.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from bankrag.flags import (
    BLUE,
    COLOR_TITLES,
    FLAG_CASH,
    FLAG_FAST,
    FLAG_LINKED,
    FLAG_SENDERS,
    FLAG_SPLIT,
    FLAG_TITLES,
    FLAG_TRANSIT,
    FLAGS,
    GREEN,
    ORANGE,
    RED,
)
from bankrag.tools import fmt
from bankrag.tools.data import KIND_TITLES, CaseData, parse_amount

CASH_NODE = "Наличные"
NOT_FOUND = "Счёт {account} не найден в данных."


def _not_found(account: str) -> dict:
    text = NOT_FOUND.format(account=account)
    return {"error": text, "summary": text, "accounts": []}


def flag_phrases(data: CaseData, account: str) -> list[str]:
    """Сработавшие флажки счёта понятными фразами с числами."""
    if account not in data.flags.index:
        return []
    row = data.flags.loc[account]
    money = lambda value: fmt.money(value, data.currency)  # noqa: E731
    phrases = []
    if row[FLAG_TRANSIT]:
        phrases.append(f"транзит: ушло дальше {fmt.percent(row['transit_share'])} полученного "
                       f"({money(row['forwarded'])} из {money(row['received'])})")
    if row[FLAG_CASH]:
        phrases.append(f"вывод наличными: снято {money(row['cash_amount'])}, "
                       f"быстрее всего через {fmt.duration(row['cash_delay_minutes'])} после поступления")
    if row[FLAG_FAST]:
        phrases.append(f"быстрая пересылка: деньги ушли дальше через {fmt.duration(row['fastest_minutes'])}")
    if row[FLAG_SENDERS]:
        phrases.append(f"много отправителей: {int(row['max_senders_day'])} разных за сутки")
    if row[FLAG_SPLIT]:
        phrases.append(f"дробление: {money(row['split_total'])} разослано на {int(row['split_parts'])} счёта "
                       f"примерно равными частями")
    if row[FLAG_LINKED]:
        phrases.append("связь с подтверждённым дропом: " + ", ".join(row["linked_to"]))
    return phrases


def _hit_phrase(hit: dict) -> str:
    added = f", включён {fmt.date(hit['added_to_list'])}" if hit["added_to_list"] else ""
    if hit.get("strict", True):
        return f"владелец в перечне ({hit['list_short']}): «{hit['listed_name']}»{added}; совпадение по: {hit['match']}"
    birth = f", дата рождения в перечне {fmt.date(hit['listed_birth_date'])}" if hit.get("listed_birth_date") else ""
    return f"владелец похож на запись перечня ({hit['list_short']}): «{hit['listed_name']}»{birth}{added}; {hit['match']}"


def list_phrases(data: CaseData, account: str) -> list[str]:
    """Совпадения владельца счёта с перечнями АФМ понятными фразами: точные и похожие."""
    return [_hit_phrase(h) for h in data.list_hits.get(account, []) + data.list_similar.get(account, [])]


def _list_entries(data: CaseData) -> dict[str, list[dict]]:
    """Все счета с совпадениями: точные и похожие."""
    return {**data.list_similar, **data.list_hits}


def _lists_meta(data: CaseData) -> dict:
    """Какие перечни загружены и когда: без этого «не найден в перечнях» ничего не значит."""
    watchlist = data.watchlist
    loaded = watchlist.downloaded_at
    return {"list_records": watchlist.real_count, "list_records_text": fmt.number(watchlist.real_count),
            "lists_downloaded_text": fmt.date(loaded) if loaded else "перечни не загружены",
            "lists_source": "api.websfm.kz (АФМ РК)"}


def _listed_transfers(data: CaseData, listed: str, other: str | None = None) -> dict:
    """Переводы на счёт и со счёта лица из перечня; other — только с этим контрагентом."""
    edges = data.edges[~data.edges["is_self"]]
    incoming = edges[edges["to_account"] == listed]
    outgoing = edges[edges["from_account"] == listed]
    if other is not None:
        incoming = incoming[incoming["from_account"] == other]
        outgoing = outgoing[outgoing["to_account"] == other]
    money = lambda value: fmt.money(value, data.currency)  # noqa: E731
    last = pd.concat([incoming["timestamp"], outgoing["timestamp"]]).max()
    return {"account": listed, "incoming_count": len(incoming), "incoming_total": float(incoming["amount"].sum()),
            "incoming_total_text": money(incoming["amount"].sum()), "incoming_senders": int(incoming["from_account"].nunique()),
            "outgoing_count": len(outgoing), "outgoing_total": float(outgoing["amount"].sum()),
            "outgoing_total_text": money(outgoing["amount"].sum()),
            "last_time_text": fmt.when(last) if pd.notna(last) else "—"}


def check_lists(data: CaseData, account: str | None = None, name: str | None = None) -> dict:
    """Проверка по перечням АФМ: владелец счёта, его контрагенты, ФИО/название или ИИН/БИН."""
    meta = _lists_meta(data)
    source = (f"Сверка с перечнями АФМ РК ({meta['list_records_text']} записей, загружены "
              f"{meta['lists_downloaded_text']}) и тестовой записью демо.")
    if name and not account:
        hits = [h.as_dict() for h in data.watchlist.lookup(name)]
        exact = [h for h in hits if h["strict"]]
        similar = [h for h in hits if not h["strict"]]
        parts = []
        if exact:
            parts.append(f"По запросу «{name}» найдено в перечнях: " + "; ".join(
                f"«{h['listed_name']}» ({h['list_short']}, совпадение по: {h['match']})" for h in exact) + ".")
        if similar:
            parts.append(("Похожее написание" if exact else f"Точных совпадений по запросу «{name}» нет, похожее написание")
                         + ", проверить вручную: " + "; ".join(
                             f"«{h['listed_name']}» ({h['list_short']}, сходство {h['similarity']}%)" for h in similar) + ".")
        summary = " ".join(parts) if parts else f"По запросу «{name}» в перечнях ничего не найдено."
        summary += f" {source}"
        return {**meta, "query": name, "hits": hits, "hit_count": len(exact), "similar_count": len(similar),
                "summary": summary, "accounts": []}

    if not account:
        # Без счёта — обзор: какие счета в данных принадлежат лицам из перечней.
        def overview_row(account: str, hits: list[dict]) -> dict:
            return {**_account_line(data, account), "owner": (data.owner(account) or {}).get("name"), "hits": hits,
                    **_listed_transfers(data, account)}

        rows = [overview_row(a, data.list_hits[a]) for a in sorted(data.list_hits)]
        similar = [overview_row(a, data.list_similar[a]) for a in sorted(data.list_similar)]
        if not data.owners.size:
            summary = f"В наборе «{data.title}» нет данных о владельцах счетов, сверить их с перечнями нельзя. {source}"
        elif not rows and not similar:
            summary = f"Ни один владелец счёта в данных не найден в перечнях. {source}"
        else:
            parts = []
            if rows:
                parts.append("Счета лиц из перечней: " + "; ".join(
                    f"{r['account']} — «{r['owner']}» ({r['hits'][0]['list_short']}), входящих переводов "
                    f"{r['incoming_count']} от {r['incoming_senders']} отправителей на {r['incoming_total_text']}"
                    for r in rows) + ".")
            else:
                parts.append("Точных совпадений с перечнями нет.")
            if similar:
                parts.append("Похожее написание, проверить вручную: " + "; ".join(
                    f"{r['account']} — «{r['owner']}» похож на «{r['hits'][0]['listed_name']}» "
                    f"({r['hits'][0]['list_short']}, сходство {r['hits'][0]['similarity']}%)" for r in similar) + ".")
            summary = " ".join(parts) + f" {source}"
        return {**meta, "listed": rows, "listed_count": len(rows), "similar": similar, "similar_count": len(similar),
                "summary": summary, "accounts": [r["account"] for r in rows + similar]}

    account = data.normalize_account(account)
    if not data.has(account):
        return _not_found(account)
    owner = data.owner(account)
    hits = data.list_hits.get(account, [])
    edges = data.edges[~data.edges["is_self"]]
    others = set(edges.loc[edges["from_account"] == account, "to_account"]) | \
        set(edges.loc[edges["to_account"] == account, "from_account"])
    entries = _list_entries(data)
    counterparties = [{**_account_line(data, other), "owner": (data.owner(other) or {}).get("name"),
                       "hits": entries[other], "strict": other in data.list_hits,
                       **_listed_transfers(data, other, account)}
                      for other in sorted(others) if other in entries]
    if owner is None:
        summary = f"Для {account} нет данных о владельце: в наборе «{data.title}» владельцев нет. {source}"
        return {**meta, "account": account, "owner": None, "listed": False, "hits": [], "counterparties": [],
                "summary": summary, "accounts": [account]}
    if hits:
        transfers = _listed_transfers(data, account)
        summary = (f"Владелец {account} — «{owner['name']}» — найден в перечне: " + "; ".join(list_phrases(data, account))
                   + f". Входящих переводов {transfers['incoming_count']} от {transfers['incoming_senders']} "
                   f"отправителей на {transfers['incoming_total_text']}, исходящих {transfers['outgoing_count']} на "
                   f"{transfers['outgoing_total_text']}. Операции с ним нужно передать на проверку. {source}")
        return {**meta, "account": account, "owner": owner, "listed": True, "hits": hits, **transfers,
                "counterparties": counterparties, "summary": summary, "accounts": [account]}
    similar = data.list_similar.get(account, [])
    if similar:
        summary = (f"Владелец {account} — «{owner['name']}» — точно в перечнях не найден, но "
                   + "; ".join(list_phrases(data, account)) + ". Это может быть другой человек: сверьте документы.")
    else:
        summary = f"Владелец {account} — «{owner['name']}» — в перечнях не найден."
    if counterparties:
        summary += " Но среди контрагентов есть лица из перечня или похожие на них: " + "; ".join(
            f"{c['account']} — «{c['owner']}» ({'в перечне' if c['strict'] else 'похож, проверить вручную'}), "
            f"переводов от {account}: {c['incoming_count']} на "
            f"{c['incoming_total_text']}, к {account}: {c['outgoing_count']}" for c in counterparties) + "."
    elif owner is not None:
        summary += " Среди контрагентов лиц из перечней нет."
    summary += f" {source}"
    return {**meta, "account": account, "owner": owner, "listed": False, "hits": [], "similar": similar,
            "list_similar": bool(similar), "counterparties": counterparties,
            "counterparty_count": len(counterparties), "summary": summary,
            "accounts": [account, *[c["account"] for c in counterparties]]}


def _node(data: CaseData, account: str) -> dict:
    if account == CASH_NODE:
        return {"id": CASH_NODE, "label": "Наличные (банкомат)", "color": "cash", "flags": 0, "confirmed": False}
    row = data.flags.loc[account] if account in data.flags.index else None
    return {"id": account, "label": account, "color": data.color(account),
            "flags": int(row["flag_count"]) if row is not None else 0, "confirmed": account in data.confirmed,
            "kind": data.kind(account)}


def _account_line(data: CaseData, account: str) -> dict:
    color = data.color(account)
    return {"account": account, "color": color, "color_text": COLOR_TITLES[color],
            "flags": int(data.flags.at[account, "flag_count"]) if account in data.flags.index else 0}


def trace_money(data: CaseData, account: str, start_time: Any = None, amount: Any = None, window_hours: float = 72,
                max_depth: int = 4) -> dict:
    """Идёт по ходу денег от поступления на счёт: куда, сколько, через сколько и где сняли наличными."""
    account = data.normalize_account(account)
    if not data.has(account):
        return _not_found(account)
    settings = data.config["chat"]
    max_depth = max(1, min(int(max_depth or settings["trace_max_depth"]), 6))
    window = pd.Timedelta(hours=float(window_hours or settings["trace_window_hours"]))
    edges = data.edges[~data.edges["is_self"]]
    inflows = edges[(edges["to_account"] == account) & ~edges["is_withdrawal"]]
    if inflows.empty:
        text = f"На счёт {account} не было поступлений, прослеживать нечего."
        return {"error": text, "summary": text, "accounts": [account]}

    when, wanted = data.parse_time(start_time), parse_amount(amount)
    candidates = inflows
    if when is not None:
        near = inflows[(inflows["timestamp"] - when).abs() <= pd.Timedelta(hours=1)]
        candidates = near if len(near) else candidates.iloc[0:0]
    if wanted is not None and len(candidates):
        candidates = candidates.loc[[(candidates["amount"] - wanted).abs().idxmin()]]
    if when is not None and candidates.empty:
        text = f"Поступления на {account} около {fmt.when(when)} в данных нет."
        return {"error": text, "summary": text, "accounts": [account]}
    if when is None and wanted is None:
        complained = candidates[candidates["from_account"].isin(data.victims)]
        candidates = complained if len(complained) else candidates
        candidates = candidates.loc[[candidates["amount"].idxmax()]]
    start = candidates.iloc[0]

    money = lambda value: fmt.money(value, data.currency)  # noqa: E731
    businesses = data.businesses
    steps = [{"step": 1, "from": str(start["from_account"]), "to": account, "amount": float(start["amount"]),
              "amount_text": money(start["amount"]), "time": str(start["timestamp"]),
              "time_text": fmt.when(start["timestamp"]), "delay_minutes": None, "share": None, "cash": False,
              "depth": 0}]
    queue = deque([(account, start["timestamp"], float(start["amount"]), 0, {account})])
    chain, cash_total, retained, last_time = [account], 0.0, 0.0, start["timestamp"]
    while queue:
        holder, arrived, received, depth, path = queue.popleft()
        out = edges[(edges["from_account"] == holder) & (edges["timestamp"] >= arrived)
                    & (edges["timestamp"] <= arrived + window)
                    & (edges["is_withdrawal"] | ~edges["to_account"].isin(businesses))]
        if depth >= max_depth:
            out = out.iloc[0:0]
        taken = 0.0
        for row in out.head(50).itertuples(index=False):
            if taken >= settings["trace_cover_share"] * received or \
                    sum(1 for s in steps if s["from"] == holder and s["depth"] == depth + 1) >= settings["trace_max_branches"]:
                break
            target = CASH_NODE if row.is_withdrawal else str(row.to_account)
            if target in path:
                continue
            delay = (row.timestamp - arrived).total_seconds() / 60
            steps.append({"step": len(steps) + 1, "from": holder, "to": target, "amount": float(row.amount),
                          "amount_text": money(row.amount), "time": str(row.timestamp),
                          "time_text": fmt.when(row.timestamp), "delay_minutes": round(delay),
                          "delay_text": fmt.duration(delay), "share": round(row.amount / received, 2),
                          "share_text": fmt.percent(row.amount / received), "cash": bool(row.is_withdrawal),
                          "depth": depth + 1})
            taken += float(row.amount)
            last_time = max(last_time, row.timestamp)
            if row.is_withdrawal:
                cash_total += float(row.amount)
            else:
                if target not in chain:
                    chain.append(target)
                queue.append((target, row.timestamp, float(row.amount), depth + 1, path | {target}))
        retained += max(received - taken, 0.0)

    # Ветки обходятся по очереди, а показывать шаги удобнее по времени.
    steps = [steps[0], *sorted(steps[1:], key=lambda s: (s["time"], s["depth"]))]
    for number, step in enumerate(steps, start=1):
        step["step"] = number
    minutes = (last_time - start["timestamp"]).total_seconds() / 60
    flagged = [a for a in chain if data.color(a) in (RED, ORANGE)]
    lines = []
    for step in steps:
        if step["depth"] == 0:
            lines.append(f"{step['step']}. {step['time_text']} — {step['amount_text']} пришли на {step['to']} от {step['from']}.")
        elif step["cash"]:
            lines.append(f"{step['step']}. {step['time_text']} — с {step['from']} сняли наличными {step['amount_text']} "
                         f"(через {step['delay_text']}).")
        else:
            lines.append(f"{step['step']}. {step['time_text']} — {step['from']} перевёл {step['amount_text']} на "
                         f"{step['to']} (через {step['delay_text']}, {step['share_text']} суммы).")
    summary = (f"За {fmt.duration(minutes)} деньги прошли {fmt.plural(len(steps), 'шаг', 'шага', 'шагов')}:\n"
               + "\n".join(lines)
               + f"\nИтог: в цепочке {fmt.plural(len(chain), 'счёт', 'счёта', 'счетов')}, "
               f"по флажкам подозрительны {len(flagged)}. "
               f"Снято наличными {money(cash_total)}. На счетах цепочки осталось {money(retained)}.")
    nodes = [_node(data, a) for a in [steps[0]["from"], *chain]] + ([_node(data, CASH_NODE)] if cash_total else [])
    graph_edges = [{"from": s["from"], "to": s["to"], "label": f"{s['amount_text']} · {fmt.clock(s['time'])}"}
                   for s in steps]
    return {"account": account, "start_from": steps[0]["from"], "start_time_text": steps[0]["time_text"],
            "start_amount": steps[0]["amount"], "start_amount_text": steps[0]["amount_text"], "steps": steps,
            "step_count": len(steps), "chain": [_account_line(data, a) for a in chain], "chain_count": len(chain),
            "suspicious_count": len(flagged), "cash_total": cash_total, "cash_text": money(cash_total),
            "retained": retained, "retained_text": money(retained), "duration_minutes": round(minutes),
            "duration_text": fmt.duration(minutes), "summary": summary, "accounts": [steps[0]["from"], *chain],
            "graph": {"nodes": nodes, "edges": graph_edges, "title": f"Путь денег от {steps[0]['from']}"}}


def trace_sources(data: CaseData, account: str, window_hours: float = 72) -> dict:
    """Кто переводил деньги на счёт за последние window_hours часов и есть ли у них жалобы."""
    account = data.normalize_account(account)
    if not data.has(account):
        return _not_found(account)
    hours = float(window_hours or 72)
    edges = data.edges
    since = data.end - pd.Timedelta(hours=hours)
    inflows = edges[(edges["to_account"] == account) & ~edges["is_self"] & (edges["timestamp"] >= since)]
    money = lambda value: fmt.money(value, data.currency)  # noqa: E731
    if inflows.empty:
        text = f"За последние {fmt.hours(hours)} на {account} переводов не было."
        return {"account": account, "transfer_count": 0, "sender_count": 0, "summary": text, "accounts": [account]}
    complained = set(data.complaints.loc[data.complaints["target_account"] == account, "account_id"]) \
        if len(data.complaints) else set()
    grouped = inflows.groupby("from_account").agg(transfers=("amount", "size"), total=("amount", "sum"),
                                                  first=("timestamp", "min")).sort_values("first")
    senders = [{"account": str(sender), "transfers": int(row.transfers), "total": float(row.total),
                "total_text": money(row.total), "first_time_text": fmt.when(row.first),
                "complaint": str(sender) in complained or str(sender) in data.victims, "color": data.color(str(sender))}
               for sender, row in grouped.iterrows()]
    with_complaints = [s for s in senders if s["complaint"]]
    total = float(inflows["amount"].sum())
    summary = (f"За последние {fmt.hours(hours)} на {account} пришло "
               f"{fmt.plural(len(inflows), 'перевод', 'перевода', 'переводов')} от {len(senders)} разных "
               f"отправителей на {fmt.money_short(total, data.currency)}. "
               f"Жалобы о мошенничестве есть у {len(with_complaints)} из них")
    summary += (": " + ", ".join(s["account"] for s in with_complaints) + "." if with_complaints else ".")
    if len(with_complaints) > 1:
        summary += " Похоже, пострадавших больше одного."
    nodes = [_node(data, account)] + [_node(data, s["account"]) for s in senders[:30]]
    graph_edges = [{"from": s["account"], "to": account, "label": f"{s['total_text']} · {s['first_time_text']}"}
                   for s in senders[:30]]
    return {"account": account, "window_hours": hours, "transfer_count": len(inflows), "sender_count": len(senders),
            "total": total, "total_text": money(total), "total_short_text": fmt.money_short(total, data.currency),
            "complaint_count": len(with_complaints), "senders": senders[:30], "truncated": len(senders) > 30,
            "summary": summary, "accounts": [account, *[s["account"] for s in senders[:30]]],
            "graph": {"nodes": nodes, "edges": graph_edges, "title": f"Кто переводил на {account}"}}


def _history_text(data: CaseData, account: str) -> str:
    """Обычные операции счёта: зарплата, покупки, снятие наличных."""
    edges = data.edges
    salary = int(((edges["to_account"] == account) & edges["from_account"].map(data.kinds).eq("employer")).sum())
    purchases = int(((edges["from_account"] == account) & edges["to_account"].map(data.kinds).eq("merchant")).sum())
    if not salary and not purchases:
        return "обычных операций (зарплата, покупки) нет"
    return f"обычные операции: зарплата {salary}, покупки {purchases}"


HIDDEN_FEATURES_TEXT = "особенности данных IBM (валюты, форматы платежей, суммы, переводы себе, связи соседей)"


def model_factors(data: CaseData, account: str) -> dict:
    """Вклад признаков в оценку модели понятными фразами с долями (pred_contrib LightGBM)."""
    explained = data.model_explanation(account)
    if explained is None:
        return {"model_up": [], "model_down": [], "model_factors_text": ""}

    def side(name: str) -> tuple[list[dict], str]:
        part = explained[name]
        # Вклады меньше 1% не показываются: «0%» рядом с причиной только путает.
        factors = [{**f, "share_text": fmt.percent(f["share"])} for f in part["factors"] if f["share"] >= 0.01]
        pieces = [f"{f['text']} ({f['share_text']})" for f in factors]
        if part["hidden"] >= 0.005:
            pieces.append(f"{HIDDEN_FEATURES_TEXT} ({fmt.percent(part['hidden'])})")
        return factors, "; ".join(pieces)

    up, up_text = side("up")
    down, down_text = side("down")
    parts = []
    if up_text:
        parts.append(f"повысили оценку: {up_text}")
    if down_text:
        parts.append(f"понизили: {down_text}")
    return {"model_up": up, "model_down": down, "model_up_hidden_text": fmt.percent(explained["up"]["hidden"]),
            "model_down_hidden_text": fmt.percent(explained["down"]["hidden"]),
            "model_factors_text": ("Вклад признаков в оценку модели (доля среди повышающих или понижающих): "
                                   + "; ".join(parts) + ".") if parts else ""}


def account_profile(data: CaseData, account: str) -> dict:
    """Профиль счёта: цвет, флажки, оценка модели, основные числа."""
    account = data.normalize_account(account)
    if not data.has(account):
        return _not_found(account)
    row = data.flags.loc[account]
    money = lambda value: fmt.money(value, data.currency)  # noqa: E731
    risk = data.risk(account)
    phrases = flag_phrases(data, account)
    color = data.color(account)
    kind = data.kind(account)
    complaints = data.complaints[data.complaints["account_id"] == account] if len(data.complaints) else pd.DataFrame()
    reasons = data.model_reasons(account)
    risk_text = "нет оценки" if risk is None else fmt.number(risk, 3)
    level = ("нет оценки" if risk is None else "высокий" if risk >= data.high_risk
             else "средний" if risk >= data.high_risk * data.config["colors"]["medium_risk_share"] else "низкий")
    listed = list_phrases(data, account)
    summary = f"{account} ({KIND_TITLES.get(kind, kind)}): {COLOR_TITLES[color]}, флажков {len(phrases)}."
    if listed:
        summary += " " + "; ".join(listed).capitalize() + "."
    if phrases:
        summary += " " + "; ".join(phrases) + "."
    if len(complaints):
        summary += f" Подал жалобу: {complaints.iloc[0]['text']}"
    if data.kinds.size:
        summary += f" {_history_text(data, account).capitalize()}."
    summary += f" Оценка модели риска: {risk_text} ({level})."
    factors = model_factors(data, account)
    if factors["model_factors_text"]:
        summary += " " + factors["model_factors_text"]
    return {**factors, "account": account, "kind": KIND_TITLES.get(kind, kind), "color": color, "color_text": COLOR_TITLES[color],
            "flags": [FLAG_TITLES[f] for f in FLAGS if row[f]], "flag_count": len(phrases), "flag_phrases": phrases,
            "risk": risk, "risk_text": risk_text, "risk_level": level, "model_reasons": reasons[:3],
            "received": float(row["received"]), "received_text": money(row["received"]), "sent": float(row["sent"]),
            "sent_text": money(row["sent"]), "transfer_count": int(row["transfers"]),
            "history_text": _history_text(data, account) if data.kinds.size else "",
            "complaint": complaints.iloc[0]["text"] if len(complaints) else None,
            "confirmed": account in data.confirmed, "list_phrases": listed, "listed": data.is_listed(account),
            "list_similar": data.is_list_similar(account),
            "owner": data.owner(account), "summary": summary, "accounts": [account]}


def find_network(data: CaseData, accounts: Any, depth: int = 2) -> dict:
    """Счета, связанные переводами с цепочкой. Раскрываются только подозрительные счета, не магазины."""
    seeds = [data.normalize_account(a) for a in (accounts if isinstance(accounts, list) else str(accounts).split(","))]
    seeds = [a for a in seeds if a]
    missing = [a for a in seeds if not data.has(a)]
    if missing:
        return _not_found(", ".join(missing))
    settings = data.config["chat"]
    limit, hub = settings["network_limit"], settings["hub_degree"]
    depth = max(1, min(int(depth or settings["network_depth"]), 3))
    edges = data.edges[~data.edges["is_self"]]
    businesses = data.businesses
    included = list(dict.fromkeys(seeds))
    frontier, truncated = list(included), False
    for _ in range(depth):
        following = []
        for account in frontier:
            touching = edges[(edges["from_account"] == account) | (edges["to_account"] == account)]
            others = touching["from_account"].where(touching["from_account"] != account, touching["to_account"])
            for other in others.astype(str).unique():
                if other in included or other in businesses:
                    continue
                if len(included) >= limit:
                    truncated = True
                    break
                included.append(other)
                degree = data.flags.at[other, "transfers"] if other in data.flags.index else 0
                if data.color(other) in (RED, ORANGE) and degree <= hub:
                    following.append(other)
        frontier = following
    inside = edges[edges["from_account"].isin(included) & (edges["to_account"].isin(included) | edges["is_withdrawal"])]
    inside = inside.assign(target=inside["to_account"].where(~inside["is_withdrawal"], CASH_NODE))
    grouped = inside.groupby(["from_account", "target"]).agg(count=("amount", "size"), total=("amount", "sum")).reset_index()
    lines = [_account_line(data, a) for a in included]
    counts = {color: sum(1 for line in lines if line["color"] == color) for color in (RED, ORANGE, BLUE, GREEN)}
    summary = (f"Сеть вокруг {', '.join(seeds)}: {fmt.plural(len(included), 'счёт', 'счёта', 'счетов')}, "
               "связанных переводами. "
               f"Подозрительных {counts[RED]}, стоит проверить {counts[ORANGE]}, пострадавших {counts[BLUE]}, "
               f"чистых {counts[GREEN]}.")
    reds = [line["account"] for line in lines if line["color"] == RED]
    if reds:
        summary += " Подозрительные: " + ", ".join(reds[:15]) + "."
    if truncated:
        summary += f" Показаны первые {limit} счетов, сеть больше."
    nodes = [_node(data, a) for a in included] + ([_node(data, CASH_NODE)] if (grouped["target"] == CASH_NODE).any() else [])
    graph_edges = [{"from": r.from_account, "to": r.target,
                    "label": f"{fmt.money(r.total, data.currency)} · переводов {r.count}"} for r in grouped.itertuples()]
    return {"seeds": seeds, "account_count": len(included), "suspicious_count": counts[RED],
            "check_count": counts[ORANGE], "victim_count": counts[BLUE], "clean_count": counts[GREEN],
            "accounts_detail": lines, "truncated": truncated, "summary": summary, "accounts": included,
            "graph": {"nodes": nodes, "edges": graph_edges, "title": f"Сеть вокруг {', '.join(seeds)}"}}


def find_similar(data: CaseData, account: str, k: int = 10) -> dict:
    """Счета с похожим поведением (FAISS). Они не обязательно связаны переводами с этим счётом."""
    account = data.normalize_account(account)
    if not data.has(account):
        return _not_found(account)
    if data.similarity is None or account not in data.similarity.positions:
        text = f"Для счёта {account} нет вектора поведения: за окно признаков у него нет переводов."
        return {"error": text, "summary": text, "accounts": [account]}
    k = max(1, min(int(k or 10), 20))
    found = data.similarity.find_similar(account, k=k).neighbors
    edges = data.edges
    linked = set(edges.loc[edges["from_account"] == account, "to_account"]) | \
        set(edges.loc[edges["to_account"] == account, "from_account"])
    rows = [{**_account_line(data, str(r.account_id)), "similarity": round(float(r.distance), 3),
             "similarity_text": fmt.number(float(r.distance), 3), "linked_by_transfers": str(r.account_id) in linked}
            for r in found.itertuples(index=False)]
    unlinked = [r for r in rows if not r["linked_by_transfers"]]
    summary = (f"Нашёл {fmt.plural(len(rows), 'счёт', 'счёта', 'счетов')} с похожим почерком, как у {account}. Подозрительных среди них "
               f"{sum(r['color'] == RED for r in rows)}. {len(unlinked)} из них не связаны переводами с {account}: "
               "это могут быть другие сети с такой же схемой.")
    return {"account": account, "similar": rows, "similar_count": len(rows), "unlinked_count": len(unlinked),
            "note": "похожий почерк, переводами не связаны", "summary": summary,
            "accounts": [account, *[r["account"] for r in rows]]}


def list_suspicious(data: CaseData, period: Any = None, limit: int = 10) -> dict:
    """Самые подозрительные счета за день или за весь период: сначала по флажкам, затем по оценке модели."""
    limit = max(1, min(int(limit or 10), 50))
    edges = data.edges
    day, period_text = None, "весь период данных"
    text = str(period or "").strip().lower()
    last_day = data.end.normalize()
    if text in ("сегодня", "today"):
        day = last_day
    elif text in ("вчера", "yesterday"):
        day = last_day - pd.Timedelta(days=1)
    elif text:
        parsed = data.parse_time(text + (" 00:00" if ":" not in text else ""))
        if parsed is None:
            try:
                parsed = pd.Timestamp(text)
            except ValueError:
                parsed = None
        day = parsed.normalize() if parsed is not None else None
    table = data.flags.copy()
    if day is not None:
        active = edges[(edges["timestamp"] >= day) & (edges["timestamp"] < day + pd.Timedelta(days=1))]
        table = table[table.index.isin(set(active["from_account"]) | set(active["to_account"]))]
        period_text = f"{day.day} {fmt.MONTHS[day.month - 1]}"
    listed = table.index.isin(set(data.list_hits) | set(data.list_similar))
    table = table[listed | (~table.index.isin(data.businesses | data.victims) & ~table.index.str.startswith("ATM"))]
    table["risk"] = data.scores["risk_score"].reindex(table.index).fillna(0)
    # Лица из перечней — всегда в начале: операции с ними проверяются независимо от поведения.
    # Следом похожие на запись перечня: их нужно проверить вручную.
    table["listed"] = table.index.isin(set(data.list_hits))
    table["similar"] = table.index.isin(set(data.list_similar))
    table = table.sort_values(["listed", "similar", "flag_count", "risk"], ascending=False).head(limit)
    rows = []
    for account in table.index:
        phrases = list_phrases(data, account) + flag_phrases(data, account)
        reason = phrases[0] if phrases else (data.model_reasons(account) or ["—"])[0]
        rows.append({**_account_line(data, account), "risk_text": fmt.number(data.risk(account) or 0, 3),
                     "main_reason": reason})
    summary = f"Самые подозрительные счета за {period_text}:\n" + "\n".join(
        f"{i + 1}. {r['account']} — флажков {r['flags']}, риск {r['risk_text']}: {r['main_reason']}"
        for i, r in enumerate(rows))
    return {"period_text": period_text, "rows": rows, "count": len(rows), "summary": summary,
            "accounts": [r["account"] for r in rows]}


def search_knowledge(data: CaseData, query: str, k: int = 3) -> dict:
    """Дословные фрагменты базы знаний (ст. 232-1 УК РК, описания схем) с источником."""
    if data.knowledge is None:
        text = "База знаний не загружена."
        return {"error": text, "summary": text, "accounts": []}
    k = max(1, min(int(k or 3), 5))
    found = data.knowledge.search(str(query), k)
    fragments = [{"title": f["title"], "source": f["source"], "kind": f["kind"], "text": f["text"],
                  "score": round(f["score"], 3)} for f in found]
    titles = sorted({f["title"] for f in fragments})
    summary = (f"Нашёл {fmt.plural(len(fragments), 'фрагмент', 'фрагмента', 'фрагментов')} базы знаний: " + "; ".join(titles)
               + ". Текст приведён ниже дословно. Правовую оценку даёт уполномоченный орган.")
    return {"query": query, "fragments": fragments, "count": len(fragments), "summary": summary, "accounts": []}


def make_report(data: CaseData, accounts: Any, case_title: str = "Справка по делу", out_dir: Path | None = None) -> dict:
    """Собирает DOCX: движение денег, граф, флажки счетов и дословный текст нормы."""
    from bankrag.tools.report_docx import build_case_report

    seeds = [data.normalize_account(a) for a in (accounts if isinstance(accounts, list) else str(accounts).split(","))]
    seeds = [a for a in dict.fromkeys(seeds) if a]
    missing = [a for a in seeds if not data.has(a)]
    if missing or not seeds:
        return _not_found(", ".join(missing) or "—")
    from bankrag.data_loader import ROOT
    from bankrag.feedback import all_latest_decisions

    # Решения аналитика из журнала кнопок «подтвердил / отклонил» попадают в фабулу справки.
    feedback = ROOT / "data" / "feedback" / "feedback.sqlite"
    decisions = all_latest_decisions(feedback) if feedback.exists() else {}
    path, transfers = build_case_report(data, seeds, str(case_title or "Справка по делу"), out_dir, decisions)
    summary = (f"Справка «{case_title}» готова: {fmt.plural(len(seeds), 'счёт', 'счёта', 'счетов')}, "
               f"{fmt.plural(transfers, 'перевод', 'перевода', 'переводов')} в таблице движения денег, "
               "краткая фабула, граф, флажки и дословный текст ст. 232-1 УК РК, рядом приложение JSON. "
               "Скачать их можно кнопками под ответом.")
    return {"path": str(path), "file_name": path.name, "json_path": str(path.with_suffix(".json")),
            "account_count": len(seeds), "transfer_count": transfers,
            "summary": summary, "accounts": seeds}
