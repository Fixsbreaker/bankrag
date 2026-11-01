"""Простое правило-ориентир, как в банках. Пороги подбираются только на валидации.

    PYTHONPATH=src python -m bankrag.baseline_rule
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import f1_score

from bankrag.data_loader import ROOT, load_config
from bankrag.labels import DROP, OTHER_LABELED


@dataclass(frozen=True)
class Rule:
    """Счёт подозрителен, если отправителей много, деньги уходят дальше и быстро."""

    min_senders: int
    min_transit: float
    max_forward_minutes: float
    max_transit: float | None = None

    def flags(self, features: pd.DataFrame) -> pd.Series:
        """True для счетов, на которых правило срабатывает. Пропуски условие не выполняют."""
        max_transit = self.max_transit if self.max_transit is not None else 1 / self.min_transit
        return (
            (features["unique_senders"] >= self.min_senders)
            & (features["transit_ratio"] >= self.min_transit)
            & (features["transit_ratio"] <= max_transit)
            & (features["forward_minutes_median"] <= self.max_forward_minutes)
        )


def fast_transit_rule(config: dict) -> Rule:
    """Второй сигнал очереди: быстрый транзит. Пороги заданы вручную и на данных не подбираются."""
    return Rule(**config["rule"]["fast_transit"])


def fast_transit_reasons(rule: Rule, unique_senders: float, transit_ratio: float, forward_minutes: float) -> list[str]:
    """Объяснение срабатывания правила теми числами, по которым оно сработало."""
    return [f"разных отправителей: {unique_senders:.0f} (порог {rule.min_senders})",
            f"ушло дальше {transit_ratio * 100:.0f}% от полученной суммы "
            f"(порог {rule.min_transit * 100:.0f}–{rule.max_transit * 100:.0f}%)",
            f"деньги уходят дальше через {forward_minutes:.0f} мин. (порог {rule.max_forward_minutes:.0f} мин.)"]


def evaluable(features: pd.DataFrame) -> pd.DataFrame:
    """Убирает «прочие помеченные»: в метриках они не считаются ни дропами, ни чистыми."""
    return features[features["label"] != OTHER_LABELED]


def tune_rule(validation: pd.DataFrame, grid: dict[str, list[float]]) -> tuple[Rule, float]:
    """Перебирает пороги и возвращает правило с лучшим F1 на переданной выборке."""
    frame = evaluable(validation)
    truth = frame["label"] == DROP
    best_rule, best_score = None, -1.0
    for senders, transit, minutes in itertools.product(
            grid["min_senders"], grid["min_transit"], grid["max_forward_minutes"]):
        rule = Rule(int(senders), float(transit), float(minutes))
        score = f1_score(truth, rule.flags(frame), zero_division=0)
        if score > best_score:
            best_rule, best_score = rule, score
    if best_rule is None:
        raise ValueError("пустая сетка порогов")
    return best_rule, float(best_score)


def main() -> None:
    config = load_config()
    processed = ROOT / config["paths"]["processed_dir"]
    features = pd.read_parquet(processed / "features.parquet")
    validation = features[features["window"] == "val"]

    rule, score = tune_rule(validation, config["rule"]["grid"])
    frame = evaluable(validation)
    alerts = int(rule.flags(frame).sum())
    hits = int((rule.flags(frame) & (frame["label"] == DROP)).sum())
    result = {"rule": asdict(rule), "val_f1": score, "val_alerts": alerts, "val_drops_found": hits,
              "val_drops": int((frame["label"] == DROP).sum())}
    (ROOT / config["paths"]["reports_dir"] / "rule.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
