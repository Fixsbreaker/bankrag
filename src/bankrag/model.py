"""LightGBM: оценка риска счёта и причины простыми словами.

LightGBM на macOS падает в одном процессе с sentence-transformers из-за конфликта
копий OpenMP. Поэтому библиотека загружается только внутри функций обучения и оценки,
а интерфейс читает готовые оценки и причины из scores.parquet.

    PYTHONPATH=src python -m bankrag.model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from bankrag.baseline_rule import evaluable
from bankrag.data_loader import ROOT, load_config
from bankrag.features import FEATURE_COLUMNS
from bankrag.labels import DROP

REASON_COUNT = 3


def _as_count(value: float) -> str:
    return f"{value:.0f}"


# Каждая фраза описывает значение признака за окно; 48 часов подставляются при выводе.
REASON_TEMPLATES = {
    "unique_senders": lambda v: f"разных отправителей: {_as_count(v)}",
    "unique_receivers": lambda v: f"разных получателей: {_as_count(v)}",
    "in_count": lambda v: f"входящих переводов: {_as_count(v)}",
    "out_count": lambda v: f"исходящих переводов: {_as_count(v)}",
    "self_transfer_count": lambda v: f"переводов самому себе: {_as_count(v)}",
    "transit_ratio": lambda v: (f"ушло дальше {v * 100:.0f}% от полученной суммы" if v <= 2
                                else f"отправлено в {v:.0f} раз больше, чем получено"),
    "forward_minutes_median": lambda v: f"деньги уходят дальше в среднем (медиана) через {v:.0f} мин.",
    "forward_minutes_min": lambda v: f"самая быстрая пересылка за {v:.0f} мин.",
    "night_share": lambda v: f"доля ночных операций: {v * 100:.0f}%",
    "format_count": lambda v: f"разных форматов платежа: {_as_count(v)}",
    "currency_count": lambda v: f"разных валют: {_as_count(v)}",
    "amount_log_mean": lambda v: f"средняя сумма перевода около {np.expm1(v):,.0f} USD".replace(",", " "),
    "amount_log_std": lambda v: f"суммы сильно различаются (разброс {v:.1f})",
    "neighbor_degree_mean": lambda v: f"у контрагентов в среднем {v:.1f} связей",
    "neighbor_degree_max": lambda v: f"самый связанный контрагент имеет {_as_count(v)} связей",
    "known_drop_neighbors": lambda v: f"связан с известными дропами: {_as_count(v)}",
}


def describe_reason(feature: str, value: float) -> str:
    """Переводит признак и его значение в понятную фразу."""
    if pd.isna(value):
        return {
            "transit_ratio": "входящих переводов нет",
            "forward_minutes_median": "после входящих переводов исходящих не было",
            "forward_minutes_min": "после входящих переводов исходящих не было",
        }.get(feature, "переводов с другими счетами нет")
    return REASON_TEMPLATES[feature](float(value))


def top_reasons(contributions: np.ndarray, values: np.ndarray, count: int = REASON_COUNT) -> list[list[str]]:
    """Для каждого счёта берёт признаки, сильнее всего повышающие риск, и строит фразы."""
    order = np.argsort(-contributions, axis=1)[:, :count]
    reasons = []
    for row, columns in enumerate(order):
        phrases = [describe_reason(FEATURE_COLUMNS[c], values[row, c]) for c in columns if contributions[row, c] > 0]
        reasons.append(phrases)
    return reasons


def train_model(train: pd.DataFrame, validation: pd.DataFrame, params: dict[str, Any],
                num_boost_round: int, early_stopping_rounds: int) -> Any:
    """Обучает LightGBM с ранней остановкой на валидации. Чистые и дропы, без прочих помеченных."""
    import lightgbm as lgb

    train = evaluable(train)
    validation = evaluable(validation)
    train_set = lgb.Dataset(train[FEATURE_COLUMNS], label=(train["label"] == DROP).astype(int))
    valid_set = lgb.Dataset(validation[FEATURE_COLUMNS], label=(validation["label"] == DROP).astype(int),
                            reference=train_set)
    return lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)])


def best_f1_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """Порог с лучшим F1. Подбирается на валидации и затем не меняется."""
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score_table(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Оценка риска и причины для каждого счёта окна."""
    values = features[FEATURE_COLUMNS]
    # Последний столбец pred_contrib — базовое значение, а не признак.
    contributions = model.predict(values, pred_contrib=True, num_iteration=model.best_iteration)[:, :-1]
    table = features[["account_id", "window", "period", "label"]].copy()
    table["risk_score"] = model.predict(values, num_iteration=model.best_iteration)
    table["reasons"] = top_reasons(contributions, values.to_numpy(dtype=float))
    return table


def main() -> None:
    config = load_config()
    processed: Path = ROOT / config["paths"]["processed_dir"]
    settings = config["model"]
    features = pd.read_parquet(processed / "features.parquet")
    train = features[features["period"] == "train"]
    validation = features[features["window"] == "val"]

    params = {**settings["params"], "seed": config["seed"], "verbose": -1}
    model = train_model(train, validation, params, settings["num_boost_round"], settings["early_stopping_rounds"])

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model.save_model(str(models_dir / "model.txt"), num_iteration=model.best_iteration)

    frame = evaluable(validation)
    truth = (frame["label"] == DROP).to_numpy()
    scores = model.predict(frame[FEATURE_COLUMNS], num_iteration=model.best_iteration)
    threshold = best_f1_threshold(truth, scores)
    flagged = scores >= threshold
    gain = model.feature_importance(importance_type="gain")
    result = {
        "train_drops": int((train["label"] == DROP).sum()),
        "train_clean": int((train["label"] == "clean").sum()),
        "best_iteration": model.best_iteration,
        "threshold": threshold,
        "val_pr_auc": float(average_precision_score(truth, scores)),
        "val_prevalence": float(truth.mean()),
        "val_f1": float(f1_score(truth, flagged)),
        "val_alerts": int(flagged.sum()),
        "val_drops_found": int((flagged & truth).sum()),
        "val_drops": int(truth.sum()),
        "feature_gain_share": {name: float(g / gain.sum()) for name, g in
                               sorted(zip(FEATURE_COLUMNS, gain), key=lambda item: -item[1])},
    }
    (ROOT / config["paths"]["reports_dir"] / "model.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    scoring = features[features["period"] != "train"]
    score_table(model, scoring).to_parquet(processed / "scores.parquet", index=False)


if __name__ == "__main__":
    main()
