"""Проверка модели риска: на чём держатся оценки верха очереди.

Считает средний вклад признаков (pred_contrib) у 100 счетов с наибольшим риском на тесте и
сравнивает модель с подозрительными признаками и без них. Подозрительные — те, что описывают
особенности синтетики IBM, а не поведение дропа: число валют, форматов и переводов самому себе.
Порог и раннюю остановку новая модель подбирает на валидации, тест только для отчёта.

    PYTHONPATH=src python -m bankrag.model_audit
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bankrag.baseline_rule import evaluable
from bankrag.data_loader import ROOT, load_config
from bankrag.features import FEATURE_COLUMNS
from bankrag.labels import DROP
from bankrag.model import best_f1_threshold

SUSPECT_FEATURES = ["currency_count", "format_count", "self_transfer_count"]
TOP = 100


def contributions_of_top(model, frame: pd.DataFrame, columns: list[str], top: int = TOP) -> dict[str, float]:
    """Средний вклад каждого признака в оценку у top счетов с наибольшим риском."""
    scores = model.predict(frame[columns], num_iteration=model.best_iteration)
    chosen = frame.iloc[np.argsort(-scores)[:top]]
    contributions = model.predict(chosen[columns], pred_contrib=True, num_iteration=model.best_iteration)[:, :-1]
    mean = contributions.mean(axis=0)
    return {name: float(value) for name, value in sorted(zip(columns, mean), key=lambda item: -abs(item[1]))}


def fit_and_score(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, columns: list[str],
                  config: dict) -> tuple[object, dict]:
    """Обучает LightGBM на заданных признаках и считает метрики на валидации и тесте."""
    import lightgbm as lgb

    settings = config["model"]
    params = {**settings["params"], "seed": config["seed"], "verbose": -1}
    train, validation, test = evaluable(train), evaluable(validation), evaluable(test)
    train_set = lgb.Dataset(train[columns], label=(train["label"] == DROP).astype(int))
    valid_set = lgb.Dataset(validation[columns], label=(validation["label"] == DROP).astype(int), reference=train_set)
    model = lgb.train(params, train_set, num_boost_round=settings["num_boost_round"], valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(settings["early_stopping_rounds"], first_metric_only=True,
                                                    verbose=False)])
    val_truth = (validation["label"] == DROP).to_numpy()
    val_scores = model.predict(validation[columns], num_iteration=model.best_iteration)
    threshold = best_f1_threshold(val_truth, val_scores)
    truth = (test["label"] == DROP).to_numpy()
    scores = model.predict(test[columns], num_iteration=model.best_iteration)
    flagged = scores >= threshold
    top = np.argsort(-scores)[:TOP]
    return model, {
        "features": len(columns), "val_pr_auc": float(average_precision_score(val_truth, val_scores)),
        "test_pr_auc": float(average_precision_score(truth, scores)), "test_alerts": int(flagged.sum()),
        "test_drops_found": int((flagged & truth).sum()), "test_drops": int(truth.sum()),
        "test_drops_in_top100": int(truth[top].sum()), "threshold": float(threshold)}


def main() -> None:
    config = load_config()
    features = pd.read_parquet(ROOT / config["paths"]["processed_dir"] / "features.parquet")
    train = features[features["period"] == "train"]
    validation = features[features["window"] == "val"]
    test = features[features["window"] == "test"]

    full_model, full = fit_and_score(train, validation, test, FEATURE_COLUMNS, config)
    reduced_columns = [c for c in FEATURE_COLUMNS if c not in SUSPECT_FEATURES]
    reduced_model, reduced = fit_and_score(train, validation, test, reduced_columns, config)
    result = {
        "suspect_features": SUSPECT_FEATURES,
        "top100_mean_contribution_all_features": contributions_of_top(full_model, test, FEATURE_COLUMNS),
        "top100_mean_contribution_without_suspects": contributions_of_top(reduced_model, test, reduced_columns),
        "all_features": full, "without_suspects": reduced,
    }
    (ROOT / config["paths"]["reports_dir"] / "model_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
