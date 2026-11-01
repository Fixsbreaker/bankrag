"""Все метрики проекта: reports/metrics.json и reports/metrics.md.

Пороги берутся из reports/rule.json и reports/model.json и здесь не подбираются.
LightGBM здесь не загружается: оценки читаются из scores.parquet.

    PYTHONPATH=src python -m bankrag.evaluate
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from bankrag.baseline_rule import Rule
from bankrag.data_loader import ROOT, load_config
from bankrag.labels import CLEAN, DROP, OTHER_LABELED
from bankrag.similarity import SimilarityIndex, VectorScaler, build_hnsw, measure, precision_at_k

MAIN_WINDOW = "test"
WINDOWS = {"test": "тест без хвоста", "test_with_tail": "тест с хвостом"}


def classification_metrics(truth: np.ndarray, flagged: np.ndarray) -> dict:
    """Precision, recall, F1 и число тревог для решения «да/нет»."""
    return {"alerts": int(flagged.sum()), "drops_found": int((flagged & truth).sum()), "drops": int(truth.sum()),
            "precision": float(precision_score(truth, flagged, zero_division=0)),
            "recall": float(recall_score(truth, flagged, zero_division=0)),
            "f1": float(f1_score(truth, flagged, zero_division=0))}


def extra_drops_at_same_alerts(truth: np.ndarray, scores: np.ndarray, rule_flags: np.ndarray) -> dict:
    """Сколько дропов находит модель, если подаёт столько же тревог, сколько правило."""
    alerts = int(rule_flags.sum())
    top = np.argsort(-scores, kind="stable")[:alerts]
    model_found = int(truth[top].sum())
    rule_found = int((truth & rule_flags).sum())
    return {"alerts": alerts, "rule_drops": rule_found, "model_drops": model_found,
            "extra_drops": model_found - rule_found}


def queue_composition(labels: pd.Series, scores: np.ndarray, size: int) -> dict:
    """Состав верхней части очереди. Чистый счёт в ней — ложная тревога."""
    top = labels.to_numpy()[np.argsort(-scores, kind="stable")[:size]]
    return {"size": size, DROP: int((top == DROP).sum()), OTHER_LABELED: int((top == OTHER_LABELED).sum()),
            "false_alarms": int((top == CLEAN).sum())}


def bootstrap_pr_auc(truth: np.ndarray, scores: np.ndarray, rounds: int, seed: int) -> list[float]:
    """95% интервал PR-AUC: при малом числе дропов одна цифра вводит в заблуждение."""
    random = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        chosen = random.integers(0, len(truth), len(truth))
        if truth[chosen].any():
            values.append(average_precision_score(truth[chosen], scores[chosen]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def drop_pattern_types(patterns: pd.DataFrame, drops: set[str]) -> dict[str, set[str]]:
    """Какие дропы участвовали в попытках каждого типа. Часть дропов в файле паттернов отсутствует."""
    sides = pd.concat([patterns[["from_account", "pattern_type"]].set_axis(["account_id", "pattern_type"], axis=1),
                       patterns[["to_account", "pattern_type"]].set_axis(["account_id", "pattern_type"], axis=1)])
    sides = sides[sides["account_id"].isin(drops)].drop_duplicates()
    return {name: set(group["account_id"]) for name, group in sides.groupby("pattern_type")}


def precision_for_starts(similar: SimilarityIndex, labels: pd.Series, starts: set[str], k: int) -> float:
    """Precision@k только для указанных стартовых счетов."""
    position_labels = labels.reindex(similar.accounts).to_numpy()
    values = []
    for account in starts:
        ids, _ = similar.neighbors_of(similar.positions[account], k)
        values.append((position_labels[ids] == DROP).sum() / k)
    return float(np.mean(values))


def evaluate_window(window: str, features: pd.DataFrame, scores: pd.DataFrame, patterns: pd.DataFrame,
                    rule: Rule, threshold: float, scaler: VectorScaler, config: dict) -> dict:
    """Считает все метрики одного окна."""
    frame = features[features["window"] == window].merge(
        scores[scores["window"] == window][["account_id", "risk_score"]], on="account_id", validate="one_to_one")
    counted = frame[frame["label"] != OTHER_LABELED]
    truth = (counted["label"] == DROP).to_numpy()
    model_scores = counted["risk_score"].to_numpy()
    rule_flags = rule.flags(counted).to_numpy()
    settings = config["similarity"]
    k = settings["k"]

    result = {
        "accounts": len(frame), "drops": int(truth.sum()),
        "other_labeled_excluded": int((frame["label"] == OTHER_LABELED).sum()),
        "prevalence": float(truth.mean()),
        "rule": classification_metrics(truth, rule_flags),
        "model": classification_metrics(truth, model_scores >= threshold),
        "model_pr_auc": float(average_precision_score(truth, model_scores)),
        "model_pr_auc_ci95": bootstrap_pr_auc(truth, model_scores, config["evaluate"]["bootstrap_rounds"], config["seed"]),
        "extra_drops": extra_drops_at_same_alerts(truth, model_scores, rule_flags),
        "queue": [queue_composition(frame["label"], frame["risk_score"].to_numpy(), size)
                  for size in config["evaluate"]["queue_sizes"]],
    }

    vectors = scaler.transform(frame)
    metric = config["evaluate"]["similarity_metric"]
    started = time.perf_counter()
    index = build_hnsw(vectors, metric, settings["m"], settings["ef_construction"], settings["ef_search"])
    build_seconds = time.perf_counter() - started
    similar = SimilarityIndex(index, frame["account_id"].tolist(), vectors, metric)
    labels = frame.set_index("account_id")["label"]
    drops = set(labels[labels == DROP].index)
    by_type = drop_pattern_types(patterns, drops)
    result["similarity"] = {
        "metric": metric, "build_seconds": build_seconds,
        **precision_at_k(similar, labels, k),
        **measure(index, vectors, metric, k, settings["timing_queries"], config["seed"]),
        "by_pattern": {name: {"starts": len(starts), f"precision_at_{k}": precision_for_starts(similar, labels, starts, k)}
                       for name, starts in sorted(by_type.items())},
        "drops_without_pattern": len(drops - set().union(*by_type.values())) if by_type else len(drops),
    }
    return result


def percent(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def render_markdown(metrics: dict) -> str:
    """Собирает отчёт для людей. Все числа подставляются из metrics.json."""
    k = metrics["k"]
    lines = ["# Метрики BankRAG", "",
             "Получено запуском `python -m bankrag.evaluate`. Данные синтетические (IBM AML, HI-Small). "
             "Пороги и настройки выбраны на валидации, на тесте не подбирались.", ""]
    main = metrics["windows"][MAIN_WINDOW]
    lines += ["## Главный результат: тест без хвоста", "",
              f"Счетов {thousands(main['accounts'])}, дропов {main['drops']}, прочих помеченных (не считаются) "
              f"{thousands(main['other_labeled_excluded'])}. Доля дропов {main['prevalence'] * 100:.3f}%.".replace(".", ","),
              "", "Тест с хвостом показан только для сравнения: после 10 сентября почти все переводы помечены, "
              "поэтому там задача заметно легче.", ""]
    lines += ["## Правило и модель", "",
              "| Окно | Метод | Тревог | Найдено дропов | Precision | Recall | F1 | PR-AUC (95% интервал) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for window, title in WINDOWS.items():
        data = metrics["windows"][window]
        for name, key in (("Правило", "rule"), ("LightGBM", "model")):
            row = data[key]
            auc = "—"
            if key == "model":
                low, high = data["model_pr_auc_ci95"]
                auc = f"{data['model_pr_auc']:.3f} ({low:.3f}–{high:.3f})".replace(".", ",")
            lines.append(f"| {title} | {name} | {row['alerts']} | {row['drops_found']} из {row['drops']} | "
                         f"{percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {auc} |")
    lines += ["", "## Дополнительные дропы при том же числе тревог", "",
              "| Окно | Тревог | Правило | Модель | Сверх правила |", "| --- | --- | --- | --- | --- |"]
    for window, title in WINDOWS.items():
        row = metrics["windows"][window]["extra_drops"]
        lines.append(f"| {title} | {row['alerts']} | {row['rule_drops']} | {row['model_drops']} | {row['extra_drops']} |")
    lines += ["", "## Верх очереди", "",
              "В очередь входят и прочие помеченные счета. Ложная тревога — чистый счёт.", "",
              "| Окно | Подсказок | Дропы | Прочие помеченные | Ложные тревоги |", "| --- | --- | --- | --- | --- |"]
    for window, title in WINDOWS.items():
        for row in metrics["windows"][window]["queue"]:
            lines.append(f"| {title} | {row['size']} | {row[DROP]} | {row[OTHER_LABELED]} | {row['false_alarms']} |")
    lines += ["", "## «Найти похожие» (FAISS HNSW)", "",
              f"| Окно | Стартов | Precision@{k} | Доля дропов в окне | Запрос, среднее / p95, мс | Построение, с | Полнота к точному поиску |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for window, title in WINDOWS.items():
        row = metrics["windows"][window]["similarity"]
        lines.append(f"| {title} | {row['starts']} | {percent(row[f'precision_at_{k}'])} | {row['prevalence'] * 100:.3f}% | "
                     f"{row['query_ms_mean']:.2f} / {row['query_ms_p95']:.2f} | {row['build_seconds']:.1f} | "
                     f"{percent(row['recall_vs_exact'])} |".replace(".", ","))
    lines += ["", "Полнота считается по совпадению id. На тесте около 9% счетов имеют одинаковые векторы, "
              "поэтому при равных расстояниях индекс и точный поиск выбирают разные счета.",
              "", f"### Precision@{k} по типам схем, тест без хвоста", "",
              "У типа мало стартов, поэтому разница между типами может быть случайной. Счёт, участвовавший "
              "в нескольких схемах, считается в каждой из них, поэтому сумма стартов больше числа дропов.", "",
              f"| Тип | Стартов | Precision@{k} |", "| --- | --- | --- |"]
    similarity = main["similarity"]
    for name, row in similarity["by_pattern"].items():
        lines.append(f"| {name} | {row['starts']} | {percent(row[f'precision_at_{k}'])} |")
    lines += ["", f"Дропов, которых нет в файле паттернов: {similarity['drops_without_pattern']} из {main['drops']}.", "",
              "## Сравнение со статьёй IBM", "",
              "Altman et al., [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models]"
              "(https://arxiv.org/abs/2306.16424), таблица 2, HI-Small, F1 миноритарного класса: "
              "PNA 56,77 ± 2,41%; GFP + LightGBM 62,86 ± 0,25%; GFP + XGBoost 63,23 ± 0,17%.", "",
              "**Напрямую сравнивать нельзя.** В статье метка стоит на переводах, и классифицируется каждый перевод; "
              "у нас метка стоит на счетах и считается только за 48-часовое окно. "
              "Общее только разбиение: тоже 60/20/20 по порядку транзакций. "
              "Их тест включает хвост, поэтому ближе к нему наш вариант с хвостом, но и он не сопоставим.", "",
              "## Казахстанские сценарии", "", "Ещё не посчитаны (этап 7).", ""]
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    processed = ROOT / config["paths"]["processed_dir"]
    reports = ROOT / config["paths"]["reports_dir"]
    features = pd.read_parquet(processed / "features.parquet")
    scores = pd.read_parquet(processed / "scores.parquet", columns=["account_id", "window", "risk_score"])
    patterns = pd.read_parquet(processed / "patterns.parquet", columns=["from_account", "to_account", "pattern_type"])
    rule = Rule(**json.loads((reports / "rule.json").read_text(encoding="utf-8"))["rule"])
    threshold = json.loads((reports / "model.json").read_text(encoding="utf-8"))["threshold"]
    scaler = VectorScaler.fit(features[features["period"] == "train"])
    config["evaluate"]["similarity_metric"] = json.loads(
        (reports / "similarity.json").read_text(encoding="utf-8"))["chosen_metric"]

    metrics = {"k": config["similarity"]["k"], "rule": rule.__dict__, "model_threshold": threshold, "windows": {}}
    for window in WINDOWS:
        metrics["windows"][window] = evaluate_window(window, features, scores, patterns, rule, threshold, scaler, config)

    (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "metrics.md").write_text(render_markdown(metrics), encoding="utf-8")
    print(render_markdown(metrics))


if __name__ == "__main__":
    main()
