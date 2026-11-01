"""«Найти похожие»: поиск счетов с похожим поведением через FAISS HNSW.

    PYTHONPATH=src python -m bankrag.similarity
    PYTHONPATH=src python -m bankrag.similarity --sample 0.1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
import pandas as pd

from bankrag.data_loader import ROOT, load_config
from bankrag.features import FEATURE_COLUMNS
from bankrag.graph import build_edges, estimate_usd_rates, subgraph_between
from bankrag.labels import DROP, OTHER_LABELED, split_cutoffs

# Счётчики и транзит различаются на порядки: без логарифма расстояние определял бы один такой признак.
LOG_COLUMNS = [
    "unique_senders", "unique_receivers", "in_count", "out_count", "self_transfer_count", "transit_ratio",
    "forward_minutes_median", "forward_minutes_min", "neighbor_degree_mean", "neighbor_degree_max",
    "known_drop_neighbors",
]
# Пропуск здесь сам по себе говорит о поведении: не было входящих или пересылки.
MISSING_FLAGS = ["transit_ratio", "forward_minutes_median"]
METRICS = ("l2", "cosine")


@dataclass
class VectorScaler:
    """Стандартизация признаков счёта. Параметры считаются только на обучающем периоде."""

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def _raw(features: pd.DataFrame) -> np.ndarray:
        values = features[FEATURE_COLUMNS].astype("float64").copy()
        values[LOG_COLUMNS] = np.log1p(values[LOG_COLUMNS])
        flags = features[MISSING_FLAGS].isna().astype("float64").to_numpy()
        return np.hstack([values.to_numpy(), flags])

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "VectorScaler":
        """Обучает масштаб на переданных счетах."""
        raw = cls._raw(features)
        std = np.nanstd(raw, axis=0)
        return cls(mean=np.nanmean(raw, axis=0), std=np.where(std > 0, std, 1.0))

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Возвращает векторы float32. Пропуск становится средним значением, то есть нулём."""
        scaled = (self._raw(features) - self.mean) / self.std
        return np.nan_to_num(scaled, nan=0.0).astype("float32")

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "VectorScaler":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])


def _prepare(vectors: np.ndarray, metric: str) -> np.ndarray:
    """Для косинуса векторы нормализуются: скалярное произведение становится косинусом."""
    if metric not in METRICS:
        raise ValueError(f"неизвестная метрика {metric!r}, доступны: {METRICS}")
    prepared = vectors.copy()
    if metric == "cosine":
        faiss.normalize_L2(prepared)
    return prepared


def build_hnsw(vectors: np.ndarray, metric: str, m: int, ef_construction: int, ef_search: int) -> faiss.Index:
    """Строит индекс HNSW."""
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss_metric)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(_prepare(vectors, metric))
    return index


@dataclass
class SimilarAccounts:
    """Результат поиска: похожие счета с расстояниями и граф переводов между ними."""

    account_id: str
    neighbors: pd.DataFrame
    graph: nx.MultiDiGraph | None = None


class SimilarityIndex:
    """Индекс счетов одного окна."""

    def __init__(self, index: faiss.Index, accounts: list[str], vectors: np.ndarray, metric: str,
                 edges: pd.DataFrame | None = None):
        self.index = index
        self.accounts = accounts
        self.vectors = vectors
        self.metric = metric
        self.edges = edges
        self.positions = {account: i for i, account in enumerate(accounts)}

    def neighbors_of(self, position: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Позиции и расстояния k ближайших счетов, без самого счёта."""
        query = _prepare(self.vectors[position:position + 1], self.metric)
        # Запрос с запасом: у счетов с одинаковыми признаками сам счёт может не войти в первые k.
        distances, ids = self.index.search(query, k + 1)
        keep = (ids[0] != position) & (ids[0] != -1)
        return ids[0][keep][:k], distances[0][keep][:k]

    def find_similar(self, account_id: str, k: int = 10) -> SimilarAccounts:
        """Возвращает k похожих счетов, расстояния и подграф связей для отрисовки."""
        if account_id not in self.positions:
            raise KeyError(f"счёта {account_id} нет в индексе")
        ids, distances = self.neighbors_of(self.positions[account_id], k)
        found = [self.accounts[i] for i in ids]
        table = pd.DataFrame({"account_id": found, "distance": distances.astype(float)})
        graph = subgraph_between(self.edges, [account_id, *found]) if self.edges is not None else None
        return SimilarAccounts(account_id, table, graph)

    def save(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / f"{name}.faiss"))
        np.save(directory / f"{name}_vectors.npy", self.vectors)
        (directory / f"{name}_meta.json").write_text(
            json.dumps({"metric": self.metric, "accounts": self.accounts}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, name: str, edges: pd.DataFrame | None = None) -> "SimilarityIndex":
        meta = json.loads((directory / f"{name}_meta.json").read_text(encoding="utf-8"))
        return cls(faiss.read_index(str(directory / f"{name}.faiss")), meta["accounts"],
                   np.load(directory / f"{name}_vectors.npy"), meta["metric"], edges)


def precision_at_k(similar: SimilarityIndex, labels: pd.Series, k: int) -> dict:
    """Средняя доля дропов среди k найденных при старте от каждого дропа окна."""
    position_labels = labels.reindex(similar.accounts).to_numpy()
    starts = np.flatnonzero(position_labels == DROP)
    drops, labeled = [], []
    for start in starts:
        ids, _ = similar.neighbors_of(int(start), k)
        found = position_labels[ids]
        drops.append((found == DROP).sum() / k)
        labeled.append(((found == DROP) | (found == OTHER_LABELED)).sum() / k)
    return {"starts": len(starts), f"precision_at_{k}": float(np.mean(drops)),
            f"any_labeled_at_{k}": float(np.mean(labeled)),
            "prevalence": float((position_labels == DROP).mean())}


def measure(index: faiss.Index, vectors: np.ndarray, metric: str, k: int, queries: int, seed: int) -> dict:
    """Время одного запроса в один поток и полнота HNSW относительно точного поиска."""
    chosen = np.random.default_rng(seed).choice(len(vectors), size=min(queries, len(vectors)), replace=False)
    prepared = _prepare(vectors, metric)
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    times = []
    for i in chosen:
        started = time.perf_counter()
        index.search(prepared[i:i + 1], k)
        times.append((time.perf_counter() - started) * 1000)
    faiss.omp_set_num_threads(threads)

    exact = faiss.IndexFlatL2(vectors.shape[1]) if metric == "l2" else faiss.IndexFlatIP(vectors.shape[1])
    exact.add(prepared)
    truth = exact.search(prepared[chosen], k)[1]
    approx = index.search(prepared[chosen], k)[1]
    recall = np.mean([len(set(a) & set(t)) / k for a, t in zip(approx, truth)])
    return {"query_ms_mean": float(np.mean(times)), "query_ms_p95": float(np.percentile(times, 95)),
            "recall_vs_exact": float(recall), "queries": len(chosen)}


def network_share(similar: SimilarityIndex, labels: pd.Series, edges: pd.DataFrame, k: int) -> float:
    """Доля стартов от дропов, у которых среди k похожих есть хотя бы один перевод со стартом."""
    position_labels = labels.reindex(similar.accounts).to_numpy()
    between = edges[~edges["is_self"]]
    pairs = set(zip(between["from_account"], between["to_account"]))
    starts = np.flatnonzero(position_labels == DROP)
    linked = 0
    for start in starts:
        ids, _ = similar.neighbors_of(int(start), k)
        account = similar.accounts[start]
        linked += any((account, similar.accounts[i]) in pairs or (similar.accounts[i], account) in pairs for i in ids)
    return linked / len(starts) if len(starts) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=float, default=None, metavar="SHARE",
                        help="строить индексы по случайной доле счетов (для быстрой проверки)")
    args = parser.parse_args()

    config = load_config()
    settings = config["similarity"]
    processed = ROOT / config["paths"]["processed_dir"]
    features = pd.read_parquet(processed / "features.parquet")
    transactions = pd.read_parquet(processed / "transactions.parquet")
    train_end, val_end = split_cutoffs(transactions["timestamp"], config["split"]["fractions"])
    window_length = pd.Timedelta(hours=config["features"]["window_hours"])
    edges = build_edges(transactions, estimate_usd_rates(transactions[transactions["timestamp"] < train_end]))

    scaler = VectorScaler.fit(features[features["period"] == "train"])
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    scaler.save(models / "similarity_scaler.npz")

    def window_data(name: str) -> pd.DataFrame:
        frame = features[features["window"] == name]
        if args.sample is not None:
            frame = frame.sample(frac=args.sample, random_state=config["seed"])
        return frame

    k = settings["k"]
    params = (settings["m"], settings["ef_construction"], settings["ef_search"])
    result = {"k": k, "m": settings["m"], "ef_construction": settings["ef_construction"],
              "ef_search": settings["ef_search"], "sample": args.sample, "val": {}}

    # Метрика выбирается на валидации, тест на этом этапе не оценивается.
    validation = window_data("val")
    val_vectors = scaler.transform(validation)
    val_labels = validation.set_index("account_id")["label"]
    val_accounts = validation["account_id"].tolist()
    window = result["val"]
    for metric in METRICS:
        started = time.perf_counter()
        index = build_hnsw(val_vectors, metric, *params)
        window[metric] = {"build_seconds": time.perf_counter() - started, "accounts": len(val_vectors)}
        similar = SimilarityIndex(index, val_accounts, val_vectors, metric)
        window[metric].update(precision_at_k(similar, val_labels, k))
        window[metric].update(measure(index, val_vectors, metric, k, settings["timing_queries"], config["seed"]))
        print(metric, window[metric], flush=True)

    best = max(METRICS, key=lambda metric: window[metric][f"precision_at_{k}"])
    result["chosen_metric"] = best

    index = build_hnsw(val_vectors, best, *params)
    similar = SimilarityIndex(index, val_accounts, val_vectors, best)
    val_edges = edges[(edges["timestamp"] >= val_end - window_length) & (edges["timestamp"] < val_end)]
    result["val"]["share_of_drops_with_transfer_to_similar"] = network_share(similar, val_labels, val_edges, k)

    # Индекс и рёбра тестового окна нужны интерфейсу.
    test = window_data("test")
    test_vectors = scaler.transform(test)
    started = time.perf_counter()
    test_index = build_hnsw(test_vectors, best, *params)
    result["test_index"] = {"build_seconds": time.perf_counter() - started, "accounts": len(test_vectors)}
    SimilarityIndex(test_index, test["account_id"].tolist(), test_vectors, best).save(models, "similarity_test")
    test_end = pd.Timestamp(config["split"]["tail_start"])
    test_edges = edges[(edges["timestamp"] >= test_end - window_length) & (edges["timestamp"] < test_end)]
    test_edges[["timestamp", "from_account", "to_account", "paid_usd", "payment_format", "payment_currency",
                "is_self"]].to_parquet(processed / "edges_test.parquet", index=False)

    if args.sample is None:
        (ROOT / config["paths"]["reports_dir"] / "similarity.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
