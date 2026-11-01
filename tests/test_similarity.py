"""Тесты «Найти похожие» на маленькой фикстуре."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bankrag.features import FEATURE_COLUMNS
from bankrag.labels import CLEAN, DROP, OTHER_LABELED
from bankrag.similarity import SimilarityIndex, VectorScaler, build_hnsw, precision_at_k


def _features() -> pd.DataFrame:
    """Две группы счетов с разным поведением: по 10 посредников и 30 обычных."""
    random = np.random.default_rng(0)
    frame = pd.DataFrame(random.random((40, len(FEATURE_COLUMNS))) * 0.1, columns=FEATURE_COLUMNS)
    # Группы различаются по всем признакам: после стандартизации шум не должен их смешивать.
    frame.loc[:9, FEATURE_COLUMNS] += 5.0
    frame["account_id"] = [f"d{i}" if i < 10 else f"c{i}" for i in range(40)]
    frame["label"] = [DROP] * 10 + [CLEAN] * 29 + [OTHER_LABELED]
    return frame


def _index(metric: str = "l2") -> SimilarityIndex:
    frame = _features()
    vectors = VectorScaler.fit(frame).transform(frame)
    edges = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-09-09"] * 3), "from_account": ["d0", "d1", "c20"],
        "to_account": ["d1", "d1", "c21"], "paid_usd": [1.0, 2.0, 3.0], "payment_format": ["ACH"] * 3,
        "payment_currency": ["Euro"] * 3, "is_self": [False, True, False]})
    return SimilarityIndex(build_hnsw(vectors, metric, 16, 64, 64), frame["account_id"].tolist(), vectors, metric, edges)


def test_scaler_handles_missing_and_keeps_zero_mean() -> None:
    frame = _features()
    frame.loc[0, "transit_ratio"] = np.nan
    vectors = VectorScaler.fit(frame).transform(frame)
    assert np.isfinite(vectors).all()
    assert vectors.shape == (40, len(FEATURE_COLUMNS) + 2)
    assert abs(vectors[:, 0].mean()) < 1e-5


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_find_similar_returns_same_group_without_itself(metric: str) -> None:
    result = _index(metric).find_similar("d0", k=5)
    assert len(result.neighbors) == 5
    assert "d0" not in set(result.neighbors["account_id"])
    assert all(account.startswith("d") for account in result.neighbors["account_id"])


def test_find_similar_builds_graph_without_self_transfers() -> None:
    graph = _index().find_similar("d0", k=9).graph
    assert graph.has_edge("d0", "d1")
    assert not graph.has_edge("d1", "d1")
    assert not graph.has_edge("c20", "c21")


def test_unknown_account_is_an_error() -> None:
    with pytest.raises(KeyError):
        _index().find_similar("нет такого")


def test_precision_at_k_counts_only_drops() -> None:
    frame = _features()
    result = precision_at_k(_index(), frame.set_index("account_id")["label"], k=9)
    assert result["starts"] == 10
    assert result["precision_at_9"] == pytest.approx(1.0)
    assert result["prevalence"] == pytest.approx(0.25)


def test_save_and_load_keep_results(tmp_path: Path) -> None:
    similar = _index()
    similar.save(tmp_path, "w")
    loaded = SimilarityIndex.load(tmp_path, "w")
    assert loaded.find_similar("d0", k=3).neighbors["account_id"].tolist() == \
        similar.find_similar("d0", k=3).neighbors["account_id"].tolist()
