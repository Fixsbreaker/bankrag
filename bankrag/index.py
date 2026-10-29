"""Векторные индексы FAISS. Базовая версия; собственный движок подключится вместо неё."""
from __future__ import annotations

import faiss
import numpy as np

KINDS = ("flat", "hnsw", "hnsw_sq8", "ivf_pq")


def build_index(kind: str, vectors: np.ndarray, hnsw_m: int = 32, pq_m: int = 16) -> faiss.Index:
    """Векторы нормализованы, поэтому скалярное произведение = косинусная близость."""
    d = vectors.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, ip)
    elif kind == "hnsw_sq8":
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, hnsw_m, ip)
    elif kind == "ivf_pq":
        if len(vectors) < 256:
            raise ValueError("ivf_pq: для обучения PQ нужно минимум 256 фрагментов")
        # Число кластеров ограничено размером корпуса, иначе обучение не сойдётся.
        nlist = max(1, min(64, len(vectors) // 39))
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, pq_m, 8, ip)
        index.nprobe = min(8, nlist)
    else:
        raise ValueError(f"неизвестный индекс {kind!r}, доступны: {KINDS}")
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    return index


def bytes_per_vector(kind: str, d: int, pq_m: int = 16) -> int:
    """Размер кода вектора. Не включает граф, ID, центроиды и словари PQ."""
    return {"flat": 4 * d, "hnsw": 4 * d, "hnsw_sq8": d, "ivf_pq": pq_m}[kind]
