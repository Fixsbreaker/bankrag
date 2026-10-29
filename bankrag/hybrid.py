"""Гибридный поиск: векторы, TF-IDF и явное название банка/документа в запросе."""
from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridRetriever:
    def __init__(self, index, chunks: list[dict]):
        self.index = index
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.lexical = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def allowed_ids(self, query: str) -> np.ndarray:
        """Пользователь может явно ограничить поиск документом или банком."""
        q = query.casefold()
        requested = set()
        if re.search(r"homebank|хоумбанк", q):
            requested = {i for i, c in enumerate(self.chunks) if "homebank" in c["source"].casefold()}
        else:
            banks = []
            if re.search(r"kaspi|каспи", q):
                banks.append("kaspi")
            if re.search(r"halyk|халык|халық|народного банка", q):
                banks.append("halyk")
            if banks:
                requested = {i for i, c in enumerate(self.chunks)
                             if any(bank in (c.get("bank", "") + " " + c["source"]).casefold() for bank in banks)}
        return np.array(sorted(requested) if requested else range(len(self.chunks)), dtype="int64")

    def search(self, query: str, vector: np.ndarray, k: int = 5) -> list[dict]:
        if k < 1:
            raise ValueError("k должно быть положительным")
        allowed = self.allowed_ids(query)
        # На текущем небольшом корпусе берём всю векторную выдачу и фильтруем её.
        # Для большого корпуса потребуется фильтрация во время поиска или overfetch.
        _, vector_ids = self.index.search(np.ascontiguousarray(vector, dtype="float32"), len(self.chunks))
        allowed_set = set(map(int, allowed))
        dense_order = [int(i) for i in vector_ids[0] if int(i) in allowed_set]
        lexical_scores = (self.vectorizer.transform([query]) @ self.lexical.T).toarray()[0]
        lexical_order = sorted(map(int, allowed), key=lambda i: (-lexical_scores[i], i))
        scores = {}
        # Reciprocal rank fusion: оба списка получают одинаковый вес.
        for order in [dense_order, lexical_order]:
            for rank, i in enumerate(order, 1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (60 + rank)
        best = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [{**self.chunks[i], "score": scores[i], "score_kind": "rrf"} for i in best]
