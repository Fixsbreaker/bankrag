"""Поиск фрагментов по смыслу вопроса."""
from __future__ import annotations

import json
from pathlib import Path

import faiss

from bankrag.embed import DEFAULT_MODEL, embed


class Searcher:
    def __init__(self, index: faiss.Index, chunks: list[dict], model_name: str = DEFAULT_MODEL):
        self.index = index
        self.chunks = chunks
        self.model_name = model_name

    @classmethod
    def load(cls, index_dir: str | Path = "index") -> "Searcher":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        return cls(faiss.read_index(str(index_dir / "faiss.index")), chunks, meta["model"])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Возвращает до k фрагментов: {text, source, chunk_id, score}, лучшие первыми."""
        scores, ids = self.index.search(embed([query], "query", self.model_name), k)
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def search_hybrid(self, query: str, k: int = 5) -> list[dict]:
        """Гибридная выдача. score — RRF, его нельзя сравнивать с косинусным MIN_SCORE."""
        from bankrag.hybrid import HybridRetriever
        if not hasattr(self, "_hybrid"):
            self._hybrid = HybridRetriever(self.index, self.chunks)
        return self._hybrid.search(query, embed([query], "query", self.model_name), k)
