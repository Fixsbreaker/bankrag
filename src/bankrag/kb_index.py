"""База знаний для справок: фрагменты markdown-файлов и поиск по смыслу.

    PYTHONPATH=src python -m bankrag.kb_index
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from bankrag.data_loader import ROOT
from bankrag.faiss_safe import faiss

MODEL_NAME = "intfloat/multilingual-e5-small"
TODO_MARK = "TODO:"

Embedder = Callable[[list[str], str], np.ndarray]


def parse_document(text: str) -> list[dict]:
    """Режет документ на абзацы. Заглушки TODO и заголовки в фрагменты не попадают."""
    if not text.startswith("---\n"):
        raise ValueError("у файла базы знаний нет заголовка с полями title, kind и source")
    _, header, body = text.split("---\n", 2)
    meta = yaml.safe_load(header)
    fragments = []
    for paragraph in body.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#") or paragraph.startswith(TODO_MARK):
            continue
        fragments.append({"text": paragraph, "kind": meta["kind"], "title": meta["title"], "source": meta["source"]})
    return fragments


def load_fragments(kb_dir: Path) -> list[dict]:
    """Читает все файлы базы знаний с заголовком. sources.md — только список источников."""
    fragments = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name != "sources.md":
            fragments.extend(parse_document(path.read_text(encoding="utf-8")))
    return fragments


def e5_embedder(model_name: str = MODEL_NAME) -> Embedder:
    """Эмбеддинги e5 на CPU. Модель требует префиксы query: и passage: перед текстом."""
    from sentence_transformers import SentenceTransformer

    # После первой загрузки модель берётся из кеша, интернет не нужен.
    try:
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except OSError:
        model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: list[str], kind: str) -> np.ndarray:
        vectors = model.encode([f"{kind}: {text}" for text in texts], normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    return embed


class KnowledgeBase:
    """Поиск фрагментов по близости. Текст фрагмента возвращается дословно."""

    def __init__(self, fragments: list[dict], index: faiss.Index, embed: Embedder):
        self.fragments = fragments
        self.index = index
        self.embed = embed

    @classmethod
    def build(cls, fragments: list[dict], embed: Embedder) -> "KnowledgeBase":
        """Считает эмбеддинги фрагментов и строит точный индекс: фрагментов мало."""
        if not fragments:
            raise ValueError("база знаний пуста")
        vectors = embed([fragment["text"] for fragment in fragments], "passage")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(fragments, index, embed)

    def search(self, query: str, k: int, kind: str | None = None) -> list[dict]:
        """Лучшие k фрагментов; kind оставляет фрагменты одного вида. score — косинус."""
        scores, ids = self.index.search(self.embed([query], "query"), len(self.fragments))
        found = [{**self.fragments[i], "score": float(s)} for s, i in zip(scores[0], ids[0])
                 if kind is None or self.fragments[i]["kind"] == kind]
        return found[:k]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "kb.faiss"))
        (directory / "kb_fragments.json").write_text(json.dumps(self.fragments, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embed: Embedder) -> "KnowledgeBase":
        fragments = json.loads((directory / "kb_fragments.json").read_text(encoding="utf-8"))
        return cls(fragments, faiss.read_index(str(directory / "kb.faiss")), embed)


def main() -> None:
    fragments = load_fragments(ROOT / "kb")
    kb = KnowledgeBase.build(fragments, e5_embedder())
    kb.save(ROOT / "models")
    print(f"фрагментов: {len(fragments)}, " + ", ".join(
        f"{kind}: {sum(f['kind'] == kind for f in fragments)}" for kind in sorted({f['kind'] for f in fragments})))


if __name__ == "__main__":
    main()
