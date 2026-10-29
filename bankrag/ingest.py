"""Загрузка документов из data/raw/ и разбиение на фрагменты."""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str, max_chars: int = 400, overlap: int = 80) -> list[str]:
    """Режет текст по абзацам; длинные абзацы режет окном с перекрытием."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = f"{current}\n{p}".strip()
            continue
        if current:
            chunks.append(current)
        while len(p) > max_chars:
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap:]
        current = p
    if current:
        chunks.append(current)
    return chunks


def load_corpus(raw_dir: Path, max_chars: int = 400) -> list[dict]:
    """Возвращает фрагменты: {chunk_id, source, text}."""
    corpus = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            pages = [(number, page.extract_text() or "") for number, page in enumerate(PdfReader(path).pages, 1)]
        else:
            pages = [(None, read_document(path))]
        for page_number, text in pages:
            for i, chunk in enumerate(split_into_chunks(text, max_chars)):
                page_id = f"p{page_number}:" if page_number is not None else ""
                corpus.append({
                    "chunk_id": f"{path.name}#{page_id}{i}",
                    "source": str(path.relative_to(raw_dir)),
                    "text": chunk,
                    **({"page": page_number} if page_number is not None else {}),
                })
    return corpus
