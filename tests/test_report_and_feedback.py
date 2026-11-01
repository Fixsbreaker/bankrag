"""Тесты базы знаний, справки и решений аналитика. Настоящая модель эмбеддингов здесь не нужна."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bankrag.feedback import CONFIRMED, REJECTED, all_latest_decisions, latest_decision, save_decision
from bankrag.kb_index import KnowledgeBase, load_fragments, parse_document
from bankrag.report import NO_NORMS, NO_TYPOLOGY, build_report

ROOT = Path(__file__).resolve().parents[1]


def fake_embed(texts: list[str], kind: str) -> np.ndarray:
    """Эмбеддинг по набору слов: тексты с общими словами оказываются близки."""
    vectors = np.zeros((len(texts), 64), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().replace(";", " ").replace(":", " ").replace(".", " ").split():
            vectors[row, hashlib.md5(word.encode()).digest()[0] % 64] += 1
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def _kb() -> KnowledgeBase:
    return KnowledgeBase.build([
        {"text": "Много отправителей сходятся в одной точке.", "kind": "typology", "title": "T", "source": "http://t"},
        {"text": "Деньги возвращаются на исходный счёт.", "kind": "typology", "title": "T", "source": "http://t"},
        {"text": "Статья про отправителей и дроп.", "kind": "norm", "title": "N", "source": "http://n"},
    ], fake_embed)


def test_parse_skips_todo_and_keeps_source() -> None:
    text = "---\ntitle: T\nkind: norm\nsource: http://x\n---\n\nTODO: вставить текст\n\nПервый абзац.\n\nВторой абзац.\n"
    fragments = parse_document(text)
    assert [f["text"] for f in fragments] == ["Первый абзац.", "Второй абзац."]
    assert all(f["source"] == "http://x" and f["kind"] == "norm" for f in fragments)


def test_law_fragments_keep_part_and_sanction_and_source() -> None:
    fragments = load_fragments(ROOT / "kb")
    norms = [f for f in fragments if f["kind"] == "norm"]
    assert norms[0]["text"].startswith("Статья 232-1.")
    assert len([f for f in norms if f["text"][0].isdigit()]) == 7
    assert all("наказыва" in f["text"] for f in norms if f["text"][0].isdigit())
    assert not [f for f in norms if f["text"].startswith("Сноска.")]
    assert all(f["source"] == "https://adilet.zan.kz/rus/docs/K1400000226" for f in norms)
    assert all(f["source"].startswith("https://") for f in fragments)


def test_report_quotes_fragments_word_for_word() -> None:
    report = build_report("A", ["много отправителей"], _kb(), top_k=3, min_similarity=0.1)
    originals = {f["text"] for f in _kb().fragments}
    assert report.typologies and all(f["text"] in originals for f in report.typologies + report.norms)
    assert report.typologies[0]["text"] == "Много отправителей сходятся в одной точке."
    assert report.norms[0]["source"] == "http://n"


def test_report_says_so_when_nothing_is_close() -> None:
    report = build_report("A", ["отправителей"], _kb(), top_k=3, min_similarity=0.99)
    assert report.no_typologies_message == NO_TYPOLOGY
    assert report.no_norms_message == NO_NORMS
    assert not report.typologies and not report.norms


def test_report_without_norms_in_base() -> None:
    kb = KnowledgeBase.build([{"text": "Много отправителей.", "kind": "typology", "title": "T", "source": "http://t"}], fake_embed)
    report = build_report("A", ["много отправителей"], kb, top_k=3, min_similarity=0.1)
    assert report.typologies
    assert report.no_norms_message == NO_NORMS


def test_report_without_reasons_finds_nothing() -> None:
    report = build_report("A", [], _kb(), top_k=3, min_similarity=0.0)
    assert not report.typologies and report.no_typologies_message == NO_TYPOLOGY


def test_kb_save_and_load(tmp_path: Path) -> None:
    kb = _kb()
    kb.save(tmp_path)
    loaded = KnowledgeBase.load(tmp_path, fake_embed)
    assert loaded.search("отправителей", 1)[0]["text"] == kb.search("отправителей", 1)[0]["text"]


def test_feedback_keeps_latest_decision(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "feedback.sqlite"
    assert latest_decision(path, "A") is None
    save_decision(path, "A", CONFIRMED, 0.9)
    save_decision(path, "A", REJECTED, 0.9)
    save_decision(path, "B", CONFIRMED)
    assert latest_decision(path, "A") == REJECTED
    assert all_latest_decisions(path) == {"A": REJECTED, "B": CONFIRMED}


def test_feedback_rejects_unknown_decision(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_decision(tmp_path / "f.sqlite", "A", "maybe")


def test_torch_is_loaded_before_faiss() -> None:
    code = ("import sys, bankrag.similarity;"
            "names = list(sys.modules);"
            "assert names.index('torch') < names.index('faiss')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src")}, timeout=120)
    assert done.returncode == 0, done.stderr


def test_interface_modules_do_not_load_lightgbm() -> None:
    code = ("import sys, bankrag.model, bankrag.report, bankrag.feedback, bankrag.similarity, bankrag.evaluate;"
            "assert 'lightgbm' not in sys.modules")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT / "src")}, timeout=60)
    assert done.returncode == 0, done.stderr
