"""Справка по счёту по шаблону: RAG без генерации текста.

Сработавшие признаки превращаются в запрос, а найденные фрагменты выводятся дословно.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bankrag.kb_index import KnowledgeBase

NO_NORMS = "подходящих норм в базе знаний не найдено"
NO_TYPOLOGY = "похожих типологий в базе знаний не найдено"


@dataclass
class AccountReport:
    """Справка: причины от модели и дословные фрагменты базы знаний."""

    account_id: str
    reasons: list[str]
    query: str
    typologies: list[dict] = field(default_factory=list)
    norms: list[dict] = field(default_factory=list)
    no_typologies_message: str | None = None
    no_norms_message: str | None = None


def reasons_to_query(reasons: list[str]) -> str:
    """Запрос к базе знаний — это те же причины, что видит аналитик."""
    return "Счёт: " + "; ".join(reasons) if reasons else ""


def build_report(account_id: str, reasons: list[str], kb: KnowledgeBase, top_k: int, min_similarity: float) -> AccountReport:
    """Собирает справку. Фрагменты с низкой близостью не показываются, текст не меняется."""
    query = reasons_to_query(reasons)
    report = AccountReport(account_id, reasons, query)
    if query:
        report.typologies = [f for f in kb.search(query, top_k, "typology") if f["score"] >= min_similarity]
        report.norms = [f for f in kb.search(query, top_k, "norm") if f["score"] >= min_similarity]
    if not report.typologies:
        report.no_typologies_message = NO_TYPOLOGY
    if not report.norms:
        report.no_norms_message = NO_NORMS
    return report
