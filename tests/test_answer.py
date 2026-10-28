"""Проверки логики answer() без обращения к настоящей модели.

    python -m unittest discover tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bankrag import answer as ans  # noqa: E402

HITS = [
    {"chunk_id": "t.md#0", "source": "t.md", "score": 0.82,
     "text": "Перевод клиенту другого банка по номеру карты: комиссия 0,5% от суммы, минимум 100 тенге."},
    {"chunk_id": "t.md#1", "source": "t.md", "score": 0.71,
     "text": "Снятие наличных в банкоматах Демо-банка бесплатно до 500 000 тенге в месяц."},
    {"chunk_id": "t.md#2", "source": "t.md", "score": 0.40, "text": "Открытие счёта бесплатно."},
]


def fake_llm(*replies: str):
    return mock.patch.object(ans, "_call_llm", side_effect=list(replies))


class ParseCitations(unittest.TestCase):
    def test_basic_and_order(self):
        self.assertEqual(ans.parse_citations("Комиссия 0,5% [2]. Минимум 100 тенге [1].", 3), [2, 1])

    def test_group_and_duplicates(self):
        self.assertEqual(ans.parse_citations("Да [1, 3] и ещё раз [1].", 3), [1, 3])

    def test_out_of_range_ignored(self):
        self.assertEqual(ans.parse_citations("См. [7] и [0].", 3), [])


class UnverifiedNumbers(unittest.TestCase):
    def test_all_supported(self):
        text = "Комиссия 0,5%, минимум 100 тенге [1]."
        self.assertEqual(ans.unverified_numbers(text, [HITS[0]["text"]]), [])

    def test_invented_number(self):
        text = "Комиссия 1% [1]."
        self.assertEqual(ans.unverified_numbers(text, [HITS[0]["text"]]), ["1"])

    def test_thousands_with_space(self):
        text = "Бесплатно до 500 000 тенге [2]."
        self.assertEqual(ans.unverified_numbers(text, [HITS[1]["text"]]), [])

    def test_citation_markers_are_not_numbers(self):
        self.assertEqual(ans.unverified_numbers("Бесплатно [2].", [HITS[1]["text"]]), [])


class Answer(unittest.TestCase):
    def test_answer_maps_numbers_to_chunk_ids(self):
        with fake_llm("Комиссия 0,5%, минимум 100 тенге [1]."):
            r = ans.answer("Какая комиссия за перевод?", HITS)
        self.assertFalse(r["refused"])
        self.assertEqual(r["sources"], ["t.md#0"])
        self.assertEqual(r["unverified"], [])

    def test_no_answer_is_refusal_without_sources(self):
        with fake_llm("NO_ANSWER"):
            r = ans.answer("Какая ставка по ипотеке?", HITS)
        self.assertTrue(r["refused"])
        self.assertEqual(r["sources"], [])
        self.assertEqual(r["answer"], ans.REFUSAL_TEXT)

    def test_missing_citations_retried_once(self):
        with fake_llm("Комиссия 0,5%.", "Комиссия 0,5% [1].") as m:
            r = ans.answer("Комиссия?", HITS)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(r["sources"], ["t.md#0"])
        self.assertFalse(r["uncited"])

    def test_still_uncited_is_flagged(self):
        with fake_llm("Комиссия 0,5%.", "Комиссия 0,5%."):
            r = ans.answer("Комиссия?", HITS)
        self.assertTrue(r["uncited"])
        self.assertEqual(r["sources"], [])

    def test_invented_number_is_flagged(self):
        with fake_llm("Комиссия 2% [1]."):
            r = ans.answer("Комиссия?", HITS)
        self.assertEqual(r["unverified"], ["2"])

    def test_min_score_refuses_without_calling_model(self):
        with fake_llm() as m:
            r = ans.answer("Погода в Астане?", HITS, min_score=0.9)
        m.assert_not_called()
        self.assertTrue(r["refused"])

    def test_no_hits_refuses_without_calling_model(self):
        with fake_llm() as m:
            r = ans.answer("Вопрос?", [])
        m.assert_not_called()
        self.assertTrue(r["refused"])


if __name__ == "__main__":
    unittest.main()
