"""Содержательные крайние случаи разметки и знаменателей метрик."""
import unittest

from bankrag.evaluation import score_retrieval, validate_questions


class Evaluation(unittest.TestCase):
    def setUp(self):
        self.right = {"source": "tariffs.pdf", "chunk_id": "t#1", "text": "Комиссия 200 тенге", "score": 0.7}
        self.wrong = {"source": "tariffs.pdf", "chunk_id": "t#0", "text": "Выпуск карты", "score": 0.8}
        self.q = {"id": "1", "question": "Комиссия?", "expected_source": "tariffs.pdf", "expected_text": "200 тенге"}

    def test_same_document_wrong_chunk_does_not_count(self):
        report = score_retrieval([self.q], [[self.wrong]], 5)
        self.assertEqual(report["hit_at_5"], 0)

    def test_rank_and_unanswerable_denominator(self):
        no_answer = {"id": "2", "question": "Ипотека?", "expected_source": None}
        report = score_retrieval([self.q, no_answer], [[self.wrong, self.right], [self.wrong]], 5)
        self.assertEqual(report["hit_at_5"], 1)
        self.assertEqual(report["mrr_at_5"], 0.5)
        self.assertEqual(report["unanswerable_count"], 1)
        self.assertFalse(report["refusal_evaluated"])

    def test_stale_annotation_stops_evaluation(self):
        with self.assertRaises(ValueError):
            validate_questions([self.q], [self.wrong])

    def test_pdf_whitespace_is_normalized(self):
        q = {**self.q, "expected_text": "200\nТЕНГЕ"}
        validate_questions([q], [self.right])

    def test_no_positive_questions_has_no_quality_estimate(self):
        q = {"id": "n", "question": "Ипотека?", "expected_source": None}
        self.assertIsNone(score_retrieval([q], [[]], 5)["hit_at_5"])

    def test_relevant_hit_beyond_k_does_not_count(self):
        self.assertEqual(score_retrieval([self.q], [[self.wrong, self.right]], 1)["hit_at_1"], 0)

    def test_same_phrase_on_wrong_page_is_not_the_annotated_answer(self):
        q = {**self.q, "expected_page": 2}
        report = score_retrieval([q], [[{**self.right, "page": 1}]], 5)
        self.assertEqual(report["hit_at_5"], 0)


if __name__ == "__main__":
    unittest.main()
