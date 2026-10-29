import unittest
import faiss
import numpy as np
from bankrag.hybrid import HybridRetriever


class Hybrid(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": "h0", "source": "halyk_homebank.pdf", "text": "Пополнение брокерского счета 500 тенге"},
            {"chunk_id": "k0", "source": "kaspi.pdf", "text": "Брокерский счет 100 тенге"},
            {"chunk_id": "h1", "source": "halyk_rko.pdf", "text": "Открытие текущего счета бесплатно"},
        ]
        index = faiss.IndexFlatIP(3)
        index.add(np.eye(3, dtype="float32"))
        self.retriever = HybridRetriever(index, self.chunks)

    def test_explicit_document_keeps_other_bank_out_even_when_dense_prefers_it(self):
        hits = self.retriever.search("Homebank брокерский счет", np.array([[0, 1, 0]], dtype="float32"), 5)
        self.assertEqual([h["chunk_id"] for h in hits], ["h0"])
        self.assertEqual(hits[0]["score_kind"], "rrf")

    def test_generic_question_does_not_silently_filter_a_bank(self):
        hits = self.retriever.search("Открыть счет", np.array([[0, 0, 1]], dtype="float32"), 5)
        self.assertEqual({h["chunk_id"] for h in hits}, {"h0", "k0", "h1"})


if __name__ == "__main__":
    unittest.main()
