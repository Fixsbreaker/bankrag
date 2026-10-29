import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import faiss
import numpy as np


class EvalCLI(unittest.TestCase):
    def test_saved_queries_evaluate_and_stale_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = faiss.IndexFlatIP(2)
            index.add(np.eye(2, dtype="float32"))
            faiss.write_index(index, str(root / "faiss.index"))
            (root / "meta.json").write_text(json.dumps({"model": "test-model"}))
            (root / "chunks.json").write_text(json.dumps([
                {"chunk_id": "doc#0", "source": "doc", "text": "fee 200", "page": 1},
                {"chunk_id": "doc#1", "source": "doc", "text": "other", "page": 2}]))
            questions = [{"id": "one", "question": "fee?", "expected_source": "doc", "expected_page": 1, "expected_text": "fee 200"},
                         {"id": "two", "question": "my PIN?", "expected_source": None}]
            qpath = root / "questions.jsonl"
            qpath.write_text("\n".join(json.dumps(q) for q in questions))
            np.save(root / "query_vectors.npy", np.eye(2, dtype="float32"))
            (root / "query_vectors.meta.json").write_text(json.dumps({
                "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
                "question_ids": ["one", "two"], "model": "test-model"}))
            command = [sys.executable, "scripts/eval_retrieval.py", "--index", str(root), "--questions", str(qpath),
                       "--out", str(root / "report.json"), "--reuse-queries"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["hit_at_5"], 1)
            self.assertEqual(report["answerable_count"], 1)
            self.assertEqual(report["questions"][0]["hits"][0]["page"], 1)
            qpath.write_text(qpath.read_text() + "\n")
            stale = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("--reuse-queries", stale.stderr)


if __name__ == "__main__":
    unittest.main()
