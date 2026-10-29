import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.benchmark import recall_at_k


class Benchmark(unittest.TestCase):
    def test_recall_counts_ids_and_ignores_missing_neighbors(self):
        expected = np.array([[1, 2, -1], [4, 5, 6]])
        actual = np.array([[2, 2, -1], [6, 4, -1]])
        self.assertAlmostEqual(recall_at_k(expected, actual), (0.5 + 2 / 3) / 2)

    def test_cli_small_corpus_measures_full_index_and_marks_pq_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = np.eye(4, dtype="float32")
            np.save(root / "vectors.npy", vectors)
            np.save(root / "queries.npy", vectors[:2])
            result = subprocess.run([sys.executable, "scripts/benchmark.py", "--vectors", str(root / "vectors.npy"),
                                     "--queries", str(root / "queries.npy"), "--out", str(root / "bench.csv"),
                                     "--repeats", "2", "-k", "5"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "bench.csv").open() as file:
                rows = {r["index"]: r for r in csv.DictReader(file)}
            self.assertEqual(rows["ivf_pq"]["status"], "skipped")
            self.assertEqual(float(rows["flat"]["recall_at_5"]), 1)
            self.assertGreater(int(rows["flat"]["index_bytes"]), vectors.nbytes)
            self.assertEqual(int(rows["flat"]["effective_k"]), 4)
            self.assertEqual(int(rows["flat"]["timed_queries"]), 4)
            metadata = json.loads((root / "bench.meta.json").read_text())
            self.assertEqual(metadata["parameters"]["threads"], 1)


if __name__ == "__main__":
    unittest.main()
