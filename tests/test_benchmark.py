import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.benchmark import git_state, recall_at_k


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
            self.assertEqual(metadata["paths_relative_to"], "repository root")
            for field in ("vectors", "queries", "out"):
                self.assertFalse(Path(metadata["parameters"][field]).is_absolute())

    def test_git_state_reports_clean_commit_then_local_modifications(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            source = root / "source.py"
            source.write_text("pass\n")
            subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Benchmark Test", "-c", "user.email=test@example.com",
                            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)
            revision, dirty = git_state(root)
            self.assertEqual(len(revision), 40)
            self.assertFalse(dirty)
            source.write_text("print('changed')\n")
            self.assertEqual(git_state(root), (revision, True))

    def test_require_clean_rejects_dirty_source_before_running_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "source.py").write_text("pass\n")
            subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Benchmark Test", "-c", "user.email=test@example.com",
                            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)
            (root / "source.py").write_text("modified\n")
            from unittest import mock
            from scripts import benchmark
            with mock.patch.object(benchmark, "ROOT", root), \
                    mock.patch("sys.argv", ["benchmark.py", "--require-clean"]), \
                    mock.patch.object(benchmark, "load_vectors") as load:
                with self.assertRaises(SystemExit):
                    benchmark.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
