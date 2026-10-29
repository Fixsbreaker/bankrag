import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import build_index as builder


class BuildIndex(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = self.root / "data/raw/official"
        self.raw.mkdir(parents=True)
        self.document = self.raw / "tariff.txt"
        self.document.write_text("Broker account fee: 500 KZT.")
        self.manifest = self.root / "data/sources.json"
        self.entry = {"filename": self.document.name, "url": "https://example.com/tariff.txt",
                      "downloaded_on": "2026-10-29", "sha256": hashlib.sha256(self.document.read_bytes()).hexdigest()}
        self.manifest.write_text(json.dumps([self.entry]))

    def test_default_build_excludes_demo_and_preserves_source_and_provenance(self):
        (self.root / "data/raw/demo.md").write_text("Artificial demo-bank tariff.")
        vector = np.array([[1, 0, 0, 0]], dtype="float32")
        with mock.patch.object(builder, "ROOT", self.root), mock.patch("sys.argv", ["build_index.py"]), \
                mock.patch.object(builder, "embed", return_value=vector), contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        chunks = json.loads((self.root / "index/chunks.json").read_text())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "tariff.txt")
        self.assertEqual(chunks[0]["source_url"], self.entry["url"])
        self.assertEqual(chunks[0]["source_sha256"], self.entry["sha256"])
        self.assertEqual(chunks[0]["downloaded_on"], "2026-10-29")

    def test_modified_document_cannot_silently_bypass_hash_check(self):
        self.document.write_text("Changed tariff")
        with self.assertRaisesRegex(ValueError, "изменился"):
            builder.verify_sources(self.raw, self.manifest)

    def test_unknown_document_is_rejected(self):
        (self.raw / "unlisted.txt").write_text("Not in manifest")
        with self.assertRaisesRegex(ValueError, "отсутствуют в манифесте"):
            builder.verify_sources(self.raw, self.manifest)

    def test_parent_raw_directory_is_rejected_instead_of_losing_provenance(self):
        with self.assertRaisesRegex(ValueError, "official/tariff.txt"):
            builder.verify_sources(self.raw.parent, self.manifest)

    def test_missing_document_is_rejected(self):
        self.document.unlink()
        with self.assertRaisesRegex(ValueError, "не скачаны"):
            builder.verify_sources(self.raw, self.manifest)

    def test_missing_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "манифест не найден"):
            builder.verify_sources(self.raw, self.root / "missing.json")

    def test_duplicate_manifest_names_are_rejected(self):
        self.manifest.write_text(json.dumps([self.entry, self.entry]))
        with self.assertRaisesRegex(ValueError, "повтор filename"):
            builder.verify_sources(self.raw, self.manifest)


if __name__ == "__main__":
    unittest.main()
