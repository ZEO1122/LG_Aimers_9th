"""Regression tests for accidentally publishing excluded local research."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.check_publication import candidate_paths, check_paths


class PublicationGuardTest(unittest.TestCase):
    def test_ignored_data_stays_local_but_forced_tracking_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("*.csv\n", encoding="utf-8")
            data = root / "train.csv"
            data.write_text("synthetic\n1\n", encoding="utf-8")
            self.assertNotIn("train.csv", candidate_paths(root))
            subprocess.run(["git", "add", "-f", "train.csv"], cwd=root, check=True)
            errors = check_paths(root, candidate_paths(root))
            self.assertTrue(any("train.csv" in error for error in errors))
            self.assertEqual(data.read_text(), "synthetic\n1\n")

    def test_notebook_and_model_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            errors = check_paths(root, ["EDA/report.ipynb", "model.pkl"])
            self.assertEqual(len(errors), 2)

    def test_secret_is_rejected_without_echoing_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = "ghp_" + "a" * 36
            (root / "README.md").write_text(value, encoding="utf-8")
            errors = check_paths(root, ["README.md"])
            self.assertEqual(len(errors), 1)
            self.assertNotIn(value, errors[0])

    def test_symlink_cannot_publish_an_external_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").symlink_to(root / "private.txt")
            self.assertTrue(check_paths(root, ["README.md"]))

    def test_binary_cannot_hide_in_a_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_bytes(b"data\0data")
            self.assertTrue(check_paths(root, ["README.md"]))

    def test_reviewed_source_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Preprocess/common/pipeline.py"
            source.parent.mkdir(parents=True)
            source.write_text("import pandas as pd\n", encoding="utf-8")
            self.assertEqual(check_paths(root, [source.relative_to(root).as_posix()]), [])


if __name__ == "__main__":
    unittest.main()
