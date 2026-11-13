"""Tests for the training corpus and its generator.

These are standard-library only: no torch, no skips. They keep data/input.txt
honest - it must be the exact output of data/build_corpus.py, large enough to
train on, and plain ASCII.
"""

import importlib.util
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "data" / "build_corpus.py"
CORPUS = ROOT / "data" / "input.txt"


def load_generator():
    spec = importlib.util.spec_from_file_location("amu_build_corpus", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator()
        cls.text = CORPUS.read_text(encoding="utf-8")

    def test_corpus_is_the_generator_output(self):
        self.assertEqual(self.text, self.generator.build())

    def test_corpus_is_large_enough_to_train_on(self):
        self.assertGreaterEqual(len(self.text), 100_000)

    def test_corpus_is_plain_ascii(self):
        self.assertTrue(all(ord(char) < 128 for char in self.text))

    def test_corpus_covers_all_four_applications(self):
        lowered = self.text.lower()
        for name in ["autocad", "revit", "map 3d", "qgis"]:
            self.assertIn(name, lowered)

    def test_corpus_contains_ocr_and_code_material(self):
        for marker in ["tesseract", "def ", "(defun", "-LAYER", "qgis_process"]:
            self.assertIn(marker, self.text)

    def test_generator_is_deterministic(self):
        self.assertEqual(
            self.generator.build(seed=7, target=5000),
            self.generator.build(seed=7, target=5000),
        )

    def test_no_template_placeholders_survive_formatting(self):
        # Code samples legitimately contain braces. A leftover template slot is
        # a brace pair around a lowercase name, like {layer} or {code}.
        self.assertIsNone(
            re.search(r"\{[a-z_]+\}", self.text),
            "unfilled template placeholder left in the corpus",
        )


if __name__ == "__main__":
    unittest.main()
