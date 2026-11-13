"""OCR tests: line grouping, confidence, and graceful failure.

Standard library only, no tesseract needed: parsing is tested with a
hand-written dictionary, and the failure paths are tested with mocks.
"""

import unittest
from unittest import mock

from cadagent import ocr, vision

DATA = {
    "text": ["PROJECT", "RIVERSIDE", "", "SHEET", "A-101"],
    "conf": ["96", "92", "-1", "88", "81"],
    "page_num": [1, 1, 1, 1, 1],
    "block_num": [1, 1, 1, 2, 2],
    "par_num": [1, 1, 1, 1, 1],
    "line_num": [1, 1, 1, 1, 1],
}


class ParseTests(unittest.TestCase):
    def test_words_are_grouped_into_lines(self):
        result = ocr.parse_data(DATA)
        self.assertEqual(
            [line.text for line in result.lines],
            ["PROJECT RIVERSIDE", "SHEET A-101"],
        )

    def test_confidence_is_averaged_per_line(self):
        result = ocr.parse_data(DATA)
        self.assertAlmostEqual(result.lines[0].confidence, 94.0)
        self.assertAlmostEqual(result.lines[1].confidence, 84.5)

    def test_text_joins_the_lines_with_newlines(self):
        self.assertEqual(
            ocr.parse_data(DATA).text,
            "PROJECT RIVERSIDE\nSHEET A-101",
        )

    def test_empty_output_is_an_empty_result(self):
        result = ocr.parse_data({})
        self.assertEqual(result.text, "")
        self.assertEqual(result.lines, [])
        self.assertEqual(result.confidence, 0.0)

    def test_missing_columns_do_not_crash(self):
        result = ocr.parse_data({"text": ["ONE", "TWO"]})
        self.assertEqual(result.text, "ONE TWO")


class ReadTests(unittest.TestCase):
    def test_missing_pytesseract_is_reported(self):
        with mock.patch.object(ocr, "PY_AVAILABLE", False):
            result = ocr.read("plan.png")
        self.assertIn("pytesseract", result.reason)

    def test_missing_binary_is_reported(self):
        with mock.patch.object(ocr, "PY_AVAILABLE", True):
            with mock.patch.object(ocr.shutil, "which", return_value=None):
                result = ocr.read("plan.png")
        self.assertIn("PATH", result.reason)


class DescribeTests(unittest.TestCase):
    def test_description_reports_ocr_lines_and_confidence(self):
        info = vision.ImageReport(
            path="plan.png",
            format="PNG",
            mode="RGB",
            width=100,
            height=100,
            mean_brightness=250.0,
            contrast=10.0,
            ink_ratio=0.02,
            edge_ratio=0.01,
            verdict="thin, line-like content",
            ocr_text="PROJECT RIVERSIDE",
            ocr_lines=1,
            ocr_confidence=94.0,
        )

        text = vision.describe(info)

        self.assertIn("1 lines", text)
        self.assertIn("mean confidence 94%", text)
        self.assertNotIn("low confidence", text)


if __name__ == "__main__":
    unittest.main()
