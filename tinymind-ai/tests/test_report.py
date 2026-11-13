"""Tests for the report card.

The card is built from the JSON sidecar that train.py writes, so these tests
need no torch and never skip.
"""

import os
import unittest

import report

META = {
    "checkpoint": "amu.pt",
    "kind": "best",
    "created": "2026-11-11T12:00:00Z",
    "step": 2400,
    "steps_run": 2401,
    "steps_planned": 3000,
    "best_step": 2400,
    "parameters": 733696,
    "device": "cpu",
    "model": {
        "vocab_size": 71,
        "embed_dim": 128,
        "heads": 4,
        "kv_heads": 2,
        "layers": 4,
        "block_size": 128,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.0006,
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "eval_every": 200,
        "patience": 600,
        "seed": 1337,
        "optimizer": "AdamW beta=(0.9, 0.95)",
    },
    "corpus": {
        "source": "data/input.txt",
        "characters": 120000,
        "train": 108000,
        "validation": 12000,
        "vocab_size": 71,
        "sha256": "0123456789abcdef0123456789abcdef",
    },
    "losses": {
        "train": 0.3123,
        "val": 1.0842,
        "val_bits_per_char": 1.5642,
    },
    "seconds": 630.5,
}

BENCHMARK = {
    "characters": 200,
    "seconds": 0.33,
    "chars_per_second": 606.06,
    "sample": "AutoCAD keeps every object in a drawing.",
}


class ReportTests(unittest.TestCase):
    def render(self, meta=None, benchmark=None, size_bytes=2_900_000):
        return report.build_report(
            META if meta is None else meta,
            "checkpoints/amu.pt",
            size_bytes=size_bytes,
            benchmark=benchmark,
        )

    def test_names_the_checkpoint_and_its_kind(self):
        text = self.render()
        self.assertIn("amu report card", text)
        self.assertIn("checkpoints/amu.pt (best)", text)

    def test_reports_parameters_and_losses(self):
        text = self.render()
        self.assertIn("733,696", text)
        self.assertIn("train 0.3123", text)
        self.assertIn("val 1.0842", text)
        self.assertIn("1.564 bits/char", text)

    def test_reports_the_seed_and_the_corpus_checksum(self):
        text = self.render()
        self.assertIn("seed 1337", text)
        self.assertIn("120,000 characters", text)
        self.assertIn("0123456789abcdef", text)

    def test_reports_speed_and_sample_when_benchmarked(self):
        text = self.render(benchmark=BENCHMARK)
        self.assertIn("606 characters/s", text)
        self.assertIn("AutoCAD keeps every object", text)

    def test_says_speed_is_not_measured_without_a_benchmark(self):
        self.assertIn("not measured", self.render(benchmark=None))

    def test_tolerates_a_checkpoint_without_optional_sections(self):
        text = self.render(meta={"checkpoint": "amu.pt", "kind": "best"})
        self.assertIn("amu report card", text)
        self.assertIn("(best)", text)

    def test_final_card_shows_the_best_step_and_early_stop(self):
        meta = dict(META)
        meta.update(kind="final", step=2999, best_step=2400, early_stopped=True)
        text = self.render(meta=meta)
        self.assertIn("best step", text)
        self.assertIn("2400 (run stopped early)", text)

    def test_meta_path_uses_the_checkpoint_stem(self):
        self.assertEqual(
            report.meta_path("checkpoints/amu-final.pt"),
            os.path.join("checkpoints", "amu-final.json"),
        )

    def test_human_bytes_uses_readable_units(self):
        self.assertEqual(report.human_bytes(512), "512 B")
        self.assertEqual(report.human_bytes(2048), "2.0 KB")
        self.assertEqual(report.human_bytes(3 * 1024 * 1024), "3.0 MB")


if __name__ == "__main__":
    unittest.main()
