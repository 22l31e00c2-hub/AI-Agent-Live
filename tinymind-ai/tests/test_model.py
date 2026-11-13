"""Neural-network tests for amu.

These tests need torch, which lives in the project virtualenv. Run the suite
with `python run_tests.py`: it switches to that interpreter and fails the run if
any test was skipped. Everything else in the suite is standard library only.
"""

import unittest

try:
    import torch

    from model import Amu

    TORCH_AVAILABLE = True
except ImportError:      # system python without torch
    torch = None
    Amu = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class ModelTests(unittest.TestCase):
    def build(self, **kwargs):
        torch.manual_seed(0)
        options = dict(
            vocab_size=32, embed_dim=32, heads=4, layers=2, block_size=16
        )
        options.update(kwargs)
        return Amu(**options)

    def test_model_has_parameters_and_a_vocabulary_head(self):
        model = self.build()
        total = sum(p.numel() for p in model.parameters())
        self.assertGreater(total, 1000)
        self.assertEqual(model.output_head.out_features, 32)

    def test_tied_embeddings_share_one_matrix(self):
        model = self.build()
        self.assertIs(
            model.output_head.weight, model.token_embedding.weight
        )

    def test_future_tokens_cannot_change_past_logits(self):
        model = self.build()
        model.eval()

        idx = torch.randint(0, 32, (1, 16))
        baseline, _ = model(idx)

        altered = idx.clone()
        altered[0, -1] = (altered[0, -1] + 1) % 32
        changed, _ = model(altered)

        self.assertTrue(
            torch.allclose(baseline[:, :-1], changed[:, :-1], atol=1e-5)
        )
        self.assertFalse(
            torch.allclose(baseline[:, -1], changed[:, -1], atol=1e-5)
        )

    def test_loss_is_finite_for_a_batch(self):
        model = self.build()
        idx = torch.randint(0, 32, (2, 16))
        logits, loss = model(idx, idx)

        self.assertEqual(tuple(logits.shape), (2, 16, 32))
        self.assertTrue(torch.isfinite(loss))

    def test_cached_step_matches_full_context(self):
        model = self.build()
        model.eval()

        idx = torch.randint(0, 32, (1, 12))
        full, _ = model(idx)

        _, past = model._run(idx[:, :-1])
        step_logits, _ = model._run(idx[:, -1:], past)

        self.assertTrue(
            torch.allclose(full[:, -1], step_logits[:, 0], atol=1e-5)
        )

    def test_grouped_query_attention_shrinks_the_key_value_projection(self):
        model = self.build(kv_heads=2)
        attention = model.blocks[0].attention

        self.assertEqual(attention.heads, 4)
        self.assertEqual(attention.kv_heads, 2)
        self.assertEqual(attention.key.out_features, 2 * attention.head_dim)
        self.assertEqual(attention.group, 2)

    def test_generation_crosses_the_block_size_without_failing(self):
        model = self.build()
        prompt = torch.randint(0, 32, (1, 4))

        out = model.generate(prompt, max_new_tokens=40, temperature=1.0)

        self.assertEqual(out.shape, (1, 44))
        self.assertTrue(bool((out < 32).all()))
        self.assertTrue(torch.equal(out[:, :4], prompt))


if __name__ == "__main__":
    unittest.main()
