"""
Unit tests for activation extraction and anchor resolution.
"""

import unittest
import torch

from src.activations import ExtractionResult


class TestAnchorResolution(unittest.TestCase):
    def setUp(self):
        # 4 layers, 10 prompt tokens, 5 completion tokens, 64 hidden dim
        self.num_layers = 4
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.hidden_dim = 64
        self.hidden_states = torch.randn(
            self.num_layers,
            self.prompt_tokens + self.completion_tokens,
            self.hidden_dim,
            dtype=torch.float16,
        )
        self.result = ExtractionResult(
            completion_text="think: I need planks.\nAction: craft 4 oak planks",
            hidden_states=self.hidden_states,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

    def test_supported_anchors(self):
        anchors = ExtractionResult.supported_anchors()
        self.assertIn("pre_gen", anchors)
        self.assertIn("post_gen", anchors)
        self.assertIn("first_gen", anchors)
        self.assertIn("mean_gen", anchors)
        self.assertIn("mean_prompt", anchors)

    def test_pre_gen_and_post_gen(self):
        pre = self.result.get_anchor_state("pre_gen")
        self.assertEqual(pre.shape, (self.num_layers, self.hidden_dim))
        # pre_gen should match prompt_tokens - 1
        self.assertTrue(torch.equal(pre, self.hidden_states[:, self.prompt_tokens - 1, :]))

        post = self.result.get_anchor_state("post_gen")
        self.assertEqual(post.shape, (self.num_layers, self.hidden_dim))
        self.assertTrue(torch.equal(post, self.hidden_states[:, -1, :]))

    def test_mean_pooling_anchors(self):
        mean_gen = self.result.get_anchor_state("mean_gen")
        self.assertEqual(mean_gen.shape, (self.num_layers, self.hidden_dim))
        expected_gen = self.hidden_states[:, self.prompt_tokens :, :].mean(dim=1)
        self.assertTrue(torch.allclose(mean_gen, expected_gen))

        mean_prompt = self.result.get_anchor_state("mean_prompt")
        self.assertEqual(mean_prompt.shape, (self.num_layers, self.hidden_dim))
        expected_prompt = self.hidden_states[:, : self.prompt_tokens, :].mean(dim=1)
        self.assertTrue(torch.allclose(mean_prompt, expected_prompt))

    def test_layer_slicing(self):
        sliced = self.result.get_anchor_state("post_gen", layer_ids=[1, 3])
        self.assertEqual(sliced.shape, (2, self.hidden_dim))

    def test_integer_offset_anchor(self):
        state_last = self.result.get_anchor_state("-1")
        post = self.result.get_anchor_state("post_gen")
        self.assertTrue(torch.equal(state_last, post))


if __name__ == "__main__":
    unittest.main()
