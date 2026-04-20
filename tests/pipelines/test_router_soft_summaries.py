import unittest

import torch

from flex_moe_toolkit.pipelines.flex_olmo_routing_dataset import (
    aggregate_router_token_summaries,
    slice_router_token_summaries,
    summarize_router_token_distributions,
)


class RouterSoftSummariesTests(unittest.TestCase):
    def test_router_token_summary_captures_margins_and_selected_mass(self):
        router_logits = [
            torch.tensor(
                [
                    [
                        [3.0, 1.0, 0.0],
                        [0.5, 2.5, 0.0],
                    ]
                ]
            )
        ]
        topk_experts = [
            torch.tensor(
                [
                    [
                        [0, 1],
                        [1, 0],
                    ]
                ]
            )
        ]

        summaries = summarize_router_token_distributions(router_logits, topk_experts)

        self.assertEqual(len(summaries), 1)
        layer = summaries[0]
        self.assertEqual(layer["layer_idx"], 0)
        self.assertEqual(tuple(layer["top1_expert_ids"].shape), (1, 2))
        self.assertEqual(tuple(layer["top1_probs"].shape), (1, 2))
        self.assertEqual(tuple(layer["top1_top2_margin"].shape), (1, 2))
        self.assertEqual(tuple(layer["selected_expert_prob_mass"].shape), (1, 2))
        self.assertTrue(torch.all(layer["top1_top2_margin"] > 0))
        self.assertTrue(torch.all(layer["selected_expert_prob_mass"] <= 1.0))

        aggregate = aggregate_router_token_summaries(summaries)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["layer_idx"], 0)
        self.assertGreater(aggregate[0]["mean_top1_prob"], aggregate[0]["mean_top2_prob"])
        self.assertGreater(aggregate[0]["mean_selected_expert_prob_mass"], aggregate[0]["mean_top1_prob"])

    def test_slice_router_token_summaries_keeps_suffix(self):
        summaries = [
            {
                "layer_idx": 0,
                "top1_expert_ids": torch.tensor([[0, 1, 2]]),
                "top1_probs": torch.tensor([[0.9, 0.8, 0.7]]),
                "top2_expert_ids": torch.tensor([[1, 2, 0]]),
                "top2_probs": torch.tensor([[0.1, 0.2, 0.3]]),
                "top1_top2_margin": torch.tensor([[0.8, 0.6, 0.4]]),
                "token_entropy": torch.tensor([[0.2, 0.3, 0.4]]),
                "selected_expert_prob_mass": torch.tensor([[1.0, 1.0, 1.0]]),
            }
        ]

        sliced = slice_router_token_summaries(summaries, suffix_length=2)

        self.assertEqual(tuple(sliced[0]["top1_expert_ids"].shape), (1, 2))
        self.assertTrue(torch.equal(sliced[0]["top1_expert_ids"], torch.tensor([[1, 2]])))


if __name__ == "__main__":
    unittest.main()
