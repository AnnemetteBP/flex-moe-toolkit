import unittest

import torch

from flex_moe_toolkit.core.routing_diagnostics import (
    compute_all_metrics,
    compute_coactivation,
    compute_coactivation_counts,
    normalize_coactivation_counts,
)


class RoutingCoactivationTests(unittest.TestCase):
    def test_normalized_coactivation_matches_conditional_definition(self):
        router_logits = torch.tensor(
            [
                [
                    [10.0, 9.0, -10.0],   # experts 0 and 1
                    [10.0, 9.0, -10.0],   # experts 0 and 1
                    [10.0, -10.0, 9.0],   # experts 0 and 2
                ]
            ]
        )

        coactivation_counts, activation_counts = compute_coactivation_counts(router_logits, top_k=2)
        coactivation = normalize_coactivation_counts(coactivation_counts, activation_counts)

        expected_counts = torch.tensor(
            [
                [3.0, 2.0, 1.0],
                [2.0, 2.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        expected_activation_counts = torch.tensor([3.0, 2.0, 1.0])
        expected_coactivation = torch.tensor(
            [
                [1.0, 2.0 / 3.0, 1.0 / 3.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )

        self.assertTrue(torch.allclose(coactivation_counts, expected_counts))
        self.assertTrue(torch.allclose(activation_counts, expected_activation_counts))
        self.assertTrue(torch.allclose(coactivation, expected_coactivation))
        self.assertTrue(torch.allclose(compute_coactivation(router_logits, top_k=2), expected_coactivation))

    def test_compute_all_metrics_exposes_counts_and_normalized_matrix(self):
        router_logits = torch.tensor([[[5.0, 4.0], [5.0, 4.0]]])

        metrics = compute_all_metrics(router_logits, top_k=2)

        self.assertIn("coactivation_counts", metrics)
        self.assertIn("activation_counts", metrics)
        self.assertIn("coactivation_matrix", metrics)
        self.assertEqual(tuple(metrics["coactivation_counts"].shape), (2, 2))
        self.assertEqual(tuple(metrics["activation_counts"].shape), (2,))
        self.assertTrue(torch.allclose(metrics["coactivation_matrix"], torch.ones((2, 2))))


if __name__ == "__main__":
    unittest.main()
