import unittest

import torch

from flex_moe_toolkit.pipelines.flex_olmo_routing_dataset import slice_continuation_topk_experts


class SliceContinuationTopkExpertsTests(unittest.TestCase):
    def test_returns_empty_for_non_positive_continuation_length(self):
        topk_experts = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]

        result = slice_continuation_topk_experts(topk_experts, continuation_length=0)

        self.assertEqual(result, [])

    def test_slices_2d_tensor_and_restores_batch_dimension(self):
        layer = torch.tensor(
            [
                [0, 1],
                [1, 0],
                [2, 3],
                [3, 2],
            ],
            dtype=torch.long,
        )

        result = slice_continuation_topk_experts([layer], continuation_length=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 2, 2))
        self.assertTrue(torch.equal(result[0], layer[-2:, :].unsqueeze(0)))

    def test_slices_3d_tensor_without_changing_batch_structure(self):
        layer = torch.tensor(
            [
                [
                    [0, 1],
                    [1, 0],
                    [2, 3],
                ]
            ],
            dtype=torch.long,
        )

        result = slice_continuation_topk_experts([layer], continuation_length=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (1, 2, 2))
        self.assertTrue(torch.equal(result[0], layer[:, -2:, :]))

    def test_raises_clear_error_for_unexpected_tensor_rank(self):
        layer = torch.zeros((1, 2, 3, 4), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "Unexpected top-k expert tensor shape"):
            slice_continuation_topk_experts([layer], continuation_length=2)


if __name__ == "__main__":
    unittest.main()
