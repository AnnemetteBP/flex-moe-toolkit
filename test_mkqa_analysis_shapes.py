import unittest

from eval_data.mkqa.analyze_mkqa_domain_specialization import normalize_layer_token_experts as normalize_domain
from eval_data.mkqa.analyze_mkqa_vocab_specialization import normalize_layer_token_experts as normalize_vocab


class NormalizeLayerTokenExpertsTests(unittest.TestCase):
    def test_accepts_batched_token_expert_shape(self):
        layer_assignments = [[[0, 1], [2, 3]]]

        self.assertEqual(normalize_vocab(layer_assignments), [[0, 1], [2, 3]])
        self.assertEqual(normalize_domain(layer_assignments), [[0, 1], [2, 3]])

    def test_accepts_unbatched_token_expert_shape(self):
        layer_assignments = [[0, 1], [2, 3]]

        self.assertEqual(normalize_vocab(layer_assignments), [[0, 1], [2, 3]])
        self.assertEqual(normalize_domain(layer_assignments), [[0, 1], [2, 3]])

    def test_wraps_scalar_expert_assignments(self):
        layer_assignments = [0, 2, 3]

        self.assertEqual(normalize_vocab(layer_assignments), [[0], [2], [3]])
        self.assertEqual(normalize_domain(layer_assignments), [[0], [2], [3]])

    def test_returns_empty_for_missing_assignments(self):
        self.assertEqual(normalize_vocab([]), [])
        self.assertEqual(normalize_domain([]), [])


if __name__ == "__main__":
    unittest.main()
