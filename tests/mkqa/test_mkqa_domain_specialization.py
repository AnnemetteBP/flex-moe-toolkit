import unittest

from eval.benchmarks.mkqa.analyses.analyze_mkqa_domain_specialization import build_domain_specialization_records


class MkqaDomainSpecializationTests(unittest.TestCase):
    def test_language_specialization_uses_domain_token_totals(self):
        records = [
            {
                "model_name": "test-model",
                "model_path": "/tmp/test-model",
                "run_label": "native_full",
                "language": "en",
                "input_token_ids": [10, 11],
                "prompt_topk_experts_by_layer": [[[0, 1], [0, 1]]],
            },
            {
                "model_name": "test-model",
                "model_path": "/tmp/test-model",
                "run_label": "native_full",
                "language": "da",
                "input_token_ids": [20],
                "prompt_topk_experts_by_layer": [[[1, 2]]],
            },
        ]

        summary_records = build_domain_specialization_records(records, selected_sources=["prompt"])
        by_expert = {record["expert_idx"]: record for record in summary_records}

        self.assertEqual(by_expert[0]["language_token_totals"], {"da": 1, "en": 2})
        self.assertEqual(by_expert[1]["language_token_totals"], {"da": 1, "en": 2})
        self.assertEqual(by_expert[2]["language_token_totals"], {"da": 1, "en": 2})

        self.assertEqual(by_expert[0]["language_assignment_counts"], {"en": 2})
        self.assertEqual(by_expert[1]["language_assignment_counts"], {"da": 1, "en": 2})
        self.assertEqual(by_expert[2]["language_assignment_counts"], {"da": 1})

        self.assertAlmostEqual(by_expert[0]["language_specialization"]["en"], 1.0)
        self.assertAlmostEqual(by_expert[1]["language_specialization"]["en"], 1.0)
        self.assertAlmostEqual(by_expert[1]["language_specialization"]["da"], 1.0)
        self.assertAlmostEqual(by_expert[2]["language_specialization"]["da"], 1.0)

        self.assertAlmostEqual(by_expert[1]["expert_conditioned_language_distribution"]["en"], 2.0 / 3.0)
        self.assertAlmostEqual(by_expert[1]["expert_conditioned_language_distribution"]["da"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
