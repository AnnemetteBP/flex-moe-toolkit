import unittest

from eval_data.mkqa.analyze_mkqa_vocab_specialization import build_summary_records, collect_counts


class MkqaVocabSpecializationTests(unittest.TestCase):
    def test_vocab_specialization_uses_token_totals(self):
        records = [
            {
                "model_name": "test-model",
                "model_path": "/tmp/test-model",
                "input_token_ids": [10, 11],
                "prompt_topk_experts_by_layer": [[[0, 1], [1, 2]]],
            },
            {
                "model_name": "test-model",
                "model_path": "/tmp/test-model",
                "input_token_ids": [10],
                "prompt_topk_experts_by_layer": [[[1, 2]]],
            },
        ]

        counts, token_totals = collect_counts(records, selected_sources=["prompt"])
        summary_records = build_summary_records(
            records_path=__import__("pathlib").Path("/tmp/native_full/routing_records.jsonl"),
            records=records,
            counts=counts,
            token_totals=token_totals,
            tokenizer=None,
            top_tokens=10,
        )

        self.assertEqual(len(summary_records), 1)
        summary = summary_records[0]

        self.assertEqual(summary["token_totals"], {"10": 4, "11": 2})
        self.assertAlmostEqual(summary["mean_specialization_by_expert"]["expert_0"], 0.25)
        self.assertAlmostEqual(summary["mean_specialization_by_expert"]["expert_1"], 0.5)
        self.assertAlmostEqual(summary["mean_specialization_by_expert"]["expert_2"], 0.375)

        expert_1_top = summary["top_tokens_by_expert"]["expert_1"]
        self.assertEqual(expert_1_top[0]["token_id"], 10)
        self.assertAlmostEqual(expert_1_top[0]["specialization"], 0.5)
        self.assertEqual(expert_1_top[0]["token_total_routes"], 4)

        expert_0_top = summary["top_tokens_by_expert"]["expert_0"]
        self.assertEqual(expert_0_top[0]["token_id"], 10)
        self.assertAlmostEqual(expert_0_top[0]["specialization"], 0.25)
        self.assertEqual(expert_0_top[0]["token_total_routes"], 4)


if __name__ == "__main__":
    unittest.main()
