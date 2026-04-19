import unittest

from flex_moe_toolkit.pipelines.flex_olmo_eval import build_run_specs


class BuildRunSpecsTests(unittest.TestCase):
    def test_native_only_returns_single_native_full_run(self):
        run_specs = build_run_specs(
            num_experts=8,
            public_expert_idx=0,
            combined_active_counts=(2, 4, 7),
            include_individual_experts=False,
            routing_run_mode="native_only",
        )

        self.assertEqual([run_spec.label for run_spec in run_specs], ["native_full"])
        self.assertFalse(run_specs[0].apply_restricted_routing)
        self.assertEqual(run_specs[0].run_kind, "native_full")
        self.assertEqual(run_specs[0].allowed_experts, tuple(range(8)))

    def test_native_plus_restricted_keeps_native_and_restricted_runs(self):
        run_specs = build_run_specs(
            num_experts=8,
            public_expert_idx=0,
            combined_active_counts=(2, 4),
            include_individual_experts=False,
            routing_run_mode="native_plus_restricted",
        )

        self.assertEqual(
            [run_spec.label for run_spec in run_specs],
            ["native_full", "public_only", "combined_top2", "combined_top4"],
        )
        self.assertFalse(run_specs[0].apply_restricted_routing)
        self.assertTrue(all(run_spec.apply_restricted_routing for run_spec in run_specs[1:]))

    def test_restricted_sweep_preserves_previous_behavior(self):
        run_specs = build_run_specs(
            num_experts=2,
            public_expert_idx=0,
            combined_active_counts=(2, 4, 7),
            include_individual_experts=False,
            routing_run_mode="restricted_sweep",
        )

        self.assertEqual(
            [run_spec.label for run_spec in run_specs],
            ["public_only", "combined_top2"],
        )
        self.assertTrue(all(run_spec.apply_restricted_routing for run_spec in run_specs))


if __name__ == "__main__":
    unittest.main()
