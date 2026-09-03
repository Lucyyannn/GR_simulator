import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_activity_weighted_qps.py"
SPEC = importlib.util.spec_from_file_location("activity_weighted_qps", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


class ActivityWeightedQpsTest(unittest.TestCase):
    def case(self, **overrides):
        values = {
            "chip": "910C", "model": "small", "seq_len": 4096,
            "batch_size": 4, "user": "hot", "method": "w_AR",
            "latency_us": 100.0, "layers": 4, "hidden": 256,
            "recompute_items": 0, "kv_reuse_ratio": 0.4802,
            "ar_attention_compute": "disabled",
        }
        values.update(overrides)
        return ANALYZER.CaseResult(**values)

    def test_action_reuse_matches_existing_integer_row_rule(self):
        self.assertEqual(ANALYZER.retained_action_rows(4096, 0.4802), 81)
        self.assertEqual(ANALYZER.retained_action_rows(6144, 0.4802), 122)
        self.assertEqual(ANALYZER.retained_action_rows(8192, 0.4802), 162)

    def test_persistent_kv_excludes_recomputed_items_and_reused_actions(self):
        case = self.case(recompute_items=48)
        rows, size = ANALYZER.persistent_kv_bytes(case, 2.0)
        self.assertEqual(rows, 2048 - 48 + 81)
        self.assertEqual(size, 2 * 4 * 256 * rows * 2)

    def test_scaled_activity_supports_partial_rank_bucket(self):
        # Two empirical ranks, each representing 100 users.  Capacity holds
        # all 100 users in the first rank and 50 users in the second.
        resident, user_fraction, request_fraction = ANALYZER.ddr_coverage(
            activity=[10, 2], user_scale=100,
            capacity_bytes=150, user_bytes=1,
        )
        self.assertEqual(resident, 150)
        self.assertEqual(user_fraction, 0.75)
        self.assertAlmostEqual(request_fraction, 1100 / 1200)

    def test_qps_uses_weighted_latency_and_ssd_baselines(self):
        common = dict(chip="910C", model="small", seq_len=4096, batch_size=4)
        cases = [
            self.case(**common, user="cold", method="Full_Recompute", latency_us=400),
            self.case(**common, user="cold", method="Full_Cache", latency_us=300),
            self.case(**common, user="hot", method="w_AR", latency_us=100),
            self.case(**common, user="cold", method="w_AR", latency_us=500),
        ]
        values = {
            (c.chip, c.model, c.seq_len, c.batch_size, c.user, c.method): c
            for c in cases
        }
        # Exactly half the interactions are hot.
        rows, _ = ANALYZER.analyze(
            values, activity=[1, 1], capacity_bytes=1,
            user_scale=1, bytes_per_element=1 / (2 * 4 * 256 * (2048 + 81)),
        )
        row = rows[0]
        self.assertAlmostEqual(row["weighted_latency_us"], 300)
        self.assertAlmostEqual(row["weighted_qps"], 4e6 / 300)
        self.assertAlmostEqual(row["speedup_vs_re_ssd"], 400 / 300)
        self.assertAlmostEqual(row["speedup_vs_ca_ssd"], 1.0)


if __name__ == "__main__":
    unittest.main()
