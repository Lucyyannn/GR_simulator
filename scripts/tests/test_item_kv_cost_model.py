import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FORMULA = load("item_kv_cost_model", SCRIPTS / "item_kv_cost_model.py")
ESTIMATOR = load(
    "recompute_ratio_cost_model_new_for_paper_test",
    SCRIPTS / "recompute_ratio_cost_model_new.py",
)


class PaperFormulaTest(unittest.TestCase):
    def test_terms_are_exact_paper_equations_with_batch_extension(self):
        workload = FORMULA.ItemKVCostWorkload(
            original_history_tokens=20,
            item_kv_after_akr=10,
            action_kv_after_akr=3,
            candidates_per_user=4,
            hidden=8,
            layers=2,
            bytes_per_element=2,
            batch_size=3,
        )
        hardware = FORMULA.ItemKVHardwareRates(
            b_kv=100.0,
            b_emb=200.0,
            b_core=400.0,
            f_cube=800.0,
            f_vec=1600.0,
        )
        terms = FORMULA.item_kv_cost_terms(2, workload, hardware)

        self.assertEqual(terms["kv_bytes"], 2 * 2 * 3 * (3 + 10 - 2) * 8 * 2)
        self.assertEqual(terms["embedding_bytes"], 3 * 2 * 8 * 2)
        self.assertEqual(
            terms["cube_flops"],
            2 * 3 * (8 * 4 * 8**2 + 4 * 4 * 20 * 8 + 8 * 2 * 8**2 + 4 * 2**2 * 8),
        )
        self.assertEqual(
            terms["vector_ops"],
            2 * 3 * (2 * (4 * 20 + 2**2) + 2 * (4 + 2) * 8),
        )
        self.assertEqual(
            terms["core_bytes"],
            2 * 3 * 2 * ((4 + 2) * 8 + 2 * 13 * 8 + 4 * 2 * 8),
        )
        self.assertAlmostEqual(
            terms["T_mem_s"],
            terms["kv_bytes"] / 100.0 + terms["embedding_bytes"] / 200.0,
        )
        self.assertAlmostEqual(
            terms["T_npu_s"],
            terms["cube_flops"] / 800.0
            + terms["vector_ops"] / 1600.0
            + terms["core_bytes"] / 400.0,
        )
        self.assertEqual(terms["T_s"], max(terms["T_mem_s"], terms["T_npu_s"]))

    def test_batch_multiplies_independent_work_not_k_squared_across_users(self):
        one = FORMULA.ItemKVCostWorkload(20, 10, 3, 4, 8, 2, 2, 1)
        four = FORMULA.ItemKVCostWorkload(20, 10, 3, 4, 8, 2, 2, 4)
        hardware = FORMULA.ItemKVHardwareRates(100, 200, 400, 800, 1600)
        one_terms = FORMULA.item_kv_cost_terms(2, one, hardware)
        four_terms = FORMULA.item_kv_cost_terms(2, four, hardware)
        for key in (
            "kv_bytes", "embedding_bytes", "cube_flops", "vector_ops",
            "core_bytes", "T_mem_s", "T_npu_s", "T_s",
        ):
            self.assertAlmostEqual(four_terms[key], 4 * one_terms[key])

    def test_ar_reduction_uses_post_reuse_attention_length(self):
        baseline = FORMULA.ItemKVCostWorkload(20, 10, 3, 4, 8, 2, 2, 1)
        reduced = FORMULA.ItemKVCostWorkload(
            20, 10, 3, 4, 8, 2, 2, 1,
            ar_reduces_attention_compute=True,
        )
        hardware = FORMULA.ItemKVHardwareRates(100, 200, 400, 800, 1600)
        baseline_terms = FORMULA.item_kv_cost_terms(2, baseline, hardware)
        reduced_terms = FORMULA.item_kv_cost_terms(2, reduced, hardware)
        self.assertEqual(baseline_terms["attention_history_tokens"], 20)
        self.assertEqual(reduced_terms["attention_history_tokens"], 13)
        self.assertLess(reduced_terms["cube_flops"], baseline_terms["cube_flops"])
        self.assertLess(reduced_terms["vector_ops"], baseline_terms["vector_ops"])

    def test_integer_search_is_not_restricted_to_tenth_grid(self):
        workload = FORMULA.ItemKVCostWorkload(200, 100, 20, 8, 16, 2, 2, 1)
        hardware = FORMULA.ItemKVHardwareRates(23e6, 23e6, 1e8, 1e9, 1e8)
        selected = FORMULA.select_optimal_item_recompute(workload, hardware)
        self.assertEqual(selected["k"], 14)
        self.assertAlmostEqual(selected["recompute_ratio"], 0.14)
        self.assertNotEqual(round(selected["recompute_ratio"] * 10), selected["recompute_ratio"] * 10)

    def test_saturation_and_startup_are_clean_hardware_path_terms(self):
        path = FORMULA.SaturatingHardwarePath(
            peak_rate=100.0, saturation_work=20.0, startup_s=0.25
        )
        total, body, startup = path.time_s(total_work=80.0, request_count=2)
        achieved = 100.0 * (1.0 - __import__("math").exp(-40.0 / 20.0))
        self.assertAlmostEqual(body, 80.0 / achieved)
        self.assertAlmostEqual(startup, 0.5)
        self.assertAlmostEqual(total, body + startup)

        disabled = FORMULA.SaturatingHardwarePath(100.0)
        self.assertEqual(disabled.time_s(80.0, 2), (0.8, 0.8, 0.0))


class HardwareProfileTest(unittest.TestCase):
    def test_profile_is_scoped_only_by_chip_medium_and_batch(self):
        section = {
            "hardware_profiles": {
                "chips": {
                    "910A": {
                        "default": {"eta_cube": 0.8},
                        "batches": {"4": {"F_cube": 123.0}},
                        "media": {
                            "ssd": {
                                "default": {"eta_kv": 0.5},
                                "batches": {"4": {"B_kv": 456.0}},
                            }
                        },
                    }
                }
            }
        }
        values = ESTIMATOR._paper_profile_values(section, "910A", "ssd", 4)
        self.assertEqual(values, {
            "eta_cube": 0.8, "F_cube": 123.0,
            "eta_kv": 0.5, "B_kv": 456.0,
        })

    def test_profile_rejects_workload_specific_coefficients(self):
        section = {
            "hardware_profiles": {
                "910A": {"default": {"hidden_256_scale": 2.0}}
            }
        }
        with self.assertRaisesRegex(ValueError, "disallowed profile keys"):
            ESTIMATOR._paper_profile_values(section, "910A", "ssd", 1)

    def test_profile_accepts_hardware_saturation_and_startup_only(self):
        section = {
            "hardware_profiles": {
                "910A": {"default": {
                    "kv_saturation_bytes": 4096.0,
                    "kv_startup_s": 2e-6,
                }}
            }
        }
        self.assertEqual(
            ESTIMATOR._paper_profile_values(section, "910A", "ssd", 1),
            {"kv_saturation_bytes": 4096.0, "kv_startup_s": 2e-6},
        )

    def test_calibration_profile_cannot_override_raw_hardware_rate(self):
        section = {
            "hardware_profiles": {
                "910A": {"default": {"B_kv": 123.0}}
            }
        }
        with self.assertRaisesRegex(ValueError, "disallowed profile keys"):
            ESTIMATOR._paper_profile_values(
                section, "910A", "ssd", 1, allow_rate_inputs=False
            )

    def test_all_five_rates_and_efficiencies_are_cli_inputs(self):
        config = SCRIPTS.parent / "configs" / "910A.json"
        command = [
            sys.executable, str(SCRIPTS / "recompute_ratio_cost_model_new.py"),
            "--config", str(config), "--calibration", "/does/not/exist.json",
            "--user", "hot", "--layers", "1", "--hidden", "16",
            "--kv-len", "20", "--batch-size", "4", "--candidates", "8",
            "--fixed-recompute-len", "3",
            "--b-kv", "101", "--b-emb", "102", "--b-core", "103",
            "--f-cube", "104", "--f-vec", "105",
            "--eta-kv", "0.51", "--eta-emb", "0.52",
            "--eta-core", "0.53", "--eta-cube", "0.54",
            "--eta-vec", "0.55", "--field", "json",
            "--kv-saturation-bytes", "201", "--emb-saturation-bytes", "202",
            "--core-saturation-bytes", "203", "--cube-saturation-flops", "204",
            "--vec-saturation-ops", "205", "--kv-startup-s", "0.001",
            "--emb-startup-s", "0.002", "--core-startup-s", "0.003",
            "--cube-startup-s", "0.004", "--vec-startup-s", "0.005",
        ]
        payload = json.loads(subprocess.run(
            command, check=True, text=True, capture_output=True
        ).stdout)
        self.assertEqual(payload["hardware_inputs"], {
            "B_kv_Bps": 101.0,
            "B_emb_Bps": 102.0,
            "B_core_Bps": 103.0,
            "F_cube_FLOPs": 104.0,
            "F_vec_ops": 105.0,
        })
        self.assertEqual(payload["hardware_efficiency"], {
            "eta_kv": 0.51,
            "eta_emb": 0.52,
            "eta_core": 0.53,
            "eta_cube": 0.54,
            "eta_vec": 0.55,
        })
        self.assertEqual(payload["batch_size"], 4)
        self.assertEqual(payload["hardware_saturation"], {
            "kv_saturation_bytes": 201.0,
            "emb_saturation_bytes": 202.0,
            "core_saturation_bytes": 203.0,
            "cube_saturation_flops": 204.0,
            "vec_saturation_ops": 205.0,
        })
        self.assertEqual(payload["hardware_startup_s"], {
            "kv_startup_s": 0.001,
            "emb_startup_s": 0.002,
            "core_startup_s": 0.003,
            "cube_startup_s": 0.004,
            "vec_startup_s": 0.005,
        })


if __name__ == "__main__":
    unittest.main()
