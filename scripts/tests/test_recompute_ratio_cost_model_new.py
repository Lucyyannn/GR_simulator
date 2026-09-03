import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "recompute_ratio_cost_model_new.py"
SPEC = importlib.util.spec_from_file_location("recompute_ratio_cost_model_new", SCRIPT)
MODEL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODEL)


class ChipCalibrationLookupTest(unittest.TestCase):
    def test_current_chip_names_use_builtin_compute_correction(self):
        expected = {"910A": 4.13, "910B": 5.74, "910C": 11.27}
        for chip, expected_factor in expected.items():
            factor, detail = MODEL.ir_compute_correction_factor({}, chip, 256)
            self.assertEqual(factor, expected_factor)
            self.assertEqual(detail["mode"], "chip_hidden_power")

    def test_configured_compute_correction_precedes_builtin_default(self):
        calibration = {
            "ir_cost_model": {
                "compute_correction": {"910A": 3.0}
            }
        }
        factor, detail = MODEL.ir_compute_correction_factor(
            calibration, "910A", 256
        )
        self.assertEqual(factor, 3.0)
        self.assertEqual(detail["source"], "ir_cost_model.compute_correction")

    def test_legacy_task2_context_is_restricted_to_910c_family(self):
        calibration = {
            "task2_contexts": {
                "cold": {"256": {"4096": {"1": {"marker": "legacy-c"}}}}
            }
        }
        args = ("cold", 256, 4096, 1)
        self.assertEqual(
            MODEL.lookup_task2_context(calibration, "910C", *args)["marker"],
            "legacy-c",
        )
        self.assertEqual(
            MODEL.lookup_task2_context(calibration, "910C", *args)["marker"],
            "legacy-c",
        )
        self.assertEqual(MODEL.lookup_task2_context(calibration, "910A", *args), {})
        self.assertEqual(MODEL.lookup_task2_context(calibration, "910B", *args), {})

    def test_chip_scoped_task2_context_supports_exact_and_family_keys(self):
        calibration = {
            "task2_contexts": {
                "chips": {
                    "910A": {
                        "hot": {"512": {"8192": {"4": {"marker": "a"}}}}
                    },
                    "910B": {
                        "hot": {"512": {"8192": {"4": {"marker": "b-new"}}}}
                    },
                }
            }
        }
        args = ("hot", 512, 8192, 4)
        self.assertEqual(
            MODEL.lookup_task2_context(calibration, "910A", *args)["marker"], "a"
        )
        self.assertEqual(
            MODEL.lookup_task2_context(calibration, "910B", *args)["marker"],
            "b-new",
        )
        self.assertEqual(MODEL.lookup_task2_context(calibration, "910C", *args), {})


if __name__ == "__main__":
    unittest.main()
