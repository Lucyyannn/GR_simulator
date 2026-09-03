import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts" / "run_meta_hstu_full_matrix.py"
SPEC = importlib.util.spec_from_file_location("full_matrix_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class FullMatrixRunnerTest(unittest.TestCase):
    def test_supported_matrix_includes_mtia2_and_new_shapes(self):
        cases = RUNNER.matrix_cases(
            RUNNER.CHIPS,
            RUNNER.MODELS,
            RUNNER.SEQS,
            RUNNER.BATCHES,
            RUNNER.USERS,
            RUNNER.METHODS,
        )
        self.assertEqual(len(cases), 1536)
        self.assertEqual(
            set(RUNNER.CHIPS), {"910A", "910B", "910C", "MTIA2"}
        )
        self.assertEqual(
            set(RUNNER.METHODS),
            {"Full_Recompute", "Full_Cache", "w_AR", "w_both"},
        )
        for case in cases:
            self.assertEqual(
                RUNNER.config_for_case(None, case),
                Path(f"configs/{case.chip}.json"),
            )

    def test_command_uses_speedup_comparison_embedding_sources(self):
        for user, history_source in (("hot", "ddr"), ("cold", "ssd")):
            case = RUNNER.Case(
                "910A", "small", 4096, 1, user, "Full_Cache"
            )
            command, _ = RUNNER.build_command(
                REPO,
                Path("results/test"),
                RUNNER.CHIP_CONFIGS[case.chip],
                Path("scripts/recompute_ratio_calibration.json"),
                "warn",
                case,
            )
            self.assertEqual(
                command[command.index("--embedding-source-medium") + 1], "ssd"
            )
            self.assertEqual(
                command[command.index("--source-medium") + 1], history_source
            )
            self.assertEqual(
                command[command.index("--history-recompute-source-medium") + 1],
                history_source,
            )

    def test_per_chip_generated_config_override(self):
        case = RUNNER.Case(
            "910B", "small", 4096, 1, "hot", "w_both"
        )
        overrides = RUNNER.parse_chip_config_overrides(
            ["910A=generated/a.json", "910B=generated/b.json"]
        )
        self.assertEqual(
            RUNNER.config_for_case(None, case, overrides),
            Path("generated/b.json"),
        )

    def test_ar_methods_enable_attention_compute_reduction(self):
        for method in ("w_AR", "w_both"):
            case = RUNNER.Case("910A", "small", 4096, 1, "hot", method)
            with mock.patch.object(
                RUNNER, "compute_k", return_value=(0, {"history_recompute_len": 0})
            ):
                command, _ = RUNNER.build_command(
                    REPO, Path("results/test"), RUNNER.CHIP_CONFIGS[case.chip],
                    Path("configs/item_kv_calib.json"), "warn", case,
                )
            self.assertIn("--enable-ar-reduce-attention-compute", command)
            self.assertNotIn("--disable-ar-reduce-attention-compute", command)

    def test_run_case_forces_single_thread_libraries_and_unique_lock(self):
        case = RUNNER.Case(
            "910C", "small", 4096, 1, "hot", "Full_Cache"
        )
        captured_env = {}

        def fake_run(*args, **kwargs):
            captured_env.update(kwargs["env"])
            return subprocess.CompletedProcess(args[0], 0)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            RUNNER.subprocess, "run", side_effect=fake_run
        ), mock.patch.object(RUNNER, "input_digests", return_value={}), \
             mock.patch.object(
                 RUNNER, "completed_outputs_match_method", return_value=True
             ):
            record = RUNNER.run_case(
                REPO,
                Path(temp_dir),
                RUNNER.CHIP_CONFIGS[case.chip],
                Path("scripts/recompute_ratio_calibration.json"),
                "warn",
                case,
                False,
            )

        self.assertEqual(record["returncode"], 0)
        self.assertEqual(
            captured_env["SIMULATOR_SLOT_LOCK"], f"/tmp/{case.case_id}.lock"
        )
        for variable, value in RUNNER.SINGLE_THREAD_ENV.items():
            self.assertEqual(captured_env[variable], value)


if __name__ == "__main__":
    unittest.main()
