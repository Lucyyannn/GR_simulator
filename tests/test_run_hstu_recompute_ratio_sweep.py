import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts" / "run_hstu_recompute_ratio_sweep.py"
SPEC = importlib.util.spec_from_file_location("recompute_ratio_sweep", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def flag_value(command, flag):
    return command[command.index(flag) + 1]


class RecomputeRatioSweepTest(unittest.TestCase):
    def test_supported_matrix_has_4224_unique_cases(self):
        cases = RUNNER.matrix_cases(
            RUNNER.CHIPS,
            RUNNER.MODELS,
            RUNNER.SEQS,
            RUNNER.BATCHES,
            RUNNER.USERS,
            RUNNER.RATIOS,
        )
        self.assertEqual(len(cases), 4224)
        self.assertEqual(len({case.case_id for case in cases}), 4224)
        self.assertEqual(len({str(RUNNER.case_dir(Path("out"), case)) for case in cases}), 4224)
        self.assertEqual(
            set(RUNNER.CHIPS), {"910A", "910B", "910C", "MTIA2"}
        )

    def test_item_ratio_uses_deterministic_half_up(self):
        # Odd item counts exercise exact .5 ties: 5 * 0.1 = 0.5 -> 1.
        self.assertEqual(RUNNER.round_half_up(0.5), 1)
        self.assertEqual(RUNNER.round_half_up(1.5), 2)
        case = RUNNER.Case("910A", "small", 9, 1, "cold", 0.1)
        self.assertEqual(case.item_count, 5)
        self.assertEqual(case.history_recompute_len, 1)
        self.assertAlmostEqual(case.actual_ratio, 0.2)

        real = RUNNER.Case("910A", "small", 4096, 1, "cold", 0.1)
        self.assertEqual(real.item_count, 2048)
        self.assertEqual(real.history_recompute_len, 205)
        self.assertAlmostEqual(real.actual_ratio, 205 / 2048)
        full = RUNNER.Case("910A", "small", 4096, 1, "cold", 1.0)
        self.assertEqual(full.history_recompute_len, full.item_count)

    def test_command_forces_ar_and_speedup_comparison_sources(self):
        for user, history_source in (("hot", "ddr"), ("cold", "ssd")):
            case = RUNNER.Case("910B", "middle", 8192, 4, user, 0.7)
            command = RUNNER.build_command(
                Path("results/sweep"), RUNNER.CHIP_CONFIGS[case.chip], "warn", case
            )
            self.assertEqual(flag_value(command, "--source-medium"), history_source)
            self.assertEqual(flag_value(command, "--embedding-source-medium"), "ssd")
            self.assertEqual(
                flag_value(command, "--history-recompute-source-medium"),
                history_source,
            )
            self.assertEqual(
                int(flag_value(command, "--history-recompute-len")),
                case.history_recompute_len,
            )
            self.assertIn("--enable-kv-reuse", command)
            self.assertEqual(
                float(flag_value(command, "--kv-reuse-ratio")),
                RUNNER.KV_REUSE_RATIO,
            )
            self.assertIn("--enable-ar-reduce-attention-compute", command)
            self.assertNotIn("--disable-ar-reduce-attention-compute", command)

    def test_ratio_label_distinguishes_every_requested_point(self):
        cases = [
            RUNNER.Case("910C", "large", 16384, 8, "hot", ratio)
            for ratio in RUNNER.RATIOS
        ]
        self.assertEqual(
            [case.ratio_label for case in cases],
            [f"r{value:03d}" for value in range(0, 101, 10)],
        )
        self.assertEqual(len({case.case_id for case in cases}), 11)
        self.assertEqual(len({RUNNER.case_dir(Path("root"), case) for case in cases}), 11)

    def test_max_concurrency_constant_and_single_thread_environment(self):
        self.assertEqual(RUNNER.MAX_CONCURRENT, 196)
        self.assertEqual(RUNNER.DEFAULT_MAX_CONCURRENT, 48)
        for value in RUNNER.SINGLE_THREAD_ENV.values():
            self.assertEqual(value, "1")

    def test_run_case_sets_unique_lock_and_resumes_on_matching_digest(self):
        case = RUNNER.Case("910C", "small", 4096, 1, "hot", 0.3)
        captured_env = {}

        def fake_run(command, **kwargs):
            captured_env.update(kwargs["env"])
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory(dir=REPO) as temp, \
             mock.patch.object(RUNNER, "input_digests", return_value={"x": "abc"}), \
             mock.patch.object(RUNNER, "completed_outputs", return_value=True), \
             mock.patch.object(RUNNER.subprocess, "run", side_effect=fake_run) as run:
            root = Path(temp)
            first = RUNNER.run_case(
                REPO, root, RUNNER.CHIP_CONFIGS[case.chip], "warn", case, False
            )
            self.assertEqual(first["returncode"], 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(
                captured_env["SIMULATOR_SLOT_LOCK"], f"/tmp/{case.case_id}.lock"
            )
            for variable, value in RUNNER.SINGLE_THREAD_ENV.items():
                self.assertEqual(captured_env[variable], value)

            second = RUNNER.run_case(
                REPO, root, RUNNER.CHIP_CONFIGS[case.chip], "warn", case, False
            )
            self.assertEqual(second["returncode"], 0)
            self.assertEqual(run.call_count, 1, "matching successful case must resume")


if __name__ == "__main__":
    unittest.main()
