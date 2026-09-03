import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import npu_config
import item_kv_cost_model as paper
import recompute_ratio_cost_model_new as estimator
import search_npu_reconfiguration as search


class NpuConfigTest(unittest.TestCase):
    def test_final_configs_use_one_64x64_simulator_core_per_cube(self):
        expected = {
            "910A": (1000, 32, 2048, 1200.0),
            "910B": (1800, 24, 4096, 1600.0),
            "910C": (1800, 48, 4096, 3200.0),
            "MTIA2": (1350, 16, 2048, 204.8),
        }
        for chip, (frequency, cores, vector_width, bandwidth) in expected.items():
            config = json.loads(
                (REPO / "configs" / f"{chip}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["metadata"]["name"], chip)
            self.assertEqual(config["core_freq"], frequency)
            self.assertEqual(config["num_cores"], cores)
            self.assertEqual(len(config["core_config"]), cores)
            for core in config["core_config"].values():
                self.assertEqual((core["core_width"], core["core_height"]), (64, 64))
                self.assertEqual(core["vector_process_bit"], vector_width)
            self.assertEqual(
                config["metadata"]["derived_hbm_bandwidth_GBps"], bandwidth
            )

    def test_baseline_resource_budgets_use_physical_core_counts(self):
        expected = {
            "910A": (32, 32, 2048),
            "910B": (24, 24, 4096),
            "910C": (48, 48, 4096),
        }
        for chip, (cube_cores, vector_cores, vector_width) in expected.items():
            baseline = json.loads(
                (REPO / "configs" / f"{chip}.json").read_text(encoding="utf-8")
            )
            point = npu_config.baseline_design(baseline)
            self.assertEqual(
                point,
                npu_config.DesignPoint(cube_cores, vector_cores, vector_width),
            )
            usage = npu_config.resource_usage(point)
            self.assertAlmostEqual(
                usage.area_mm2,
                cube_cores * 2.57 + vector_cores * 0.70 * vector_width / 2048,
            )
            self.assertAlmostEqual(
                usage.power_w,
                cube_cores * 3.13 + vector_cores * 0.46 * vector_width / 2048,
            )

    def test_baseline_design_materializes_at_original_peak_throughput(self):
        for chip in ("910A", "910B", "910C"):
            baseline = json.loads(
                (REPO / "configs" / f"{chip}.json").read_text(encoding="utf-8")
            )
            point = npu_config.baseline_design(baseline)
            _, mapping = npu_config.materialize_config(baseline, point)
            expected = baseline["metadata"]["derived_fp16_tflops"] * 1e12
            self.assertAlmostEqual(
                mapping["actual_cube_flops"] / expected, 1.0, places=12
            )

    def test_materialized_config_tracks_target_throughput(self):
        baseline = json.loads(
            (REPO / "configs" / "910A.json").read_text(encoding="utf-8")
        )
        point = npu_config.DesignPoint(31, 47, 3072)
        generated, mapping = npu_config.materialize_config(baseline, point)
        self.assertEqual(generated["num_cores"], 8)
        self.assertLess(abs(mapping["cube_relative_error"]), 0.002)
        self.assertLess(abs(mapping["vector_relative_error"]), 1e-12)
        self.assertEqual(
            generated["metadata"]["npu_reconfiguration"]["physical_design"],
            {"cube_cores": 31, "vector_cores": 47, "vector_width_bits": 3072},
        )


class IntegerRecomputeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.calibration = json.loads(
            (REPO / "configs" / "item_kv_calib.json").read_text(encoding="utf-8")
        )
        self.config = REPO / "configs" / "910A.json"
        self.hw = estimator.derive_hardware(self.config)
        self.workload = search.Workload("small", 4, 256, 4096, 1, "cold")
        self.profile = search.hardware_profile(
            self.calibration, "910A", "ssd", 1
        )

    def test_crossing_search_matches_exhaustive_integer_search(self):
        cost = search.WorkloadCost(self.workload, self.hw, self.profile)
        cube_rates = np.asarray([64e12, 128e12, 256e12, 512e12])
        vector_rates = np.asarray([1e12, 2e12, 4e12, 8e12])

        selected_k, selected_latency = cost.select(cube_rates, vector_rates)
        for index, (cube_rate, vector_rate) in enumerate(
            zip(cube_rates, vector_rates)
        ):
            npu = (
                cost.cube_numerator / cube_rate
                + cost.vec_numerator / vector_rate
                + cost.fixed_npu
            )
            latency = np.maximum(cost.memory, npu)
            brute_k = int(np.argmin(latency))
            self.assertEqual(int(selected_k[index]), brute_k)
            self.assertAlmostEqual(float(selected_latency[index]), latency[brute_k])

    def test_vectorized_terms_equal_shared_paper_implementation(self):
        cost = search.WorkloadCost(self.workload, self.hw, self.profile)
        original_actions = self.workload.sequence // 2
        action_reuse = estimator.action_reuse_ratio_from_total(
            self.workload.sequence, original_actions, search.KV_REUSE_RATIO
        )
        item_count = (self.workload.sequence + 1) // 2
        action_count = estimator.compressed_rows_for_ratio(
            original_actions, action_reuse
        )
        workload = paper.ItemKVCostWorkload(
            self.workload.sequence, item_count, action_count, 128,
            self.workload.hidden, self.workload.layers, self.hw["s"],
            self.workload.batch, ar_reduces_attention_compute=True,
        )
        rates = paper.ItemKVHardwareRates(
            b_kv=self.hw["B_ssd"], b_emb=self.hw["B_ssd"],
            b_core=self.hw["B_core"], f_cube=self.hw["F_cube"],
            f_vec=self.hw["F_vec"],
            **self.profile,
        )
        for k in (0, 1, 137, item_count):
            expected = paper.item_kv_cost_terms(k, workload, rates)
            actual_npu = (
                cost.cube_numerator[k] / self.hw["F_cube"]
                + cost.vec_numerator[k] / self.hw["F_vec"]
                + cost.fixed_npu[k]
            )
            self.assertAlmostEqual(cost.memory[k], expected["T_mem_s"])
            self.assertAlmostEqual(actual_npu, expected["T_npu_s"])


if __name__ == "__main__":
    unittest.main()
