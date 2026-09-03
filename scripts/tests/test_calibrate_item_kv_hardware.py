import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import calibrate_item_kv_hardware as calibration  # noqa: E402


class E2ECalibrationDecisionTest(unittest.TestCase):
    def make_curve(self):
        # Columns are constant, linear, quadratic coefficients in k.
        polynomials = np.asarray([
            [10.0, -1.0, 0.0],
            [0.0, 0.2, 0.0],
            [1.0, 0.5, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        ratios = np.linspace(0.0, 1.0, 11)
        measured = 1.0 + np.square(ratios - 0.4)
        components = np.asarray([
            [sum(polynomials[index, power] * k**power for power in range(3))
             for index in range(5)]
            for k in range(11)
        ])
        return calibration.Curve(
            context=("910A", "small", 20, 1, "hot"),
            ratios=ratios,
            measured_s=measured,
            components_s=components,
            component_polynomials=polynomials,
            max_k=10,
            work_polynomials=polynomials,
            raw_rates=np.ones(5),
            request_counts=np.ones(5),
        )

    def test_closed_form_candidates_equal_exhaustive_integer_argmin(self):
        curve = self.make_curve()
        log_eta = np.log(np.asarray([0.7, 0.4, 0.8, 0.6, 0.5]))
        selected_k, selected_value = calibration.optimal_integer_k(log_eta, curve)

        inverse = np.exp(-log_eta)
        poly = curve.component_polynomials
        memory = poly[0] * inverse[0] + poly[1] * inverse[1]
        npu = poly[2] * inverse[2] + poly[3] * inverse[3] + poly[4] * inverse[4]
        exhaustive = []
        for k in range(curve.max_k + 1):
            powers = np.asarray([1.0, float(k), float(k * k)])
            exhaustive.append(max(memory @ powers, npu @ powers))
        self.assertEqual(selected_k, int(np.argmin(exhaustive)))
        self.assertAlmostEqual(selected_value, min(exhaustive))

    def test_validation_reports_geometric_mean_e2e(self):
        curve = self.make_curve()
        key = ("910A", "ddr", 1)
        result = calibration.validation(
            {key: [curve]}, {key: np.ones(5, dtype=float)}
        )
        self.assertIn("geometric_mean_cost_model_e2e_us", result)
        self.assertIn("geometric_mean_selected_e2e_us", result)
        self.assertIn("geometric_mean_grid_oracle_e2e_us", result)
        self.assertGreaterEqual(result["geometric_mean_e2e_regret_factor"], 1.0)

    def test_common_hardware_rate_scale_matches_geometric_mean_without_changing_k(self):
        curve = self.make_curve()
        eta, _, scale = calibration.fit_hardware_eta(
            [curve], random_samples=20, e2e_weight=1.0,
            grid_optimum_weight=0.05, absolute_weight=0.0,
            regularization=0.0, seed=3,
        )
        selected_after, _ = calibration.optimal_integer_k(np.log(eta), curve)
        selected_before, _ = calibration.optimal_integer_k(
            np.log(eta / scale), curve
        )
        self.assertEqual(selected_after, selected_before)

        key = ("910A", "ddr", 1)
        result = calibration.validation({key: [curve]}, {key: eta})
        self.assertAlmostEqual(
            result["geometric_mean_cost_model_e2e_us"],
            result["geometric_mean_selected_e2e_us"],
        )

    def test_extended_integer_search_includes_zero_startup_boundary(self):
        curve = self.make_curve()
        scales = calibration.extension_scales([curve])
        log_eta = np.zeros(5)
        log_saturation = np.full(5, -4.0)
        log_startup = np.asarray([-8.0, 2.0, -8.0, -8.0, -8.0])
        selected, value = calibration.optimal_integer_k_extended(
            log_eta, log_saturation, log_startup, scales, curve
        )
        exhaustive = [
            calibration.extended_time_at_k(
                log_eta, log_saturation, log_startup, scales, curve, k
            )
            for k in range(curve.max_k + 1)
        ]
        self.assertEqual(selected, int(np.argmin(exhaustive)))
        self.assertAlmostEqual(value, min(exhaustive))


if __name__ == "__main__":
    unittest.main()
