#!/usr/bin/env python3
"""Fit hardware-only efficiency profiles for the explicit item-KV formula.

The output profile is scoped only by chip, storage medium, and batch size.  It
contains eta_kv/eta_emb/eta_core/eta_cube/eta_vec and cannot alter any formula
numerator or introduce model/hidden/sequence/ratio-specific coefficients.

The primary objective is end-to-end decision quality: select ``k`` with the
paper cost model, interpolate the measured ratio sweep at that ratio, and
minimize the geometric mean E2E regret over configurations.  The measured
0.1-grid optimum is auxiliary supervision, not the final decision rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import recompute_ratio_cost_model_new as estimator


MODELS = {"small": (4, 256), "middle": (8, 512), "large": (12, 1024)}
ETA_NAMES = ("eta_kv", "eta_emb", "eta_cube", "eta_vec", "eta_core")
SATURATION_NAMES = (
    "kv_saturation_bytes", "emb_saturation_bytes", "cube_saturation_flops",
    "vec_saturation_ops", "core_saturation_bytes",
)
STARTUP_NAMES = (
    "kv_startup_s", "emb_startup_s", "cube_startup_s",
    "vec_startup_s", "core_startup_s",
)


@dataclass(frozen=True)
class Curve:
    context: tuple[str, str, int, int, str]
    ratios: np.ndarray
    measured_s: np.ndarray
    components_s: np.ndarray
    component_polynomials: np.ndarray
    max_k: int
    work_polynomials: np.ndarray | None = None
    raw_rates: np.ndarray | None = None
    request_counts: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=10000)
    parser.add_argument("--e2e-weight", type=float, default=1.0)
    parser.add_argument("--grid-optimum-weight", type=float, default=0.05)
    parser.add_argument("--absolute-weight", type=float, default=0.0)
    parser.add_argument("--curve-shape-weight", type=float, default=0.02)
    parser.add_argument("--extension-random-samples", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--regularization", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def sim_time_us(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(
            item for item in csv.DictReader(handle)
            if item.get("scope") == "npu_total"
        )
    return float(row["sim_time_us"])


def load_complete_curves(root: Path) -> dict[tuple[str, str, int], list[Curve]]:
    definition_path = root / "matrix_definition.json"
    configured_hardware = {}
    if definition_path.is_file():
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        configured_hardware = {
            chip: Path(path)
            for chip, path in definition.get("chip_configs", {}).items()
        }
    points: dict[tuple[str, str, int, int, str], list[dict]] = defaultdict(list)
    for status_path in sorted((root / "logs").glob("*.status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("returncode") != 0 or status.get("dry_run", False):
            continue
        case_dir = (
            root / "cases" / status["chip"] / "AR_IR_ratio"
            / status["ratio_label"]
            / f"HSTU-{status['model']}_seq{status['seq_len']}"
              f"_bs{status['batch_size']}_{status['user']}"
        )
        activity = case_dir / "compute_activity.csv"
        if not activity.is_file():
            continue
        key = (
            status["chip"], status["model"], int(status["seq_len"]),
            int(status["batch_size"]), status["user"],
        )
        points[key].append({
            "ratio": float(status["actual_ratio"]),
            "k": int(status["history_recompute_len"]),
            "measured_s": sim_time_us(activity) * 1e-6,
        })

    output: dict[tuple[str, str, int], list[Curve]] = defaultdict(list)
    for context, rows in sorted(points.items()):
        if len(rows) != 11:
            continue
        rows.sort(key=lambda row: row["ratio"])
        chip, model, sequence, batch, user = context
        layers, hidden = MODELS[model]
        hw = estimator.derive_hardware(
            configured_hardware.get(chip, Path(f"configs/{chip}.json"))
        )
        item_count = (sequence + 1) // 2
        original_actions = sequence // 2
        action_reuse = estimator.action_reuse_ratio_from_total(
            sequence, original_actions, 0.4802
        )
        retained_actions = estimator.compressed_rows_for_ratio(
            original_actions, action_reuse
        )
        workload = estimator.ItemKVCostWorkload(
            original_history_tokens=sequence,
            item_kv_after_akr=item_count,
            action_kv_after_akr=retained_actions,
            candidates_per_user=128,
            hidden=hidden,
            layers=layers,
            bytes_per_element=hw["s"],
            batch_size=batch,
            ar_reduces_attention_compute=True,
        )
        medium = "ddr" if user == "hot" else "ssd"
        rates = estimator.ItemKVHardwareRates(
            b_kv=hw[f"B_{medium}"],
            b_emb=hw[f"B_{medium}"],
            b_core=hw["B_core"],
            f_cube=hw["F_cube"],
            f_vec=hw["F_vec"],
        )
        components = []
        for row in rows:
            terms = estimator.item_kv_cost_terms(row["k"], workload, rates)
            # Columns match ETA_NAMES: KV, embedding, Cube, Vector, core.
            components.append([
                terms["kv_time_s"], terms["embedding_time_s"],
                terms["T_cube_s"], terms["T_vec_s"], terms["T_core_s"],
            ])
        # Every formula component is at most quadratic in k.  Keeping these
        # exact polynomial coefficients lets calibration use the same
        # all-integer-k decision as the production cost model without reducing
        # the decision to the measured 0.1 grid.
        samples = []
        work_samples = []
        for k in (0, 1, 2):
            terms = estimator.item_kv_cost_terms(k, workload, rates)
            samples.append(np.asarray([
                terms["kv_time_s"], terms["embedding_time_s"],
                terms["T_cube_s"], terms["T_vec_s"], terms["T_core_s"],
            ]))
            work_samples.append(np.asarray([
                terms["kv_bytes"], terms["embedding_bytes"],
                terms["cube_flops"], terms["vector_ops"], terms["core_bytes"],
            ]))
        value0, value1, value2 = samples
        quadratic = (value2 - 2.0 * value1 + value0) / 2.0
        linear = value1 - value0 - quadratic
        polynomials = np.stack((value0, linear, quadratic), axis=1)
        work0, work1, work2 = work_samples
        work_quadratic = (work2 - 2.0 * work1 + work0) / 2.0
        work_linear = work1 - work0 - work_quadratic
        work_polynomials = np.stack((work0, work_linear, work_quadratic), axis=1)
        output[(chip, medium, batch)].append(Curve(
            context=context,
            ratios=np.asarray([row["ratio"] for row in rows]),
            measured_s=np.asarray([row["measured_s"] for row in rows]),
            components_s=np.asarray(components),
            component_polynomials=polynomials,
            max_k=item_count,
            work_polynomials=work_polynomials,
            raw_rates=np.asarray([
                rates.b_kv, rates.b_emb, rates.f_cube,
                rates.f_vec, rates.b_core,
            ]),
            request_counts=np.asarray([
                layers * batch, batch, layers * batch,
                layers * batch, layers * batch,
            ]),
        ))
    return output


def prediction(log_eta: np.ndarray, components: np.ndarray) -> np.ndarray:
    inverse_eta = np.exp(-log_eta)
    memory = (
        components[:, 0] * inverse_eta[0]
        + components[:, 1] * inverse_eta[1]
    )
    npu = (
        components[:, 2] * inverse_eta[2]
        + components[:, 3] * inverse_eta[3]
        + components[:, 4] * inverse_eta[4]
    )
    return np.maximum(memory, npu)


def optimal_integer_k(log_eta: np.ndarray, curve: Curve) -> tuple[int, float]:
    """Apply argmin_k max(T_mem(k), T_npu(k)) for every integer k.

    This is the production decision formula in polynomial form.  ``T_mem`` is
    linear and ``T_npu`` is convex quadratic, so the integer optimum must be
    adjacent to an endpoint, the NPU vertex, or a T_mem/T_npu intersection.
    """

    inverse_eta = np.exp(-log_eta)
    poly = curve.component_polynomials
    memory = poly[0] * inverse_eta[0] + poly[1] * inverse_eta[1]
    npu = (
        poly[2] * inverse_eta[2]
        + poly[3] * inverse_eta[3]
        + poly[4] * inverse_eta[4]
    )
    locations = [0.0, float(curve.max_k)]
    if npu[2] > 0.0:
        locations.append(-npu[1] / (2.0 * npu[2]))

    # Solve T_npu(k) - T_mem(k) = 0.
    a = npu[2] - memory[2]
    b = npu[1] - memory[1]
    c = npu[0] - memory[0]
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= 1e-14 * scale:
        if abs(b) > 1e-14 * scale:
            locations.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            locations.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))

    candidates: set[int] = {0, curve.max_k}
    for location in locations:
        if not math.isfinite(location):
            continue
        center = int(math.floor(location))
        for k in range(center - 2, center + 4):
            if 0 <= k <= curve.max_k:
                candidates.add(k)

    def evaluate(k: int) -> float:
        powers = np.asarray((1.0, float(k), float(k * k)))
        return float(max(memory @ powers, npu @ powers))

    return min(
        ((k, evaluate(k)) for k in candidates),
        key=lambda item: (item[1], item[0]),
    )


def measured_e2e_at_ratio(curve: Curve, ratio: float) -> float:
    """Log-linear interpolation of measured E2E between sweep points."""

    return float(np.exp(np.interp(
        ratio, curve.ratios, np.log(curve.measured_s)
    )))


def extension_scales(curves: list[Curve]) -> tuple[np.ndarray, np.ndarray]:
    """Return unit scales for dimensionless saturation/startup optimization."""

    per_request_work = []
    startup_time = []
    for curve in curves:
        assert curve.work_polynomials is not None
        assert curve.request_counts is not None
        for k in (0, curve.max_k // 2, curve.max_k):
            powers = np.asarray((1.0, float(k), float(k * k)))
            work = np.maximum(curve.work_polynomials @ powers, 0.0)
            per_request_work.append(work / curve.request_counts)
        startup_time.append(
            np.full(5, float(np.median(curve.measured_s))) / curve.request_counts
        )
    saturation_scale = np.maximum(
        np.median(np.asarray(per_request_work), axis=0), 1e-30
    )
    startup_scale = np.maximum(
        np.median(np.asarray(startup_time), axis=0), 1e-30
    )
    return saturation_scale, startup_scale


def extended_time_at_k(
    log_eta: np.ndarray,
    log_saturation_ratio: np.ndarray,
    log_startup_ratio: np.ndarray,
    scales: tuple[np.ndarray, np.ndarray],
    curve: Curve,
    k: int,
) -> float:
    """Evaluate the clean saturation/startup extension at one integer k."""

    assert curve.work_polynomials is not None
    assert curve.raw_rates is not None
    assert curve.request_counts is not None
    powers = np.asarray((1.0, float(k), float(k * k)))
    work = np.maximum(curve.work_polynomials @ powers, 0.0)
    per_request = work / curve.request_counts
    saturation = scales[0] * np.exp(log_saturation_ratio)
    startup = scales[1] * np.exp(log_startup_ratio)
    peak_rate = curve.raw_rates * np.exp(log_eta)
    utilization = -np.expm1(-per_request / np.maximum(saturation, 1e-300))
    achieved_rate = peak_rate * np.maximum(utilization, 1e-15)
    path_time = np.where(
        work > 0.0,
        work / achieved_rate + curve.request_counts * startup,
        0.0,
    )
    return float(max(path_time[0] + path_time[1], path_time[2:].sum()))


def optimal_integer_k_extended(
    log_eta: np.ndarray,
    log_saturation_ratio: np.ndarray,
    log_startup_ratio: np.ndarray,
    scales: tuple[np.ndarray, np.ndarray],
    curve: Curve,
) -> tuple[int, float]:
    """Exact local integer minimum of the unimodal extended paper objective."""

    def evaluate(k: int) -> float:
        return extended_time_at_k(
            log_eta, log_saturation_ratio, log_startup_ratio, scales, curve, k
        )

    low, high = 0, curve.max_k
    while high - low > 16:
        left = low + (high - low) // 3
        right = high - (high - low) // 3
        if evaluate(left) <= evaluate(right):
            high = right - 1
        else:
            low = left + 1
    candidates = set(range(low, high + 1))
    candidates.update((0, min(1, curve.max_k), curve.max_k))
    return min(
        ((k, evaluate(k)) for k in candidates),
        key=lambda item: (item[1], item[0]),
    )


def extended_fit_loss(
    parameters: np.ndarray,
    curves: list[Curve],
    scales: tuple[np.ndarray, np.ndarray],
    *,
    e2e_weight: float,
    grid_optimum_weight: float,
    curve_shape_weight: float,
    regularization: float,
) -> float:
    log_eta = parameters[:5]
    log_saturation = parameters[5:10]
    log_startup = parameters[10:15]
    regrets = []
    ratio_errors = []
    shape_errors = []
    for curve in curves:
        selected_k, _ = optimal_integer_k_extended(
            log_eta, log_saturation, log_startup, scales, curve
        )
        selected_ratio = selected_k / curve.max_k
        selected_measured = measured_e2e_at_ratio(curve, selected_ratio)
        measured_best = int(np.argmin(curve.measured_s))
        regrets.append(math.log(selected_measured / curve.measured_s[measured_best]))
        ratio_errors.append(selected_ratio - curve.ratios[measured_best])

        predicted_grid = np.asarray([
            extended_time_at_k(
                log_eta, log_saturation, log_startup, scales, curve,
                int(round(ratio * curve.max_k)),
            )
            for ratio in curve.ratios
        ])
        predicted_shape = np.log(predicted_grid) - np.log(predicted_grid).min()
        measured_shape = np.log(curve.measured_s) - np.log(curve.measured_s).min()
        shape_errors.extend(predicted_shape - measured_shape)

    relative_log_eta = log_eta - np.mean(log_eta)
    extension_magnitude = np.concatenate((
        np.log1p(np.exp(log_saturation)),
        np.log1p(np.exp(log_startup)),
    ))
    return float(
        e2e_weight * np.mean(regrets)
        + grid_optimum_weight * np.mean(np.square(ratio_errors))
        + curve_shape_weight * np.mean(np.square(shape_errors))
        + regularization * (
            np.mean(np.square(relative_log_eta))
            + np.mean(np.square(extension_magnitude))
        )
    )


def fit_hardware_extensions(
    curves: list[Curve],
    initial_eta: np.ndarray,
    *,
    random_samples: int,
    e2e_weight: float,
    grid_optimum_weight: float,
    curve_shape_weight: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit hardware-only saturation and startup terms, never workload factors."""

    scales = extension_scales(curves)
    rng = np.random.default_rng(seed)
    best = np.concatenate((
        np.log(initial_eta), np.full(5, -14.0), np.full(5, -14.0)
    ))
    loss_args = dict(
        e2e_weight=e2e_weight,
        grid_optimum_weight=grid_optimum_weight,
        curve_shape_weight=curve_shape_weight,
        regularization=regularization,
    )
    best_loss = extended_fit_loss(best, curves, scales, **loss_args)

    for _ in range(random_samples):
        candidate = best.copy()
        candidate[:5] += rng.normal(0.0, 1.0, size=5)
        candidate[5:10] = rng.uniform(-10.0, 2.0, size=5)
        candidate[10:15] = rng.uniform(-12.0, 0.0, size=5)
        current = extended_fit_loss(candidate, curves, scales, **loss_args)
        if current < best_loss:
            best, best_loss = candidate, current

    bounds = [(-12.0, 8.0)] * 5 + [(-14.0, 4.0)] * 5 + [(-14.0, 2.0)] * 5
    step = 1.0
    while step > 0.03:
        for _pass in range(2):
            improved = False
            for index, (lower, upper) in enumerate(bounds):
                for direction in (-step, step):
                    candidate = best.copy()
                    candidate[index] = np.clip(
                        candidate[index] + direction, lower, upper
                    )
                    current = extended_fit_loss(candidate, curves, scales, **loss_args)
                    if current < best_loss:
                        best, best_loss, improved = candidate, current, True
            if not improved:
                break
        step *= 0.5

    log_eta = best[:5]
    log_saturation = best[5:10]
    log_startup = best[10:15]

    # A shared rate/time scale preserves every selected k: rates multiply by q
    # while startup seconds divide by q.
    log_ratios = []
    for curve in curves:
        selected_k, predicted_s = optimal_integer_k_extended(
            log_eta, log_saturation, log_startup, scales, curve
        )
        measured_s = measured_e2e_at_ratio(curve, selected_k / curve.max_k)
        log_ratios.append(math.log(predicted_s / measured_s))
    common_rate_scale = float(math.exp(np.mean(log_ratios)))
    eta = np.exp(log_eta) * common_rate_scale
    saturation = scales[0] * np.exp(log_saturation)
    startup = scales[1] * np.exp(log_startup) / common_rate_scale
    return eta, saturation, startup, best_loss, common_rate_scale


def fit_loss(
    log_eta: np.ndarray,
    curves: list[Curve],
    e2e_weight: float,
    grid_optimum_weight: float,
    absolute_weight: float,
    regularization: float,
) -> float:
    log_e2e_regrets = []
    grid_ratio_errors = []
    absolute_errors = []
    for curve in curves:
        selected_k, predicted_s = optimal_integer_k(log_eta, curve)
        selected_ratio = selected_k / curve.max_k
        selected_measured_s = measured_e2e_at_ratio(curve, selected_ratio)
        grid_best = int(np.argmin(curve.measured_s))
        best_measured_s = float(curve.measured_s[grid_best])

        # mean(log(selected/oracle)) is log of the geometric-mean regret.
        # Minimizing it therefore directly minimizes aggregate selected E2E.
        log_e2e_regrets.append(math.log(selected_measured_s / best_measured_s))
        grid_ratio_errors.append(selected_ratio - curve.ratios[grid_best])
        absolute_errors.append(math.log(predicted_s / selected_measured_s))
    return float(
        e2e_weight * np.mean(log_e2e_regrets)
        + grid_optimum_weight * np.mean(np.square(grid_ratio_errors))
        + absolute_weight * np.mean(np.square(absolute_errors))
        + regularization * np.mean(np.square(log_eta))
    )


def fit_hardware_eta(
    curves: list[Curve],
    *,
    random_samples: int,
    e2e_weight: float,
    grid_optimum_weight: float,
    absolute_weight: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Fit eta in (0, 1] without changing any theoretical work count."""

    rng = np.random.default_rng(seed)
    best = np.zeros(5, dtype=float)
    best_loss = fit_loss(
        best, curves, e2e_weight, grid_optimum_weight,
        absolute_weight, regularization
    )
    for candidate in rng.uniform(-8.0, 0.0, size=(random_samples, 5)):
        current = fit_loss(
            candidate, curves, e2e_weight, grid_optimum_weight,
            absolute_weight, regularization
        )
        if current < best_loss:
            best, best_loss = candidate.copy(), current

    step = 1.0
    while step > 0.003:
        improved = True
        while improved:
            improved = False
            for index in range(5):
                for direction in (-step, step):
                    candidate = best.copy()
                    candidate[index] = np.clip(
                        candidate[index] + direction, -10.0, 0.0
                    )
                    current = fit_loss(
                        candidate, curves, e2e_weight, grid_optimum_weight,
                        absolute_weight, regularization,
                    )
                    if current < best_loss:
                        best, best_loss, improved = candidate, current, True
        step *= 0.5
    eta = np.exp(best)

    # A common multiplier on all five effective rates divides every predicted
    # time by the same value and therefore cannot change argmin(k).  Use this
    # hardware-group-only degree of freedom to align the geometric mean cost-
    # model E2E with the geometric mean measured E2E at the selected points.
    # The factor may exceed one because eta is an achieved/reference rate
    # factor, not necessarily utilization against a theoretical peak.
    log_eta = np.log(eta)
    log_ratios = []
    for curve in curves:
        selected_k, predicted_s = optimal_integer_k(log_eta, curve)
        selected_measured_s = measured_e2e_at_ratio(
            curve, selected_k / curve.max_k
        )
        log_ratios.append(math.log(predicted_s / selected_measured_s))
    common_rate_scale = float(math.exp(np.mean(log_ratios)))
    eta *= common_rate_scale
    normalized_loss = fit_loss(
        np.log(eta), curves, e2e_weight, grid_optimum_weight,
        absolute_weight, regularization,
    )
    return eta, normalized_loss, common_rate_scale


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def validation(curves_by_hardware, fitted) -> dict:
    exact = near = plateau = 0
    regret = []
    ratio_error = []
    point_error = []
    normalized_error = []
    selected_e2e = []
    predicted_e2e = []
    oracle_e2e = []
    for hardware_key, curves in curves_by_hardware.items():
        log_eta = np.log(fitted[hardware_key])
        for curve in curves:
            predicted = prediction(log_eta, curve.components_s)
            measured = curve.measured_s
            measured_best = int(np.argmin(measured))
            selected_k, selected_predicted_s = optimal_integer_k(log_eta, curve)
            selected_ratio = selected_k / curve.max_k
            predicted_best = int(np.argmin(np.abs(curve.ratios - selected_ratio)))
            selected_measured_s = measured_e2e_at_ratio(curve, selected_ratio)
            exact += predicted_best == measured_best
            near += abs(predicted_best - measured_best) <= 1
            plateau += selected_measured_s <= 1.01 * measured[measured_best]
            regret.append(100.0 * (selected_measured_s / measured[measured_best] - 1.0))
            ratio_error.append(abs(selected_ratio - curve.ratios[measured_best]))
            selected_e2e.append(selected_measured_s)
            predicted_e2e.append(selected_predicted_s)
            oracle_e2e.append(float(measured[measured_best]))
            point_error.append(100.0 * abs(selected_predicted_s / selected_measured_s - 1.0))
            point_error.extend(100.0 * np.abs(predicted / measured - 1.0))
            normalized_error.extend(100.0 * np.abs(
                (predicted / predicted.min()) / (measured / measured.min()) - 1.0
            ))
    geometric_selected = float(np.exp(np.mean(np.log(selected_e2e))))
    geometric_predicted = float(np.exp(np.mean(np.log(predicted_e2e))))
    geometric_oracle = float(np.exp(np.mean(np.log(oracle_e2e))))
    return {
        "curves": int(len(regret)),
        "exact_grid_optimum": int(exact),
        "within_one_grid_step": int(near),
        "inside_one_percent_plateau": int(plateau),
        "ratio_error_median": float(np.median(ratio_error)),
        "ratio_error_p90": percentile(ratio_error, 90),
        "latency_regret_median_pct": float(np.median(regret)),
        "latency_regret_p90_pct": percentile(regret, 90),
        "latency_regret_max_pct": float(max(regret)),
        "geometric_mean_selected_e2e_us": geometric_selected * 1e6,
        "geometric_mean_cost_model_e2e_us": geometric_predicted * 1e6,
        "geometric_mean_grid_oracle_e2e_us": geometric_oracle * 1e6,
        "geometric_mean_e2e_regret_factor": geometric_selected / geometric_oracle,
        "geometric_mean_e2e_regret_pct": 100.0 * (
            geometric_selected / geometric_oracle - 1.0
        ),
        "point_absolute_error_median_pct": float(np.median(point_error)),
        "point_absolute_error_p90_pct": percentile(point_error, 90),
        "normalized_curve_error_median_pct": float(np.median(normalized_error)),
        "normalized_curve_error_p90_pct": percentile(normalized_error, 90),
    }


def validation_extended(curves_by_hardware, fitted) -> dict:
    """Validate saturation/startup profiles with the all-integer-k decision."""

    exact = near = plateau = 0
    regret = []
    ratio_error = []
    point_error = []
    normalized_error = []
    selected_e2e = []
    predicted_e2e = []
    oracle_e2e = []
    for hardware_key, curves in curves_by_hardware.items():
        eta, saturation, startup = fitted[hardware_key]
        scales = extension_scales(curves)
        log_eta = np.log(eta)
        log_saturation = np.log(saturation / scales[0])
        log_startup = np.log(startup / scales[1])
        for curve in curves:
            measured = curve.measured_s
            measured_best = int(np.argmin(measured))
            selected_k, selected_predicted_s = optimal_integer_k_extended(
                log_eta, log_saturation, log_startup, scales, curve
            )
            selected_ratio = selected_k / curve.max_k
            predicted_best = int(np.argmin(np.abs(curve.ratios - selected_ratio)))
            selected_measured_s = measured_e2e_at_ratio(curve, selected_ratio)
            predicted_grid = np.asarray([
                extended_time_at_k(
                    log_eta, log_saturation, log_startup, scales, curve,
                    int(round(ratio * curve.max_k)),
                )
                for ratio in curve.ratios
            ])
            exact += predicted_best == measured_best
            near += abs(predicted_best - measured_best) <= 1
            plateau += selected_measured_s <= 1.01 * measured[measured_best]
            regret.append(100.0 * (selected_measured_s / measured[measured_best] - 1.0))
            ratio_error.append(abs(selected_ratio - curve.ratios[measured_best]))
            selected_e2e.append(selected_measured_s)
            predicted_e2e.append(selected_predicted_s)
            oracle_e2e.append(float(measured[measured_best]))
            point_error.append(100.0 * abs(selected_predicted_s / selected_measured_s - 1.0))
            point_error.extend(100.0 * np.abs(predicted_grid / measured - 1.0))
            normalized_error.extend(100.0 * np.abs(
                (predicted_grid / predicted_grid.min())
                / (measured / measured.min()) - 1.0
            ))
    geometric_selected = float(np.exp(np.mean(np.log(selected_e2e))))
    geometric_predicted = float(np.exp(np.mean(np.log(predicted_e2e))))
    geometric_oracle = float(np.exp(np.mean(np.log(oracle_e2e))))
    return {
        "curves": int(len(regret)),
        "exact_grid_optimum": int(exact),
        "within_one_grid_step": int(near),
        "inside_one_percent_plateau": int(plateau),
        "ratio_error_median": float(np.median(ratio_error)),
        "ratio_error_p90": percentile(ratio_error, 90),
        "latency_regret_median_pct": float(np.median(regret)),
        "latency_regret_p90_pct": percentile(regret, 90),
        "latency_regret_max_pct": float(max(regret)),
        "geometric_mean_selected_e2e_us": geometric_selected * 1e6,
        "geometric_mean_cost_model_e2e_us": geometric_predicted * 1e6,
        "geometric_mean_grid_oracle_e2e_us": geometric_oracle * 1e6,
        "geometric_mean_e2e_regret_factor": geometric_selected / geometric_oracle,
        "geometric_mean_e2e_regret_pct": 100.0 * (
            geometric_selected / geometric_oracle - 1.0
        ),
        "point_absolute_error_median_pct": float(np.median(point_error)),
        "point_absolute_error_p90_pct": percentile(point_error, 90),
        "normalized_curve_error_median_pct": float(np.median(normalized_error)),
        "normalized_curve_error_p90_pct": percentile(normalized_error, 90),
    }


def fit_one_hardware_group(
    hardware_key,
    group,
    index: int,
    options: dict,
):
    """Process-safe fit for one chip/storage/batch hardware group."""

    eta, base_loss, base_scale = fit_hardware_eta(
        group,
        random_samples=options["random_samples"],
        e2e_weight=options["e2e_weight"],
        grid_optimum_weight=options["grid_optimum_weight"],
        absolute_weight=options["absolute_weight"],
        regularization=options["regularization"],
        seed=options["seed"] + index,
    )
    eta, saturation, startup, extended_loss, extension_scale = (
        fit_hardware_extensions(
            group,
            eta,
            random_samples=options["extension_random_samples"],
            e2e_weight=options["e2e_weight"],
            grid_optimum_weight=options["grid_optimum_weight"],
            curve_shape_weight=options["curve_shape_weight"],
            regularization=options["regularization"],
            seed=options["seed"] + 1000 + index,
        )
    )
    return (
        hardware_key, eta, saturation, startup,
        base_loss, base_scale, extended_loss, extension_scale,
    )


def main() -> None:
    args = parse_args()
    if args.random_samples <= 0:
        raise SystemExit("--random-samples must be positive")
    if args.extension_random_samples <= 0:
        raise SystemExit("--extension-random-samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    for name in (
        "e2e_weight", "grid_optimum_weight", "absolute_weight", "regularization"
    ):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    curves = load_complete_curves(args.result_root.resolve())
    if not curves:
        raise SystemExit("no complete 0..1 ratio curves found")
    definition_path = args.result_root.resolve() / "matrix_definition.json"
    configured_hardware = {}
    if definition_path.is_file():
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        configured_hardware = definition.get("chip_configs", {})

    fitted = {}
    profile_chips: dict = {}
    fit_summary = {}
    options = {
        "random_samples": args.random_samples,
        "extension_random_samples": args.extension_random_samples,
        "e2e_weight": args.e2e_weight,
        "grid_optimum_weight": args.grid_optimum_weight,
        "curve_shape_weight": args.curve_shape_weight,
        "absolute_weight": args.absolute_weight,
        "regularization": args.regularization,
        "seed": args.seed,
    }
    indexed_groups = [
        (index, hardware_key, group)
        for index, (hardware_key, group) in enumerate(sorted(curves.items()))
    ]
    completed_fits = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(indexed_groups))
    ) as pool:
        futures = {
            pool.submit(
                fit_one_hardware_group, hardware_key, group, index, options
            ): hardware_key
            for index, hardware_key, group in indexed_groups
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed_fits.append(result)
            print(
                f"[{completed}/{len(futures)}] fitted {futures[future]}",
                flush=True,
            )

    for result in sorted(completed_fits, key=lambda item: item[0]):
        (
            hardware_key, eta, saturation, startup,
            loss, common_rate_scale, extended_loss, extension_scale,
        ) = result
        group = curves[hardware_key]
        fitted[hardware_key] = (eta, saturation, startup)
        chip, medium, batch = hardware_key
        record = {name: float(value) for name, value in zip(ETA_NAMES, eta)}
        record.update({
            name: float(value)
            for name, value in zip(SATURATION_NAMES, saturation)
        })
        record.update({
            name: float(value)
            for name, value in zip(STARTUP_NAMES, startup)
        })
        (
            profile_chips.setdefault(chip, {})
            .setdefault("media", {})
            .setdefault(medium, {})
            .setdefault("batches", {})[str(batch)]
        ) = record
        fit_summary[f"{chip}/{medium}/bs{batch}"] = {
            "curve_count": len(group),
            "point_count": sum(len(curve.ratios) for curve in group),
            "base_rate_fit_loss": loss,
            "base_rate_common_scale": common_rate_scale,
            "extended_fit_loss": extended_loss,
            "extension_common_rate_scale": extension_scale,
            "hardware_parameters": record,
            "e2e_decision_validation": validation_extended(
                {hardware_key: group},
                {hardware_key: (eta, saturation, startup)},
            ),
        }

    output = {
        "_comment": (
            "Calibrated hardware-only parameters for the item-KV paper cost "
            "model. Pass this file with --calibration; raw B_kv/B_emb/B_core "
            "and F_cube/F_vec still come from --config or explicit CLI inputs."
        ),
        "usage": {
            "predict": (
                "python3 scripts/recompute_ratio_cost_model_new.py "
                "--cost-model paper --config configs/910C.json "
                "--calibration configs/item_kv_calib.json --user cold "
                "--layers 4 --hidden 256 --kv-len 4096 --batch-size 1 "
                "--enable-kv-reuse --kv-reuse-ratio 0.4802"
            ),
            "result_fields": (
                "history_recompute_len is k; recompute_ratio is k/S_i; "
                "Tlatency_us is the cost-model latency proxy."
            ),
            "recalibrate": (
                "python3 scripts/calibrate_item_kv_hardware.py "
                f"{args.result_root} "
                "--output configs/item_kv_calib.json"
            ),
        },
        "paper_cost_model": {
            "schema_version": 2,
            "model": "explicit_item_kv_with_saturation_startup_and_batch",
            "calibration_scope": "chip_storage_medium_batch_only",
            "numerator_calibration": False,
            "hardware_extensions": {
                "saturation": "R_eff=R_peak*eta*(1-exp(-work_per_request/x_sat))",
                "startup": "T_path=work/R_eff+n_requests*tau_startup",
            },
            "hardware_profiles": {"chips": profile_chips},
        },
        "paper_cost_model_calibration": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(args.result_root.resolve()),
            "complete_curve_count": sum(len(group) for group in curves.values()),
            "complete_point_count": 11 * sum(len(group) for group in curves.values()),
            "fit_parameters": {
                "random_samples": args.random_samples,
                "primary_objective": "geometric_mean_selected_e2e_regret",
                "e2e_weight": args.e2e_weight,
                "grid_optimum_auxiliary": True,
                "grid_optimum_weight": args.grid_optimum_weight,
                "curve_shape_weight": args.curve_shape_weight,
                "absolute_weight": args.absolute_weight,
                "extension_random_samples": args.extension_random_samples,
                "workers": min(args.workers, len(indexed_groups)),
                "regularization": args.regularization,
                "seed": args.seed,
                "relative_eta_search_bounds": [math.exp(-10.0), 1.0],
                "post_fit_common_rate_scale": (
                    "unbounded hardware-group scale; preserves selected k"
                ),
                "eta_semantics": (
                    "effective_rate=input_rate*eta; hardware-only eta may "
                    "exceed 1 for a nominal non-peak input rate"
                ),
            },
            "hardware_group_fits": fit_summary,
            "calibrated_hardware_configurations": configured_hardware,
            "in_sample_validation": validation_extended(curves, fitted),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        output["paper_cost_model_calibration"]["in_sample_validation"],
        indent=2,
    ))


if __name__ == "__main__":
    main()
