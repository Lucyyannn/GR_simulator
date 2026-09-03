#!/usr/bin/env python3
"""Refine hardware-only item-KV rates from completed NPU DSE simulations.

The fitted values remain the paper model's eta_kv/eta_emb/eta_cube/eta_vec/
eta_core hardware-rate factors.  No model, hidden-size, sequence-length, or
recompute-work numerator coefficient is introduced.  Fits use normalized E2E
latency so each baseline remains exactly 1.0, and report a deterministic
design-level holdout validation rather than tuning to a requested speedup.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import recompute_ratio_cost_model_new as estimator
import search_npu_reconfiguration as search


ETA_NAMES = ("eta_kv", "eta_emb", "eta_cube", "eta_vec", "eta_core")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--validation-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--regularization", type=float, default=2e-4)
    return parser.parse_args()


def fold_for_design(chip: str, design_id: str) -> int:
    digest = hashlib.sha256(f"{chip}:{design_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 5


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def profile_with_delta(base: dict, delta: np.ndarray) -> dict:
    profile = copy.deepcopy(base)
    for name, adjustment in zip(ETA_NAMES, delta):
        profile[name] = float(base[name]) * math.exp(float(adjustment))
    return profile


def normalized_prediction(
    workload: search.Workload,
    baseline_hw: dict,
    profile: dict,
    cube_rates: np.ndarray,
    vector_rates: np.ndarray,
    baseline_index: int,
) -> np.ndarray:
    cost = search.WorkloadCost(workload, baseline_hw, profile)
    _, latency = cost.select(cube_rates, vector_rates)
    return latency / latency[baseline_index]


def huber(values: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(
        absolute <= threshold,
        0.5 * values * values,
        threshold * (absolute - 0.5 * threshold),
    )


def fit_context(
    *,
    workload: search.Workload,
    baseline_hw: dict,
    base_profile: dict,
    cube_rates: np.ndarray,
    vector_rates: np.ndarray,
    measured_normalized: np.ndarray,
    baseline_index: int,
    train_mask: np.ndarray,
    samples: int,
    seed: int,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    log_measured = np.log(measured_normalized)

    def loss(delta: np.ndarray) -> float:
        predicted = normalized_prediction(
            workload, baseline_hw, profile_with_delta(base_profile, delta),
            cube_rates, vector_rates, baseline_index,
        )
        residual = np.log(predicted) - log_measured
        data_loss = float(np.mean(huber(residual[train_mask])))
        return data_loss + regularization * float(np.mean(delta * delta))

    rng = np.random.default_rng(seed)
    candidates = rng.uniform(-4.0, 4.0, size=(samples, len(ETA_NAMES)))
    candidates[0] = 0.0
    losses = np.asarray([loss(candidate) for candidate in candidates])
    best = candidates[int(np.argmin(losses))].copy()
    best_loss = float(np.min(losses))
    for step in (0.50, 0.20, 0.08, 0.03):
        improved = True
        while improved:
            improved = False
            for index in range(len(best)):
                for direction in (-1.0, 1.0):
                    trial = best.copy()
                    trial[index] = np.clip(
                        trial[index] + direction * step, -4.0, 4.0
                    )
                    trial_loss = loss(trial)
                    if trial_loss + 1e-15 < best_loss:
                        best, best_loss = trial, trial_loss
                        improved = True

    fitted_profile = profile_with_delta(base_profile, best)
    predicted = normalized_prediction(
        workload, baseline_hw, fitted_profile,
        cube_rates, vector_rates, baseline_index,
    )
    base_predicted = normalized_prediction(
        workload, baseline_hw, base_profile,
        cube_rates, vector_rates, baseline_index,
    )
    validation_mask = ~train_mask

    def metrics(values: np.ndarray, mask: np.ndarray) -> dict:
        log_error = np.log(values[mask]) - log_measured[mask]
        return {
            "count": int(np.sum(mask)),
            "log_rmse": float(np.sqrt(np.mean(log_error * log_error))),
            "mape": float(np.mean(
                np.abs(values[mask] - measured_normalized[mask])
                / measured_normalized[mask]
            )),
            "log_correlation": correlation(
                np.log(values[mask]), log_measured[mask]
            ),
        }

    report = {
        "eta_log_adjustments": {
            name: float(value) for name, value in zip(ETA_NAMES, best)
        },
        "train": metrics(predicted, train_mask),
        "validation": metrics(predicted, validation_mask),
        "base_validation": metrics(base_predicted, validation_mask),
        "fit_loss": best_loss,
    }
    return best, report


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    base = json.loads(args.base_calibration.read_text(encoding="utf-8"))
    output = copy.deepcopy(base)
    manifest = json.loads((root / "design_manifest.json").read_text(encoding="utf-8"))
    designs = manifest["designs"]
    points = list(csv.DictReader((root / "pareto_points.csv").open(encoding="utf-8")))
    by_latency = {
        (row["chip"], row["design_id"], int(row["batch_size"]), row["user"]):
        float(row["normalized_latency"])
        for row in points
    }
    reports = {}
    profile_root = output["paper_cost_model"]["hardware_profiles"]["chips"]
    chips = sorted({row["chip"] for row in designs})
    for chip_index, chip in enumerate(chips):
        chip_designs = [row for row in designs if row["chip"] == chip]
        baseline_index = next(
            index for index, row in enumerate(chip_designs)
            if row["role"] == "baseline"
        )
        hardware = [
            estimator.derive_hardware(Path(row["config"]))
            for row in chip_designs
        ]
        cube_rates = np.asarray([row["F_cube"] for row in hardware])
        vector_rates = np.asarray([row["F_vec"] for row in hardware])
        baseline_hw = hardware[baseline_index]
        for batch in (4, 8):
            for user in ("hot", "cold"):
                medium = "ddr" if user == "hot" else "ssd"
                measured = np.asarray([
                    by_latency[(chip, row["design_id"], batch, user)]
                    for row in chip_designs
                ])
                validation_mask = np.asarray([
                    fold_for_design(chip, row["design_id"])
                    == args.validation_fold and row["role"] == "random"
                    for row in chip_designs
                ])
                train_mask = ~validation_mask
                workload = search.Workload(
                    "small", 4, 256, int(manifest["workloads"]["sequence"]),
                    batch, user,
                )
                base_profile = profile_root[chip]["media"][medium]["batches"][str(batch)]
                delta, report = fit_context(
                    workload=workload,
                    baseline_hw=baseline_hw,
                    base_profile=base_profile,
                    cube_rates=cube_rates,
                    vector_rates=vector_rates,
                    measured_normalized=measured,
                    baseline_index=baseline_index,
                    train_mask=train_mask,
                    samples=args.random_samples,
                    seed=args.seed + chip_index * 10 + batch + (user == "cold"),
                    regularization=args.regularization,
                )
                fitted = profile_with_delta(base_profile, delta)
                profile_root[chip]["media"][medium]["batches"][str(batch)] = fitted
                key = f"{chip}/{medium}/bs{batch}"
                reports[key] = report
                print(
                    f"[{key}] validation MAPE "
                    f"{report['base_validation']['mape']:.4f} -> "
                    f"{report['validation']['mape']:.4f}",
                    flush=True,
                )

    output["npu_dse_hardware_calibration"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "scope": "chip_storage_medium_batch_hardware_rates_only",
        "model_specific_coefficients": False,
        "numerator_calibration": False,
        "validation_fold": args.validation_fold,
        "fit_parameters": {
            "random_samples": args.random_samples,
            "seed": args.seed,
            "regularization": args.regularization,
            "eta_log_adjustment_bounds": [-4.0, 4.0],
        },
        "contexts": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
