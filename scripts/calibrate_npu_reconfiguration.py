#!/usr/bin/env python3
"""Calibrate a hardware-only Cube/Vector response for NPU reconfiguration.

For each completed w_both simulator run, this script retains the run's chosen
integer recompute length k and fits only the two physical-rate exponents:

  F'_cube = F_cube * (F_cube / F_cube,ref) ** alpha_cube
  F'_vec  = F_vec  * (F_vec  / F_vec,ref ) ** alpha_vec

The work terms and the k selection are deliberately not fitted.  Thus the
saved calibration can be used for a new model, sequence length, batch size,
or recompute decision without introducing model-specific coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import recompute_ratio_cost_model_new as estimator
from search_npu_reconfiguration import MODELS, Workload, WorkloadCost, hardware_profile


@dataclass
class Sample:
    chip: str
    candidate: str
    actual_s: float
    memory_s: float
    cube_numerator: float
    vector_numerator: float
    fixed_npu_s: float
    cube_rate: float
    vector_rate: float
    reference_cube_rate: float
    reference_vector_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--candidate-config-root", type=Path, required=True)
    parser.add_argument(
        "--result-root", action="append", type=Path, required=True,
        help="A root containing one directory per candidate (repeatable).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument(
        "--min-exponent", type=float, default=0.0,
        help="Lower bound for both response exponents (0 preserves monotonic throughput).",
    )
    parser.add_argument("--max-abs-exponent", type=float, default=3.0)
    parser.add_argument("--regularization", type=float, default=0.002)
    return parser.parse_args()


def npu_latency_s(summary: Path) -> float | None:
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("component") == "NPU" and row.get("scope") == "overall":
                return float(row["sim_time_us"]) * 1e-6
    return None


def candidate_config(config_root: Path, candidate: str) -> Path:
    path = config_root / f"{candidate}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def collect_samples(args: argparse.Namespace, calibration: dict) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[Path] = set()
    references: dict[str, dict[str, float]] = {}
    for root in args.result_root:
        if not root.is_dir():
            continue
        for status_path in root.glob("*/logs/*.status.json"):
            if status_path in seen:
                continue
            seen.add(status_path)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("returncode") != 0 or status.get("method") != "w_both":
                continue
            candidate = status_path.parent.parent.name
            config = candidate_config(args.candidate_config_root, candidate)
            command = status.get("command", [])
            try:
                result_dir = Path(command[command.index("--result-dir") + 1])
                summary = result_dir / "hardware_summary.csv"
                k = int(status["history_recompute_len"])
                chip = str(status["chip"])
                model = str(status["model"])
                layers, hidden = MODELS[model]
                workload = Workload(
                    model, layers, hidden, int(status["seq_len"]),
                    int(status["batch_size"]), str(status["user"]),
                )
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(f"invalid status record: {status_path}") from exc
            actual_s = npu_latency_s(summary)
            if actual_s is None:
                continue
            cfg_hw = estimator.derive_hardware(config)
            if chip not in references:
                references[chip] = estimator.derive_hardware(
                    Path(f"configs/{chip}.json")
                )
            ref_hw = references[chip]
            medium = "ddr" if workload.user == "hot" else "ssd"
            cost = WorkloadCost(
                workload, cfg_hw,
                hardware_profile(calibration, chip, medium, workload.batch),
            )
            if not 0 <= k < len(cost.k):
                raise ValueError(f"k={k} outside range for {status_path}")
            samples.append(Sample(
                chip=chip, candidate=candidate, actual_s=actual_s,
                memory_s=float(cost.memory[k]),
                cube_numerator=float(cost.cube_numerator[k]),
                vector_numerator=float(cost.vec_numerator[k]),
                fixed_npu_s=float(cost.fixed_npu[k]),
                cube_rate=float(cfg_hw["F_cube"]),
                vector_rate=float(cfg_hw["F_vec"]),
                reference_cube_rate=float(ref_hw["F_cube"]),
                reference_vector_rate=float(ref_hw["F_vec"]),
            ))
    return samples


def fit_chip(samples: list[Sample], args: argparse.Namespace) -> dict:
    cube_ratio = np.asarray([s.cube_rate / s.reference_cube_rate for s in samples])
    vector_ratio = np.asarray([s.vector_rate / s.reference_vector_rate for s in samples])
    memory = np.asarray([s.memory_s for s in samples])
    cube = np.asarray([s.cube_numerator / s.cube_rate for s in samples])
    vector = np.asarray([s.vector_numerator / s.vector_rate for s in samples])
    fixed = np.asarray([s.fixed_npu_s for s in samples])
    actual = np.asarray([s.actual_s for s in samples])
    exponent = np.arange(
        args.min_exponent, args.max_abs_exponent + args.grid_step / 2,
        args.grid_step,
    )
    cube_scale = cube_ratio[:, None] ** (-exponent[None, :])
    vector_scale = vector_ratio[:, None] ** (-exponent[None, :])
    best = None
    for cube_index, alpha_cube in enumerate(exponent):
        npu = cube[:, None] * cube_scale[:, cube_index][:, None] + fixed[:, None]
        predicted = np.maximum(memory[:, None], npu + vector[:, None] * vector_scale)
        error = np.log(predicted / actual[:, None])
        loss = np.mean(error * error, axis=0) + args.regularization * (
            alpha_cube * alpha_cube + exponent * exponent
        )
        vector_index = int(np.argmin(loss))
        candidate = (float(loss[vector_index]), float(alpha_cube), float(exponent[vector_index]), predicted[:, vector_index])
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, alpha_cube, alpha_vector, predicted = best
    log_error = np.log(predicted / actual)
    return {
        "reference_cube_flops": samples[0].reference_cube_rate,
        "reference_vector_ops": samples[0].reference_vector_rate,
        "cube_rate_exponent": alpha_cube,
        "vector_rate_exponent": alpha_vector,
        "fit_samples": len(samples),
        "fit_candidates": len({s.candidate for s in samples}),
        "geomean_predicted_over_actual": float(math.exp(float(np.mean(log_error)))),
        "geometric_error_factor": float(math.exp(float(np.sqrt(np.mean(log_error * log_error))))),
    }


def main() -> None:
    args = parse_args()
    if args.grid_step <= 0 or args.max_abs_exponent <= 0 or args.min_exponent < 0:
        raise SystemExit("grid parameters must be positive")
    calibration = json.loads(args.base_calibration.read_text(encoding="utf-8"))
    samples = collect_samples(args, calibration)
    if not samples:
        raise SystemExit("no completed candidate w_both samples found")
    chips = {chip: [s for s in samples if s.chip == chip] for chip in sorted({s.chip for s in samples})}
    response = {
        "_comment": (
            "Hardware-only response for reconfigured arrays. The item-KV model uses "
            "F'_cube=F_cube*(F_cube/F_cube,ref)^cube_rate_exponent and "
            "F'_vec=F_vec*(F_vec/F_vec,ref)^vector_rate_exponent. It is applied "
            "automatically when --config is a generated NPU-reconfiguration JSON."
        ),
        "chips": {chip: fit_chip(chip_samples, args) for chip, chip_samples in chips.items()},
    }
    calibration["npu_reconfiguration_response"] = response
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(samples), "chips": response["chips"]}, indent=2))


if __name__ == "__main__":
    main()
