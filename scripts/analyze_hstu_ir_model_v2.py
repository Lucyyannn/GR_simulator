#!/usr/bin/env python3
"""Fit and validate the phase-aware HSTU SSD IR analytical model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np


RIDGE_LAMBDA = 0.01
STAGE_FEATURES = [
    "log2_hidden", "log2_batch", "log2_sequence", "log2_cores",
    "ratio", "ratio_sq", "is_split", "ratio_x_log2_batch",
    "ratio_x_log2_hidden", "ratio_x_log2_cores",
    "ratio_x_log2_sequence", "split_x_log2_batch",
    "split_x_log2_hidden", "log2_peak_stage_us",
    "log2_batch_x_log2_hidden", "log2_batch_x_log2_cores",
]
MEMORY_FEATURES = [
    "log2_hidden", "log2_batch", "log2_sequence", "log2_cores",
    "ratio", "ratio_sq", "is_w_both", "ratio_x_w_both",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("source_root", type=Path)
  parser.add_argument("output_root", type=Path)
  parser.add_argument(
      "--base-calibration", type=Path,
      default=Path("scripts/recompute_ratio_calibration.json"),
  )
  parser.add_argument("--max-workers", type=int, default=24)
  return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
  fields = list(rows[0]) if rows else []
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
  return float(np.percentile(np.asarray(values, dtype=float), q))


def model_command(row: dict, calibration: Path) -> list[str]:
  command = [
      "python3", "scripts/recompute_ratio_cost_model_new.py",
      "--config", f"configs/{row['chip']}.json",
      "--calibration", str(calibration),
      "--user", "cold",
      "--layers", str(row["layers"]),
      "--hidden", str(row["hidden"]),
      "--kv-len", str(row["seq_len"]),
      "--batch-size", str(row["batch_size"]),
      "--candidates", "128",
      "--embedding-source", "ssd",
      "--objective", "e2e",
      "--fixed-recompute-len", str(row["history_recompute_len"]),
      "--field", "json",
  ]
  if row["method"] == "w_both":
    command += [
        "--enable-kv-reuse", "--kv-reuse-ratio", "0.436",
        "--no-kv-reuse-reduce-npu",
    ]
  return command


def predict_one(repo: Path, row: dict, calibration: Path) -> dict:
  result = subprocess.run(
      model_command(row, calibration), cwd=repo, text=True,
      capture_output=True, check=True,
  )
  return json.loads(result.stdout)


def stage_times(path: Path, layers: int, ratio: float) -> dict[str, float]:
  rows = read_csv(path)
  history = [
      row for row in rows
      if row["pipe"] == "preload"
      and row["phase"] == "history_recompute_embedding"
  ]
  history_end = max(
      [float(row["end_us"]) for row in history], default=-1.0
  )

  def one_layer(layer: int) -> tuple[float, float, float]:
    ops = sorted(
        [
            row for row in rows
            if row["pipe"] == "compute" and row["phase"] == "op"
            and int(row["layer_id"]) == layer
        ],
        key=lambda row: float(row["start_us"]),
    )
    boundary = next(
        (
            index for index, row in enumerate(ops)
            if "cached_late" in row["name"]
            or "candidate_cached" in row["name"]
            or row["name"] == "hstu::attention"
        ),
        len(ops),
    )
    if layer == 0 and ratio > 0:
      base = sum(
          float(row["duration_us"]) for row in ops[:boundary]
          if float(row["start_us"]) < history_end - 1e-4
      )
      recompute = sum(
          float(row["duration_us"]) for row in ops[:boundary]
          if float(row["start_us"]) >= history_end - 1e-4
      )
    else:
      base = sum(float(row["duration_us"]) for row in ops[:boundary])
      recompute = 0.0
    late = sum(float(row["duration_us"]) for row in ops[boundary:])
    return base, recompute, late

  layer0 = one_layer(0)
  repeated = [one_layer(layer) for layer in range(1, layers)]
  if not repeated:
    repeated = [layer0]
  return {
      "actual_layer0_base_us": layer0[0],
      "actual_layer0_recompute_us": layer0[1],
      "actual_layer0_late_us": layer0[2],
      "actual_repeated_early_us": statistics.fmean(
          base + recompute for base, recompute, _ in repeated
      ),
      "actual_repeated_late_us": statistics.fmean(
          late for _, _, late in repeated
      ),
  }


def feature_values(row: dict) -> dict[str, float]:
  prediction = row["prediction"]
  log_hidden = math.log2(float(row["hidden"]) / 256.0)
  log_batch = math.log2(float(row["batch_size"]))
  log_sequence = math.log2(float(row["seq_len"]) / 4096.0)
  log_cores = math.log2(float(prediction["num_cores"]) / 8.0)
  ratio = float(row["actual_ratio"])
  is_split = 1.0 if int(row["history_recompute_len"]) > 0 else 0.0
  total_peak = (
      float(prediction["pre_base_peak_us"])
      + float(prediction["pre_recompute_peak_us"])
      + float(prediction["late_peak_us"])
  )
  is_both = 1.0 if row["method"] == "w_both" else 0.0
  return {
      "log2_hidden": log_hidden,
      "log2_batch": log_batch,
      "log2_sequence": log_sequence,
      "log2_cores": log_cores,
      "ratio": ratio,
      "ratio_sq": ratio * ratio,
      "is_split": is_split,
      "ratio_x_log2_batch": ratio * log_batch,
      "ratio_x_log2_hidden": ratio * log_hidden,
      "ratio_x_log2_cores": ratio * log_cores,
      "ratio_x_log2_sequence": ratio * log_sequence,
      "split_x_log2_batch": is_split * log_batch,
      "split_x_log2_hidden": is_split * log_hidden,
      "log2_peak_stage_us": math.log2(max(total_peak, 1e-12)),
      "log2_batch_x_log2_hidden": log_batch * log_hidden,
      "log2_batch_x_log2_cores": log_batch * log_cores,
      "is_w_both": is_both,
      "ratio_x_w_both": ratio * is_both,
  }


TARGETS = {
    "history_time_scale": (
        lambda row: float(row["history_embedding_us_measured"]),
        lambda row: float(row["prediction"]["history_embedding_us"]),
        MEMORY_FEATURES,
    ),
    "kv_time_scale": (
        lambda row: float(row["kv_preload_total_us"]) / int(row["layers"]),
        lambda row: float(row["prediction"]["layer_kv_preload_us"]),
        MEMORY_FEATURES,
    ),
    "layer0_base_time_scale": (
        lambda row: float(row["actual_layer0_base_us"]),
        lambda row: float(row["prediction"]["pre_base_peak_us"]),
        STAGE_FEATURES,
    ),
    "layer0_recompute_time_scale": (
        lambda row: float(row["actual_layer0_recompute_us"]),
        lambda row: float(row["prediction"]["pre_recompute_peak_us"]),
        STAGE_FEATURES,
    ),
    "layer0_late_time_scale": (
        lambda row: float(row["actual_layer0_late_us"]),
        lambda row: float(row["prediction"]["late_peak_us"]),
        STAGE_FEATURES,
    ),
    "repeated_early_time_scale": (
        lambda row: float(row["actual_repeated_early_us"]),
        lambda row: (
            float(row["prediction"]["pre_base_peak_us"])
            + float(row["prediction"]["pre_recompute_peak_us"])
        ),
        STAGE_FEATURES,
    ),
    "repeated_late_time_scale": (
        lambda row: float(row["actual_repeated_late_us"]),
        lambda row: float(row["prediction"]["late_peak_us"]),
        STAGE_FEATURES,
    ),
}


def fit_spec(rows: list[dict], target_name: str) -> dict:
  target, predicted, features = TARGETS[target_name]
  selected = [row for row in rows if target(row) > 0 and predicted(row) > 0]
  matrix = np.asarray(
      [[feature_values(row)[name] for name in features] for row in selected],
      dtype=float,
  )
  values = np.log(np.asarray(
      [target(row) / predicted(row) for row in selected], dtype=float
  ))
  mean = matrix.mean(axis=0)
  scale = matrix.std(axis=0)
  scale[scale < 1e-9] = 1.0
  design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
  penalty = np.eye(design.shape[1]) * RIDGE_LAMBDA
  penalty[0, 0] = 0.0
  beta = np.linalg.solve(
      design.T @ design + penalty, design.T @ values
  )
  return {
      "features": features,
      "intercept": float(beta[0]),
      "coefficients": [float(value) for value in beta[1:]],
      "feature_mean": [float(value) for value in mean],
      "feature_scale": [float(value) for value in scale],
      "ridge_lambda": RIDGE_LAMBDA,
      "min_scale": 0.01,
      "max_scale": 1000.0,
      "training_points": len(selected),
  }


def fitted_scale(spec: dict, row: dict) -> float:
  values = feature_values(row)
  result = float(spec["intercept"])
  for coefficient, name, mean, scale in zip(
      spec["coefficients"], spec["features"],
      spec["feature_mean"], spec["feature_scale"],
  ):
    result += coefficient * ((values[name] - mean) / scale)
  return max(
      float(spec["min_scale"]),
      min(float(spec["max_scale"]), math.exp(max(-20.0, min(20.0, result)))),
  )


def fit_model(rows: list[dict]) -> dict:
  model = {
      "version": 2,
      "mode": "phase_aware_event_dag",
      "ar_compute_reduction_default": False,
  }
  for target_name in TARGETS:
    model[target_name] = fit_spec(rows, target_name)
  return model


def predict_e2e(row: dict, model: dict) -> tuple[float, dict[str, float]]:
  prediction = row["prediction"]
  scales = {name: fitted_scale(model[name], row) for name in TARGETS}
  history = float(prediction["history_embedding_us"]) * scales[
      "history_time_scale"
  ]
  kv = float(prediction["layer_kv_preload_us"]) * scales["kv_time_scale"]
  candidate = float(prediction["candidate_embedding_us"])
  layer0_base = float(prediction["pre_base_peak_us"]) * scales[
      "layer0_base_time_scale"
  ]
  layer0_recompute = float(prediction["pre_recompute_peak_us"]) * scales[
      "layer0_recompute_time_scale"
  ]
  layer0_late = float(prediction["late_peak_us"]) * scales[
      "layer0_late_time_scale"
  ]
  repeated_early = (
      float(prediction["pre_base_peak_us"])
      + float(prediction["pre_recompute_peak_us"])
  ) * scales["repeated_early_time_scale"]
  repeated_late = float(prediction["late_peak_us"]) * scales[
      "repeated_late_time_scale"
  ]

  history_ready = candidate + history
  kv_cursor = history_ready
  compute_cursor = candidate
  for layer in range(int(row["layers"])):
    kv_cursor += kv
    if layer == 0:
      early_ready = max(compute_cursor + layer0_base, history_ready)
      early_ready += layer0_recompute
      late = layer0_late
    else:
      early_ready = compute_cursor + repeated_early
      late = repeated_late
    compute_cursor = max(early_ready, kv_cursor) + late
  return compute_cursor + float(prediction["weight_read_us"]), scales


def comparison_summary(values: list[float]) -> dict:
  return {
      "count": len(values),
      "median_pct": statistics.median(values),
      "median_absolute_pct": statistics.median(abs(value) for value in values),
      "p90_absolute_pct": percentile([abs(value) for value in values], 90),
      "max_absolute_pct": max(abs(value) for value in values),
  }


def main() -> None:
  args = parse_args()
  if not 1 <= args.max_workers <= 48:
    raise SystemExit("--max-workers must be in [1, 48]")
  repo = Path(__file__).resolve().parents[1]
  source = args.source_root.resolve()
  output = args.output_root.resolve()
  output.mkdir(parents=True, exist_ok=True)
  base_calibration = args.base_calibration.resolve()
  grid = [
      row for row in read_csv(source / "sweep_cases.csv")
      if row["point_kind"] == "grid"
  ]

  with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    predictions = list(executor.map(
        lambda row: predict_one(repo, row, base_calibration), grid
    ))
  for row, prediction in zip(grid, predictions):
    row["prediction"] = prediction
    case_name = (
        f"{row['context_id']}_{row['chip']}_{row['model']}_"
        f"seq{row['seq_len']}_bs{row['batch_size']}"
    )
    row.update(stage_times(
        source / "cases" / case_name / row["method"] / row["point_label"]
        / "layer_breakdown.csv",
        int(row["layers"]), float(row["actual_ratio"]),
    ))

  full_model = fit_model(grid)
  calibration = json.loads(base_calibration.read_text(encoding="utf-8"))
  calibration.setdefault("ir_cost_model", {})["dynamic_efficiency"] = full_model
  calibration["ir_cost_model"]["selection_objective"] = "e2e_event_dag_v2"
  calibration["ir_cost_model"]["dynamic_efficiency_source"] = str(source)
  calibration_path = output / "recompute_ratio_calibration_v2.json"
  calibration_path.write_text(
      json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )

  component_rows: list[dict] = []
  full_predictions: dict[tuple[str, str, str], tuple[float, dict]] = {}
  for row in grid:
    predicted_e2e, scales = predict_e2e(row, full_model)
    record = {
        key: row[key] for key in [
            "case_id", "context_id", "chip", "model", "method",
            "point_label", "layers", "hidden", "seq_len", "batch_size",
            "history_recompute_len", "actual_ratio", "sim_time_us",
            "kv_preload_total_us", "history_embedding_us_measured",
            "candidate_embedding_us_measured",
            "actual_layer0_base_us", "actual_layer0_recompute_us",
            "actual_layer0_late_us", "actual_repeated_early_us",
            "actual_repeated_late_us",
        ]
    }
    record.update({
        "model_v2_e2e_us": predicted_e2e,
        "model_v2_e2e_error_pct": 100.0 * (
            predicted_e2e / float(row["sim_time_us"]) - 1.0
        ),
        "model_candidate_embedding_us": row["prediction"][
            "candidate_embedding_us"
        ],
        "model_history_embedding_us": row["prediction"][
            "history_embedding_us"
        ] * scales["history_time_scale"],
        "model_kv_preload_total_us": row["prediction"][
            "layer_kv_preload_us"
        ] * scales["kv_time_scale"] * int(row["layers"]),
    })
    record.update({f"scale_{key}": value for key, value in scales.items()})
    component_rows.append(record)
    full_predictions[(row["context_id"], row["method"], row["point_label"])] = (
        predicted_e2e, row
    )
  write_csv(output / "model_v2_component_points.csv", component_rows)

  oracle_rows = read_csv(source / "grid_oracle.csv")
  oracle = {(row["context_id"], row["method"]): row for row in oracle_rows}
  loco_points: list[dict] = []
  loco_selection: list[dict] = []
  contexts = sorted({row["context_id"] for row in grid})
  for held_out in contexts:
    model = fit_model([row for row in grid if row["context_id"] != held_out])
    held_predictions: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for row in grid:
      if row["context_id"] != held_out:
        continue
      predicted_e2e, _ = predict_e2e(row, model)
      error = 100.0 * (predicted_e2e / float(row["sim_time_us"]) - 1.0)
      loco_points.append({
          "held_out_context": held_out,
          "method": row["method"],
          "ratio": row["actual_ratio"],
          "predicted_e2e_us": predicted_e2e,
          "simulator_e2e_us": row["sim_time_us"],
          "e2e_error_pct": error,
      })
      held_predictions[row["method"]].append((predicted_e2e, row))
    for method, candidates in held_predictions.items():
      selected = min(candidates, key=lambda item: (
          item[0], float(item[1]["actual_ratio"])
      ))[1]
      best = oracle[(held_out, method)]
      loco_selection.append({
          "held_out_context": held_out,
          "method": method,
          "selected_ratio": selected["actual_ratio"],
          "selected_simulator_e2e_us": selected["sim_time_us"],
          "grid_best_ratio": best["grid_best_ratio"],
          "grid_best_simulator_e2e_us": best["grid_best_sim_time_us"],
          "latency_gap_pct": 100.0 * (
              float(selected["sim_time_us"])
              / float(best["grid_best_sim_time_us"]) - 1.0
          ),
      })
  write_csv(output / "model_v2_loco_points.csv", loco_points)
  write_csv(output / "model_v2_loco_selection.csv", loco_selection)

  full_selection: list[dict] = []
  for key, best in sorted(oracle.items()):
    candidates = [
        (predicted, row) for (context, method, _), (predicted, row)
        in full_predictions.items() if (context, method) == key
    ]
    selected = min(candidates, key=lambda item: (
        item[0], float(item[1]["actual_ratio"])
    ))[1]
    full_selection.append({
        "context_id": key[0],
        "method": key[1],
        "selected_ratio": selected["actual_ratio"],
        "selected_simulator_e2e_us": selected["sim_time_us"],
        "grid_best_ratio": best["grid_best_ratio"],
        "grid_best_simulator_e2e_us": best["grid_best_sim_time_us"],
        "latency_gap_pct": 100.0 * (
            float(selected["sim_time_us"])
            / float(best["grid_best_sim_time_us"]) - 1.0
        ),
    })
  write_csv(output / "model_v2_full_fit_selection.csv", full_selection)

  full_errors = [float(row["model_v2_e2e_error_pct"]) for row in component_rows]
  loco_errors = [float(row["e2e_error_pct"]) for row in loco_points]
  full_gaps = [float(row["latency_gap_pct"]) for row in full_selection]
  loco_gaps = [float(row["latency_gap_pct"]) for row in loco_selection]
  summary = {
      "source_root": str(source),
      "grid_points": len(grid),
      "full_fit_e2e": comparison_summary(full_errors),
      "loco_e2e": comparison_summary(loco_errors),
      "full_fit_selection": comparison_summary(full_gaps),
      "loco_selection": comparison_summary(loco_gaps),
  }
  summary["gate"] = {
      "passed": (
          summary["loco_e2e"]["median_absolute_pct"] <= 5.0
          and summary["loco_e2e"]["p90_absolute_pct"] <= 10.0
          and summary["loco_selection"]["median_pct"] <= 2.0
          and summary["loco_selection"]["p90_absolute_pct"] <= 5.0
      ),
      "criteria": {
          "loco_e2e_median_absolute_pct_max": 5.0,
          "loco_e2e_p90_absolute_pct_max": 10.0,
          "loco_selection_median_gap_pct_max": 2.0,
          "loco_selection_p90_gap_pct_max": 5.0,
      },
  }
  (output / "model_v2_summary.json").write_text(
      json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )

  lines = [
      "# HSTU SSD IR phase-aware model v2",
      "",
      f"- Source grid points: {len(grid)}",
      f"- Full-fit E2E median absolute error: "
      f"{summary['full_fit_e2e']['median_absolute_pct']:.2f}%",
      f"- LOCO E2E median/p90 absolute error: "
      f"{summary['loco_e2e']['median_absolute_pct']:.2f}% / "
      f"{summary['loco_e2e']['p90_absolute_pct']:.2f}%",
      f"- LOCO selection median/p90 gap: "
      f"{summary['loco_selection']['median_pct']:.2f}% / "
      f"{summary['loco_selection']['p90_absolute_pct']:.2f}%",
      f"- Validation gate: {'PASS' if summary['gate']['passed'] else 'FAIL'}",
      "",
      "The calibration is written separately and does not replace the default.",
      "",
  ]
  (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  print(f"calibration={calibration_path}")


if __name__ == "__main__":
  main()
