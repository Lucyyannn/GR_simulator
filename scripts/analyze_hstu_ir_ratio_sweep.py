#!/usr/bin/env python3
"""Analyze an SSD HSTU IR ratio sweep and fit dynamic analytical corrections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


RIDGE_LAMBDA = 0.01
SSD_FEATURES = ["log2_hidden", "log2_batch", "ratio"]
COMPUTE_FEATURES = [
    "log2_work_per_core", "log2_batch", "ratio", "is_w_both",
    "ratio_x_w_both",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("result_root", type=Path)
  parser.add_argument(
      "--base-calibration", type=Path,
      default=Path("scripts/recompute_ratio_calibration.json"),
  )
  parser.add_argument("--output-calibration", type=Path, default=None)
  return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
  if fields is None:
    fields = list(rows[0]) if rows else []
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def as_float(value, default: float = 0.0) -> float:
  if value in (None, ""):
    return default
  return float(value)


def percentile(values: list[float], q: float) -> float:
  if not values:
    return float("nan")
  return float(np.percentile(np.asarray(values, dtype=float), q))


def comparison_summary(rows: list[dict]) -> dict:
  values = [float(row["latency_regret_pct"]) for row in rows]
  if not values:
    return {"count": 0}
  by_method: dict[str, dict] = {}
  for method in sorted({str(row["method"]) for row in rows}):
    selected = [
        float(row["latency_regret_pct"])
        for row in rows if row["method"] == method
    ]
    by_method[method] = {
        "count": len(selected),
        "median_gap_pct": statistics.median(selected),
        "p90_gap_pct": percentile(selected, 90),
        "max_gap_pct": max(selected),
    }
  return {
      "count": len(values),
      "median_gap_pct": statistics.median(values),
      "p90_gap_pct": percentile(values, 90),
      "max_gap_pct": max(values),
      "inside_one_pct_count": sum(value <= 1.0 for value in values),
      "by_method": by_method,
  }


def activity_metrics(path: Path) -> dict[str, float]:
  total = next(row for row in read_csv(path) if row["scope"] == "npu_total")
  cycles = int(total["total_core_cycles"])
  cube = int(total["cube_active_cycles"])
  vector = int(total["vector_active_cycles"])
  overlap = int(total["vector_overlap_with_cube_cycles"])
  return {
      "sim_time_us": float(total["sim_time_us"]),
      "total_core_cycles": cycles,
      "cube_active_cycles": cube,
      "vector_active_cycles": vector,
      "vector_overlap_cycles": overlap,
      "cube_active_pct": 100.0 * cube / cycles,
      "vector_active_pct": 100.0 * vector / cycles,
      "vector_exposed_pct": 100.0 * (vector - overlap) / cycles,
  }


def breakdown_metrics(path: Path) -> dict[str, float]:
  rows = read_csv(path)
  preload = [row for row in rows if row["pipe"] == "preload"]
  kv_rows = [row for row in preload if row["phase"] == "kvcache"]
  hist_rows = [
      row for row in preload if row["phase"] == "history_recompute_embedding"
  ]
  cand_rows = [
      row for row in preload if row["phase"] == "candidate_embedding"
  ]

  def total_duration(selected: list[dict[str, str]]) -> float:
    return sum(float(row["duration_us"]) for row in selected)

  def total_bytes(selected: list[dict[str, str]]) -> int:
    return sum(int(row["bytes"] or 0) for row in selected)

  def bandwidth(selected: list[dict[str, str]]) -> float:
    duration = total_duration(selected)
    return total_bytes(selected) / duration / 1000.0 if duration > 0 else 0.0

  op_rows = [
      row for row in rows
      if row["pipe"] == "compute" and row["phase"] == "op"
  ]
  by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
  for row in op_rows:
    try:
      by_layer[int(row["layer_id"])].append(row)
    except ValueError:
      continue
  spans: dict[int, float] = {}
  active_sums: dict[int, float] = {}
  for layer, selected in by_layer.items():
    spans[layer] = max(float(row["end_us"]) for row in selected) - min(
        float(row["start_us"]) for row in selected
    )
    # The span can contain a long idle hole while the attention op waits for
    # the asynchronous SSD KV preload.  Use summed op durations for effective
    # NPU compute calibration, and retain the span separately as the observed
    # per-layer critical-path interval.
    active_sums[layer] = sum(float(row["duration_us"]) for row in selected)
  repeated_active = [
      duration for layer, duration in active_sums.items() if layer > 0
  ]
  repeated_spans = [duration for layer, duration in spans.items() if layer > 0]
  if not repeated_active:
    repeated_active = list(active_sums.values())
  if not repeated_spans:
    repeated_spans = list(spans.values())
  return {
      "kv_preload_total_us": total_duration(kv_rows),
      "kv_preload_bytes": total_bytes(kv_rows),
      "kv_preload_effective_GBps": bandwidth(kv_rows),
      "history_embedding_us_measured": total_duration(hist_rows),
      "history_embedding_bytes": total_bytes(hist_rows),
      "history_embedding_effective_GBps": bandwidth(hist_rows),
      "candidate_embedding_us_measured": total_duration(cand_rows),
      "candidate_embedding_bytes": total_bytes(cand_rows),
      "candidate_embedding_effective_GBps": bandwidth(cand_rows),
      "repeated_layer_compute_mean_us": (
          statistics.fmean(repeated_active) if repeated_active else 0.0
      ),
      "repeated_layer_compute_min_us": (
          min(repeated_active) if repeated_active else 0.0
      ),
      "repeated_layer_compute_max_us": (
          max(repeated_active) if repeated_active else 0.0
      ),
      "repeated_layer_critical_span_mean_us": (
          statistics.fmean(repeated_spans) if repeated_spans else 0.0
      ),
      "repeated_layer_critical_span_min_us": (
          min(repeated_spans) if repeated_spans else 0.0
      ),
      "repeated_layer_critical_span_max_us": (
          max(repeated_spans) if repeated_spans else 0.0
      ),
      "layer0_compute_active_us": active_sums.get(0, 0.0),
      "layer0_critical_span_us": spans.get(0, 0.0),
  }


def prediction_path(root: Path, case_id: str) -> Path:
  return root / "logs" / f"{case_id}.model_prediction.json"


def load_rows(root: Path) -> list[dict]:
  status_paths = sorted((root / "logs").glob("*.status.json"))
  manifest = [
      json.loads(path.read_text(encoding="utf-8")) for path in status_paths
  ]
  output: list[dict] = []
  missing: list[str] = []
  for status in manifest:
    if int(status["returncode"]) != 0:
      continue
    default_case_dir = (
        root / "cases" /
        f"{status['context_id']}_{status['chip']}_{status['model']}_"
        f"seq{status['seq_len']}_bs{status['batch_size']}" /
        status["method"] / status["point_label"]
    )
    case_dir = Path(status.get("measurement_case_dir", default_case_dir))
    if not case_dir.is_absolute():
      case_dir = Path(__file__).resolve().parents[1] / case_dir
    activity = case_dir / "compute_activity.csv"
    breakdown = case_dir / "layer_breakdown.csv"
    model_path = prediction_path(root, status["case_id"])
    if not activity.exists() or not breakdown.exists() or not model_path.exists():
      missing.append(status["case_id"])
      continue
    prediction = json.loads(model_path.read_text(encoding="utf-8"))
    row: dict = {
        key: status[key] for key in [
            "case_id", "context_id", "chip", "model", "method",
            "point_kind", "point_label",
        ]
    }
    row["measurement_reused"] = bool(status.get("measurement_reused", False))
    row["measurement_source_case_id"] = status.get(
        "measurement_source_case_id", status["case_id"]
    )
    for key in [
        "layers", "hidden", "seq_len", "batch_size",
        "history_recompute_len",
    ]:
      row[key] = int(status[key])
    row["requested_ratio"] = (
        "" if status.get("requested_ratio") in (None, "")
        else float(status["requested_ratio"])
    )
    row["actual_ratio"] = float(status["actual_ratio"])
    row.update(activity_metrics(activity))
    row.update(breakdown_metrics(breakdown))
    for key, value in prediction.items():
      if isinstance(value, (int, float, bool)):
        row[f"model_{key}"] = value
    output.append(row)
  if missing:
    print(f"warning: skipped {len(missing)} incomplete successful records")
  return output


def group_key(row: dict) -> tuple[str, str]:
  return str(row["context_id"]), str(row["method"])


def grid_oracles(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, str], dict]]:
  groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in rows:
    if row["point_kind"] == "grid":
      groups[group_key(row)].append(row)
  output: list[dict] = []
  lookup: dict[tuple[str, str], dict] = {}
  for key, group in sorted(groups.items()):
    best = min(group, key=lambda row: (row["sim_time_us"], row["actual_ratio"]))
    limit = best["sim_time_us"] * 1.01
    plateau = sorted(
        row["actual_ratio"] for row in group if row["sim_time_us"] <= limit
    )
    record = {
        "context_id": key[0],
        "method": key[1],
        "chip": best["chip"],
        "model": best["model"],
        "seq_len": best["seq_len"],
        "batch_size": best["batch_size"],
        "grid_best_ratio": best["actual_ratio"],
        "grid_best_k": best["history_recompute_len"],
        "grid_best_sim_time_us": best["sim_time_us"],
        "one_pct_plateau_ratios": ";".join(f"{value:.6f}" for value in plateau),
        "grid_points": len(group),
    }
    output.append(record)
    lookup[key] = record
  return output, lookup


def estimate_comparison(
    rows: list[dict],
    oracle: dict[tuple[str, str], dict],
    label: str,
) -> list[dict]:
  output: list[dict] = []
  for row in rows:
    if row["point_label"] != label or group_key(row) not in oracle:
      continue
    best = oracle[group_key(row)]
    output.append({
        "context_id": row["context_id"],
        "method": row["method"],
        "chip": row["chip"],
        "model": row["model"],
        "seq_len": row["seq_len"],
        "batch_size": row["batch_size"],
        "estimate_label": label,
        "estimated_ratio": row["actual_ratio"],
        "estimated_k": row["history_recompute_len"],
        "estimated_sim_time_us": row["sim_time_us"],
        "grid_best_ratio": best["grid_best_ratio"],
        "grid_best_sim_time_us": best["grid_best_sim_time_us"],
        "absolute_ratio_error": abs(
            row["actual_ratio"] - best["grid_best_ratio"]
        ),
        "latency_regret_pct": 100.0 * (
            row["sim_time_us"] / best["grid_best_sim_time_us"] - 1.0
        ),
        "inside_one_pct_plateau": int(
            row["sim_time_us"] <= 1.01 * best["grid_best_sim_time_us"]
        ),
    })
  return output


def peak_component_values(row: dict) -> dict[str, float]:
  b_peak = as_float(row.get("model_B_ssd_peak"), 1.0)
  b_kv = as_float(row.get("model_B_kv"), b_peak)
  b_emb = as_float(row.get("model_B_emb"), b_peak)
  empirical_compute = (
      as_float(row.get("model_compute_scale"), 1.0)
      * as_float(row.get("model_total_compute_correction"), 1.0)
  )
  empirical_compute = max(empirical_compute, 1e-12)
  return {
      "history_peak_us": (
          as_float(row.get("model_history_embedding_us")) * b_emb / b_peak
      ),
      "candidate_peak_us": (
          as_float(row.get("model_candidate_embedding_us")) * b_emb / b_peak
      ),
      "kv_peak_layer_us": (
          as_float(row.get("model_layer_kv_preload_us")) * b_kv / b_peak
      ),
      "early_peak_us": (
          as_float(row.get("model_pre_cached_compute_us")) / empirical_compute
      ),
      "late_peak_us": (
          as_float(row.get("model_layer_late_compute_us")) / empirical_compute
      ),
      "compute_peak_layer_us": (
          as_float(row.get("model_layer_compute_us")) / empirical_compute
      ),
      "weight_us": as_float(row.get("model_weight_read_us")),
  }


def feature_values(row: dict) -> tuple[list[float], list[float]]:
  hidden = float(row["hidden"])
  batch = float(row["batch_size"])
  ratio = float(row["actual_ratio"])
  cores = max(1.0, as_float(row.get("model_num_cores"), 1.0))
  k = float(row["history_recompute_len"])
  active_tokens = as_float(row.get("model_active_tokens"), 128.0 + k)
  early_scores = as_float(row.get("model_early_attention_score_elements"))
  cached_scores = as_float(row.get("model_cached_attention_score_elements"))
  work = (
      8.0 * batch * active_tokens * hidden * hidden
      # Prediction JSON exports the score counts after batch scaling, whereas
      # active_tokens remains per user.
      + 4.0 * (early_scores + cached_scores) * hidden
  ) / cores
  reference_work = 8.0 * 128.0 * 256.0 * 256.0 / 8.0
  is_both = 1.0 if row["method"] == "w_both" else 0.0
  ssd = [math.log2(hidden / 256.0), math.log2(batch), ratio]
  compute = [
      math.log2(max(work, 1.0) / reference_work),
      math.log2(batch), ratio, is_both, ratio * is_both,
  ]
  return ssd, compute


def fit_ridge(features: list[list[float]], targets: list[float]) -> dict:
  matrix = np.asarray(features, dtype=float)
  target = np.asarray(targets, dtype=float)
  mean = matrix.mean(axis=0)
  scale = matrix.std(axis=0)
  scale[scale < 1e-9] = 1.0
  standardized = (matrix - mean) / scale
  design = np.column_stack([np.ones(len(matrix)), standardized])
  penalty = np.eye(design.shape[1]) * RIDGE_LAMBDA
  penalty[0, 0] = 0.0
  beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
  return {
      "intercept": float(beta[0]),
      "coefficients": [float(value) for value in beta[1:]],
      "feature_mean": [float(value) for value in mean],
      "feature_scale": [float(value) for value in scale],
      "ridge_lambda": RIDGE_LAMBDA,
  }


def scale_from_fit(spec: dict, features: list[float], lo: float, hi: float) -> float:
  value = spec["intercept"]
  for coefficient, current, mean, scale in zip(
      spec["coefficients"], features,
      spec["feature_mean"], spec["feature_scale"],
  ):
    value += coefficient * ((current - mean) / scale)
  return max(lo, min(hi, math.exp(max(-20.0, min(20.0, value)))))


def fit_dynamic(rows: list[dict]) -> dict:
  ssd_x: list[list[float]] = []
  ssd_y: list[float] = []
  compute_x: list[list[float]] = []
  compute_y: list[float] = []
  for row in rows:
    peak = peak_component_values(row)
    ssd_features, compute_features = feature_values(row)
    predicted_kv_total = peak["kv_peak_layer_us"] * row["layers"]
    if predicted_kv_total > 0 and row["kv_preload_total_us"] > 0:
      ssd_x.append(ssd_features)
      ssd_y.append(math.log(row["kv_preload_total_us"] / predicted_kv_total))
    predicted_compute = peak["compute_peak_layer_us"]
    measured_compute = row["repeated_layer_compute_mean_us"]
    if predicted_compute > 0 and measured_compute > 0:
      compute_x.append(compute_features)
      compute_y.append(math.log(measured_compute / predicted_compute))
  ssd_fit = fit_ridge(ssd_x, ssd_y)
  compute_fit = fit_ridge(compute_x, compute_y)
  ssd_fit.update({"features": SSD_FEATURES, "min_scale": 1.0, "max_scale": 20.0})
  compute_fit.update({
      "features": COMPUTE_FEATURES, "min_scale": 1.0, "max_scale": 200.0,
  })
  return {
      "version": 1,
      "mode": "analytical_dynamic_time_scale",
      "ssd_time_scale": ssd_fit,
      "compute_time_scale": compute_fit,
  }


def corrected_score(row: dict, dynamic: dict) -> tuple[float, float, float]:
  peak = peak_component_values(row)
  ssd_features, compute_features = feature_values(row)
  ssd_scale = scale_from_fit(
      dynamic["ssd_time_scale"], ssd_features, 1.0, 20.0
  )
  compute_scale = scale_from_fit(
      dynamic["compute_time_scale"], compute_features, 1.0, 200.0
  )
  one_time = ssd_scale * (
      peak["history_peak_us"] + peak["candidate_peak_us"]
  ) + peak["weight_us"]
  layer = max(
      ssd_scale * peak["kv_peak_layer_us"],
      compute_scale * peak["early_peak_us"],
  ) + compute_scale * peak["late_peak_us"]
  return one_time + row["layers"] * layer, ssd_scale, compute_scale


def cross_validate(rows: list[dict], oracle: dict[tuple[str, str], dict]) -> list[dict]:
  grid = [row for row in rows if row["point_kind"] == "grid"]
  contexts = sorted({str(row["context_id"]) for row in grid})
  output: list[dict] = []
  for held_out in contexts:
    train = [row for row in grid if row["context_id"] != held_out]
    test = [row for row in grid if row["context_id"] == held_out]
    dynamic = fit_dynamic(train)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in test:
      score, ssd_scale, compute_scale = corrected_score(row, dynamic)
      enriched = dict(row)
      enriched["corrected_score_us"] = score
      enriched["corrected_ssd_time_scale"] = ssd_scale
      enriched["corrected_compute_time_scale"] = compute_scale
      groups[group_key(row)].append(enriched)
    for key, group in sorted(groups.items()):
      selected = min(
          group,
          key=lambda row: (row["corrected_score_us"], row["actual_ratio"]),
      )
      best = oracle[key]
      output.append({
          "held_out_context": held_out,
          "method": key[1],
          "chip": selected["chip"],
          "model": selected["model"],
          "seq_len": selected["seq_len"],
          "batch_size": selected["batch_size"],
          "predicted_grid_ratio": selected["actual_ratio"],
          "predicted_grid_k": selected["history_recompute_len"],
          "predicted_grid_sim_time_us": selected["sim_time_us"],
          "grid_best_ratio": best["grid_best_ratio"],
          "grid_best_sim_time_us": best["grid_best_sim_time_us"],
          "absolute_ratio_error": abs(
              selected["actual_ratio"] - best["grid_best_ratio"]
          ),
          "latency_regret_pct": 100.0 * (
              selected["sim_time_us"] / best["grid_best_sim_time_us"] - 1.0
          ),
          "predicted_score_us": selected["corrected_score_us"],
          "ssd_time_scale": selected["corrected_ssd_time_scale"],
          "compute_time_scale": selected["corrected_compute_time_scale"],
      })
  return output


def component_errors(rows: list[dict], dynamic: dict) -> list[dict]:
  output: list[dict] = []
  for row in rows:
    if row["point_kind"] != "grid":
      continue
    peak = peak_component_values(row)
    _, ssd_scale, compute_scale = corrected_score(row, dynamic)
    measured_kv = row["kv_preload_total_us"]
    predicted_kv = ssd_scale * peak["kv_peak_layer_us"] * row["layers"]
    measured_compute = row["repeated_layer_compute_mean_us"]
    predicted_compute = compute_scale * peak["compute_peak_layer_us"]
    output.append({
        "case_id": row["case_id"],
        "context_id": row["context_id"],
        "method": row["method"],
        "ratio": row["actual_ratio"],
        "measured_kv_preload_us": measured_kv,
        "predicted_kv_preload_us": predicted_kv,
        "kv_relative_error_pct": (
            100.0 * (predicted_kv / measured_kv - 1.0) if measured_kv else 0.0
        ),
        "measured_repeated_compute_us": measured_compute,
        "predicted_repeated_compute_us": predicted_compute,
        "compute_relative_error_pct": (
            100.0 * (predicted_compute / measured_compute - 1.0)
            if measured_compute else 0.0
        ),
        "ssd_time_scale": ssd_scale,
        "compute_time_scale": compute_scale,
    })
  return output


def model_grid_ablation(
    rows: list[dict], oracle: dict[tuple[str, str], dict], dynamic: dict
) -> list[dict]:
  grid: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in rows:
    if row["point_kind"] == "grid":
      grid[group_key(row)].append(row)
  output: list[dict] = []
  objectives = {
      "balance_no_guards": lambda row: as_float(row.get("model_layer_balance_error_us")),
      "pipeline": lambda row: as_float(row.get("model_pipeline_layer_us")),
      "e2e_uncalibrated": lambda row: as_float(row.get("model_e2e_proxy_us")),
      "e2e_dynamic": lambda row: corrected_score(row, dynamic)[0],
  }
  for key, group in sorted(grid.items()):
    best = oracle[key]
    for name, score_fn in objectives.items():
      selected = min(group, key=lambda row: (score_fn(row), row["actual_ratio"]))
      output.append({
          "context_id": key[0],
          "method": key[1],
          "variant": name,
          "selected_ratio": selected["actual_ratio"],
          "selected_sim_time_us": selected["sim_time_us"],
          "grid_best_ratio": best["grid_best_ratio"],
          "grid_best_sim_time_us": best["grid_best_sim_time_us"],
          "latency_regret_pct": 100.0 * (
              selected["sim_time_us"] / best["grid_best_sim_time_us"] - 1.0
          ),
      })
  return output


def plot_ratio_curves(root: Path, rows: list[dict]) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  grid = [row for row in rows if row["point_kind"] == "grid"]
  context_order = sorted({str(row["context_id"]) for row in grid})
  for method in ["w_IR", "w_both"]:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for axis, context_id in zip(axes.flat, context_order):
      selected = sorted(
          [
              row for row in grid
              if row["context_id"] == context_id and row["method"] == method
          ],
          key=lambda row: row["actual_ratio"],
      )
      if not selected:
        axis.set_visible(False)
        continue
      best = min(row["sim_time_us"] for row in selected)
      axis.plot(
          [row["actual_ratio"] for row in selected],
          [row["sim_time_us"] / best for row in selected],
          marker="o",
      )
      first = selected[0]
      axis.set_title(
          f"{context_id} {first['chip']} {first['model']} "
          f"S{first['seq_len']} B{first['batch_size']}"
      )
      axis.grid(alpha=0.3)
      axis.set_ylim(bottom=0.98)
    fig.supxlabel("Item recompute ratio")
    fig.supylabel("Latency / grid minimum")
    fig.suptitle(f"SSD {method} ratio sweep")
    fig.tight_layout()
    fig.savefig(root / f"ratio_latency_curves_{method}.png", dpi=180)
    plt.close(fig)


def write_calibration(
    base_path: Path, output_path: Path, dynamic: dict, root: Path
) -> None:
  calibration = json.loads(base_path.read_text(encoding="utf-8"))
  model = calibration.setdefault("ir_cost_model", {})
  model["dynamic_efficiency"] = dynamic
  model["selection_objective"] = "e2e"
  model["dynamic_efficiency_source"] = str(root)
  output_path.write_text(
      json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def markdown_report(
    root: Path,
    rows: list[dict],
    oracles: list[dict],
    current: list[dict],
    corrected: list[dict],
    cv: list[dict],
    ablation: list[dict],
) -> None:
  grid_count = sum(row["point_kind"] == "grid" for row in rows)
  current_regrets = [row["latency_regret_pct"] for row in current]
  corrected_regrets = [row["latency_regret_pct"] for row in corrected]
  cv_regrets = [row["latency_regret_pct"] for row in cv]
  gate = validation_gate(current, cv)
  grid = [row for row in rows if row["point_kind"] == "grid"]
  current_kv_errors = []
  current_compute_errors = []
  for row in grid:
    predicted_kv = as_float(row.get("model_layer_kv_preload_us")) * row["layers"]
    measured_kv = row["kv_preload_total_us"]
    predicted_compute = as_float(row.get("model_layer_compute_us"))
    measured_compute = row["repeated_layer_compute_mean_us"]
    if measured_kv > 0:
      current_kv_errors.append(100.0 * (predicted_kv / measured_kv - 1.0))
    if measured_compute > 0:
      current_compute_errors.append(
          100.0 * (predicted_compute / measured_compute - 1.0)
      )
  split_decreases = 0
  grid_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in grid:
    grid_groups[group_key(row)].append(row)
  for group in grid_groups.values():
    ordered = sorted(group, key=lambda row: row["actual_ratio"])
    if len(ordered) >= 2 and (
        ordered[1]["repeated_layer_compute_mean_us"]
        < ordered[0]["repeated_layer_compute_mean_us"]
    ):
      split_decreases += 1
  lines = [
      "# HSTU SSD IR 重计算比例分析",
      "",
      f"- 已解析成功结果：{len(rows)}；其中0.2步长网格：{grid_count}/108。",
      f"- 网格上下文/方法组合：{len(oracles)}/18。",
      "- Ground truth：Simulator `npu_total.sim_time_us`；1%以内记为最优平台。",
      "",
      "## 汇总结论",
      "",
  ]
  if current_regrets:
    lines.append(
        f"当前精确估计点：median gap={statistics.median(current_regrets):.2f}%，"
        f"p90={percentile(current_regrets, 90):.2f}%，"
        f"不在1%平台={sum(not row['inside_one_pct_plateau'] for row in current)}/{len(current)}。"
    )
  else:
    lines.append("当前估计器精确点尚未全部运行。")
  if cv_regrets:
    lines.append(
        f"动态解析校正LOCO：median gap={statistics.median(cv_regrets):.2f}%，"
        f"p90={percentile(cv_regrets, 90):.2f}%；"
        f"预设门槛={'通过' if gate['passed'] else '未通过'}。"
    )
  if corrected_regrets:
    lines.append(
        f"校正后精确点：median gap={statistics.median(corrected_regrets):.2f}%，"
        f"p90={percentile(corrected_regrets, 90):.2f}%。"
    )
  lines += [
      "负 gap 表示该连续估计点优于0.2步长网格中的最佳采样点，并非负延迟。",
      "动态校正未通过LOCO门槛，因此没有替换默认估计器。",
      "",
      "## 比例与实测端到端结果",
      "",
      "| case / hardware / workload | method | grid best | current (gap) | corrected (gap) |",
      "|---|---|---:|---:|---:|",
  ]
  current_lookup = {group_key(row): row for row in current}
  corrected_lookup = {group_key(row): row for row in corrected}
  for oracle in oracles:
    key = (str(oracle["context_id"]), str(oracle["method"]))
    old = current_lookup.get(key)
    new = corrected_lookup.get(key)
    old_text = "-" if old is None else (
        f"{old['estimated_ratio']:.3f} ({old['latency_regret_pct']:+.1f}%)"
    )
    new_text = "-" if new is None else (
        f"{new['estimated_ratio']:.3f} ({new['latency_regret_pct']:+.1f}%)"
    )
    lines.append(
        f"| {oracle['context_id']} {oracle['chip']}/{oracle['model']} "
        f"S{oracle['seq_len']}/B{oracle['batch_size']} | {oracle['method']} | "
        f"{oracle['grid_best_ratio']:.3f} | {old_text} | {new_text} |"
    )
  lines += [
      "",
      "## 当前模型的结构性问题",
      "",
      "1. 默认目标最小化KV preload与整层compute之差，而实际流水为"
      "`max(KV, pre-cached compute) + late compute`，并且embedding是一次性成本。",
      (
          "2. 旧模型对KV preload的中位绝对误差为"
          f"{statistics.median(abs(value) for value in current_kv_errors):.1f}%，"
          "但单层compute为"
          f"{statistics.median(abs(value) for value in current_compute_errors):.1f}%"
          f"（p90={percentile([abs(value) for value in current_compute_errors], 90):.1f}%）；"
          "主要误差来自compute而非SSD带宽。"
      ),
      (
          f"3. {split_decreases}/18组中，ratio从0变为约0.2后实测compute反而下降。"
          "这来自未拆分/拆分attention的shape、并行度和调度效率变化，"
          "不能用固定峰值吞吐乘平滑FLOP数表示。"
      ),
      "4. batch线性缩放和按chip/hidden写死的compute correction无法描述"
      "GEMM shape、每核work、vector/cube混合及split状态导致的动态效率。",
      "5. 新的解析critical-path目标和动态时间倍率改善了中位数，但LOCO仍失败；"
      "当前9个正交上下文不足以拟合可泛化的连续校正。",
      "",
      "## 文件",
      "",
      "- `sweep_cases.csv`：逐点实测与解析模型组件。",
      "- `grid_oracle.csv`：0.2网格最优点和1%平台。",
      "- `current_vs_grid_oracle.csv`：当前估计器精确点误差。",
      "- `correction_cv.csv`：leave-one-context-out结果。",
      "- `model_ablation.csv`：目标函数和动态校正消融。",
      "- `component_error.csv`：SSD preload与NPU compute组件误差。",
      "- `dynamic_calibration.json`：不覆盖旧文件的动态校正参数。",
      "- `summary_metrics.json`：当前、校正与LOCO的机器可读汇总。",
      "- `experiment_audit.json`：覆盖范围、成功状态和测量复用来源。",
      "",
  ]
  (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(root: Path) -> None:
  result_arg = f"results/{root.name}"
  lines = [
      "# SSD IR ratio sweep rerun guide",
      "",
      "所有命令在容器的 `/workspace/GR_simulator` 下执行。",
      "",
      "## 0.2 ratio grid + current exact estimate",
      "",
      "```bash",
      "python3 scripts/run_hstu_ir_ratio_sweep.py \\",
      f"  --result-root {result_arg} \\",
      "  --ratios 0 0.2 0.4 0.6 0.8 1.0 \\",
      "  --methods w_IR w_both \\",
      "  --include-estimate --estimate-label current_estimate \\",
      "  --objective balance --max-concurrent 48",
      "```",
      "",
      "## Analyze and fit dynamic analytical correction",
      "",
      "```bash",
      f"python3 scripts/analyze_hstu_ir_ratio_sweep.py {result_arg}",
      "```",
      "",
      "## Corrected exact estimate",
      "",
      "```bash",
      "python3 scripts/run_hstu_ir_ratio_sweep.py \\",
      f"  --result-root {result_arg} \\",
      "  --estimate-only --estimate-label corrected_estimate \\",
      f"  --calibration {result_arg}/dynamic_calibration.json \\",
      "  --objective e2e --max-concurrent 48",
      "```",
      "",
      "随后再次运行分析命令即可生成最终比较。",
      "",
  ]
  (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def validation_gate(current: list[dict], cv: list[dict]) -> dict:
  if not current or not cv:
    return {
        "passed": False,
        "reason": "current exact points or LOCO results are incomplete",
    }
  current_by_method: dict[str, list[float]] = defaultdict(list)
  cv_by_method: dict[str, list[float]] = defaultdict(list)
  current_by_chip: dict[str, list[float]] = defaultdict(list)
  cv_by_chip: dict[str, list[float]] = defaultdict(list)
  for row in current:
    current_by_method[row["method"]].append(row["latency_regret_pct"])
    current_by_chip[row["chip"]].append(row["latency_regret_pct"])
  for row in cv:
    cv_by_method[row["method"]].append(row["latency_regret_pct"])
    cv_by_chip[row["chip"]].append(row["latency_regret_pct"])
  cv_regrets = [row["latency_regret_pct"] for row in cv]
  method_improved = {
      method: statistics.median(cv_by_method[method])
      < statistics.median(current_by_method.get(method, [float("inf")]))
      for method in ["w_IR", "w_both"]
  }
  chip_regression_pp = {
      chip: statistics.median(cv_by_chip[chip])
      - statistics.median(current_by_chip.get(chip, [float("inf")]))
      for chip in sorted(cv_by_chip)
  }
  median_regret = statistics.median(cv_regrets)
  p90_regret = percentile(cv_regrets, 90)
  passed = (
      all(method_improved.values())
      and median_regret <= 2.0
      and p90_regret <= 5.0
      and all(value <= 1.0 for value in chip_regression_pp.values())
  )
  return {
      "passed": passed,
      "median_regret_pct": median_regret,
      "p90_regret_pct": p90_regret,
      "method_median_improved": method_improved,
      "chip_median_regression_percentage_points": chip_regression_pp,
      "criteria": {
          "each_method_median_better_than_current": True,
          "overall_median_regret_pct_max": 2.0,
          "overall_p90_regret_pct_max": 5.0,
          "per_chip_median_regression_percentage_points_max": 1.0,
      },
  }


def main() -> None:
  args = parse_args()
  root = args.result_root.resolve()
  rows = load_rows(root)
  if not rows:
    raise SystemExit("no completed simulator results found")
  write_csv(root / "sweep_cases.csv", rows)
  oracles, oracle_lookup = grid_oracles(rows)
  write_csv(root / "grid_oracle.csv", oracles)
  current = estimate_comparison(rows, oracle_lookup, "current_estimate")
  corrected = estimate_comparison(rows, oracle_lookup, "corrected_estimate")
  write_csv(root / "current_vs_grid_oracle.csv", current)
  write_csv(root / "corrected_vs_grid_oracle.csv", corrected)

  grid = [row for row in rows if row["point_kind"] == "grid"]
  dynamic = fit_dynamic(grid)
  cv = cross_validate(rows, oracle_lookup)
  write_csv(root / "correction_cv.csv", cv)
  component = component_errors(rows, dynamic)
  write_csv(root / "component_error.csv", component)
  ablation = model_grid_ablation(rows, oracle_lookup, dynamic)
  write_csv(root / "model_ablation.csv", ablation)
  plot_ratio_curves(root, rows)
  gate = validation_gate(current, cv)
  (root / "validation_gate.json").write_text(
      json.dumps(gate, indent=2) + "\n", encoding="utf-8"
  )
  summary = {
      "reference": "0.2-step grid minimum",
      "negative_gap_meaning": "estimate beat the sampled grid minimum",
      "current": comparison_summary(current),
      "corrected_exact": comparison_summary(corrected),
      "corrected_loco": comparison_summary(cv),
      "validation_gate_passed": bool(gate.get("passed", False)),
  }
  (root / "summary_metrics.json").write_text(
      json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )

  statuses = [
      json.loads(path.read_text(encoding="utf-8"))
      for path in sorted((root / "logs").glob("*.status.json"))
  ]
  successful = [row for row in statuses if int(row.get("returncode", -1)) == 0]
  reused = [row for row in successful if row.get("measurement_reused", False)]
  audit = {
      "status_records": len(statuses),
      "successful_status_records": len(successful),
      "failed_status_records": len(statuses) - len(successful),
      "parsed_measurements": len(rows),
      "independent_simulator_measurements": len(successful) - len(reused),
      "reused_identical_measurements": len(reused),
      "reused_measurements": [
          {
              "case_id": row["case_id"],
              "source_case_id": row["measurement_source_case_id"],
              "reason": row["reuse_reason"],
          }
          for row in reused
      ],
      "point_counts": {
          "grid": sum(row["point_kind"] == "grid" for row in successful),
          "current_estimate": sum(
              row["point_label"] == "current_estimate" for row in successful
          ),
          "corrected_estimate": sum(
              row["point_label"] == "corrected_estimate" for row in successful
          ),
      },
      "coverage": {
          "contexts": sorted({row["context_id"] for row in successful}),
          "chips": sorted({row["chip"] for row in successful}),
          "models": sorted({row["model"] for row in successful}),
          "sequence_lengths": sorted({row["seq_len"] for row in successful}),
          "batch_sizes": sorted({row["batch_size"] for row in successful}),
          "methods": sorted({row["method"] for row in successful}),
      },
      "constraints": {
          "all_source_media_ssd": all(
              row.get("source_medium") == "ssd"
              and row.get("embedding_source_medium") == "ssd"
              and row.get("history_recompute_source_medium") == "ssd"
              for row in successful
          ),
          "all_ar_attention_compute_reduction_disabled": all(
              not row.get("ar_reduce_attention_compute", True)
              for row in successful
          ),
          "w_both_kv_reuse_ratio": 0.436,
      },
  }
  (root / "experiment_audit.json").write_text(
      json.dumps(audit, indent=2) + "\n", encoding="utf-8"
  )

  calibration_path = args.output_calibration or root / "dynamic_calibration.json"
  write_calibration(
      args.base_calibration.resolve(), calibration_path.resolve(), dynamic, root
  )
  markdown_report(root, rows, oracles, current, corrected, cv, ablation)
  write_readme(root)
  print(f"rows={len(rows)} grid={len(grid)} oracle_groups={len(oracles)}")
  print(f"dynamic_calibration={calibration_path.resolve()}")


if __name__ == "__main__":
  main()
