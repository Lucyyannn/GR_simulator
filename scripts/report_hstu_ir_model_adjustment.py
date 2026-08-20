#!/usr/bin/env python3
"""Compare estimator choices with measured end-to-end HSTU IR optima."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import analyze_hstu_ir_ratio_sweep as sweep_analysis
import analyze_hstu_ir_model_v2 as v2_analysis


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("source_root", type=Path)
  parser.add_argument("model_v2_root", type=Path)
  parser.add_argument("output_root", type=Path)
  parser.add_argument("--fine-root", type=Path, action="append", default=[])
  parser.add_argument(
      "--model-revision-root", type=Path, action="append", default=[],
      help="Model-fit result roots to include in the revision audit table.",
  )
  parser.add_argument("--max-workers", type=int, default=24)
  return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def penalty(row: dict, best: dict) -> float:
  return 100.0 * (float(row["sim_time_us"]) / float(best["sim_time_us"]) - 1.0)


def percentile_summary(values: list[float]) -> dict[str, float]:
  data = np.asarray(values, dtype=float)
  return {
      "median_pct": float(np.median(data)),
      "p90_pct": float(np.percentile(data, 90)),
      "max_pct": float(np.max(data)),
  }


def predict_fixed_point(row: dict, calibration: Path) -> float:
  command = v2_analysis.model_command(row, calibration)
  result = subprocess.run(
      command, check=True, text=True, capture_output=True,
  )
  return float(json.loads(result.stdout)["e2e_proxy_us"])


def main() -> None:
  args = parse_args()
  source = args.source_root.resolve()
  model_v2 = args.model_v2_root.resolve()
  output = args.output_root.resolve()
  output.mkdir(parents=True, exist_ok=True)

  source_rows = read_csv(source / "sweep_cases.csv")
  observed: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
  labels: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
  predictions: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)

  for row in source_rows:
    key = (row["context_id"], row["method"])
    item = {
        "context_id": row["context_id"],
        "method": row["method"],
        "chip": row["chip"],
        "model": row["model"],
        "seq_len": int(row["seq_len"]),
        "batch_size": int(row["batch_size"]),
        "k": int(row["history_recompute_len"]),
        "ratio": float(row["actual_ratio"]),
        "sim_time_us": float(row["sim_time_us"]),
        "source": "coarse_or_estimate",
    }
    prior = observed[key].get(item["k"])
    if prior is None or item["sim_time_us"] < prior["sim_time_us"]:
      observed[key][item["k"]] = item
    if row["point_label"] in {"current_estimate", "corrected_estimate"}:
      labels[key][row["point_label"]] = item
    if row["point_kind"] == "grid":
      labels[key].setdefault("grid", []).append(item)

  component_rows = read_csv(model_v2 / "model_v2_component_points.csv")
  for row in component_rows:
    key = (row["context_id"], row["method"])
    predictions[key][int(row["history_recompute_len"])] = float(
        row["model_v2_e2e_us"]
    )

  fine_prediction_errors: list[float] = []
  fine_point_count = 0
  fine_stage_rows: list[tuple[int, dict]] = []
  for stage, root in enumerate(args.fine_root, start=1):
    for row in sweep_analysis.load_rows(root.resolve()):
      fine_stage_rows.append((stage, row))

  calibration = model_v2 / "recompute_ratio_calibration_v2.json"
  with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    refreshed_predictions = list(executor.map(
        lambda item: predict_fixed_point(item[1], calibration), fine_stage_rows
    ))

  for (stage, row), prediction in zip(fine_stage_rows, refreshed_predictions):
    key = (row["context_id"], row["method"])
    item = {
        "context_id": row["context_id"],
        "method": row["method"],
        "chip": row["chip"],
        "model": row["model"],
        "seq_len": int(row["seq_len"]),
        "batch_size": int(row["batch_size"]),
        "k": int(row["history_recompute_len"]),
        "ratio": float(row["actual_ratio"]),
        "sim_time_us": float(row["sim_time_us"]),
        "source": f"fine_stage{stage}",
    }
    prior = observed[key].get(item["k"])
    if prior is None or item["sim_time_us"] < prior["sim_time_us"]:
      observed[key][item["k"]] = item
    if prediction > 0:
      predictions[key][item["k"]] = prediction
      fine_prediction_errors.append(
          100.0 * (prediction / item["sim_time_us"] - 1.0)
      )
    fine_point_count += 1

  comparisons: list[dict] = []
  for key in sorted(observed):
    points = list(observed[key].values())
    best = min(points, key=lambda row: (row["sim_time_us"], row["ratio"]))
    grid = min(labels[key]["grid"], key=lambda row: row["sim_time_us"])
    current = labels[key]["current_estimate"]
    corrected = labels[key]["corrected_estimate"]
    predicted_candidates = [
        (value, observed[key][k]) for k, value in predictions[key].items()
        if k in observed[key]
    ]
    _, selected = min(predicted_candidates, key=lambda pair: pair[0])
    selected_prediction = predictions[key][selected["k"]]
    comparisons.append({
        "context_id": key[0],
        "method": key[1],
        "chip": best["chip"],
        "model": best["model"],
        "seq_len": best["seq_len"],
        "batch_size": best["batch_size"],
        "current_ratio": current["ratio"],
        "current_sim_time_us": current["sim_time_us"],
        "current_penalty_pct": penalty(current, best),
        "corrected_ratio": corrected["ratio"],
        "corrected_sim_time_us": corrected["sim_time_us"],
        "corrected_penalty_pct": penalty(corrected, best),
        "coarse_best_ratio": grid["ratio"],
        "coarse_best_sim_time_us": grid["sim_time_us"],
        "coarse_penalty_pct": penalty(grid, best),
        "phase_model_selected_ratio": selected["ratio"],
        "phase_model_predicted_time_us": selected_prediction,
        "phase_model_selected_sim_time_us": selected["sim_time_us"],
        "phase_model_selected_prediction_error_pct": 100.0 * (
            selected_prediction / selected["sim_time_us"] - 1.0
        ),
        "phase_model_selection_penalty_pct": penalty(selected, best),
        "best_observed_ratio": best["ratio"],
        "best_observed_k": best["k"],
        "best_observed_sim_time_us": best["sim_time_us"],
        "best_observed_source": best["source"],
        "observed_unique_k": len(points),
    })

  write_csv(output / "estimator_vs_simulator_optimum.csv", comparisons)

  revision_rows: list[dict] = []
  for root in args.model_revision_root:
    resolved = root.resolve()
    revision_summary = json.loads(
        (resolved / "model_v2_summary.json").read_text(encoding="utf-8")
    )
    component_points = read_csv(resolved / "model_v2_component_points.csv")
    revision_rows.append({
        "revision": resolved.name,
        "full_fit_median_abs_e2e_error_pct": revision_summary[
            "full_fit_e2e"
        ]["median_absolute_pct"],
        "loco_median_abs_e2e_error_pct": revision_summary[
            "loco_e2e"
        ]["median_absolute_pct"],
        "loco_p90_abs_e2e_error_pct": revision_summary[
            "loco_e2e"
        ]["p90_absolute_pct"],
        "loco_max_abs_e2e_error_pct": revision_summary[
            "loco_e2e"
        ]["max_absolute_pct"],
        "median_layer0_base_scale": float(np.median([
            float(row["scale_layer0_base_time_scale"])
            for row in component_points
        ])),
        "median_repeated_early_scale": float(np.median([
            float(row["scale_repeated_early_time_scale"])
            for row in component_points
        ])),
        "median_repeated_late_scale": float(np.median([
            float(row["scale_repeated_late_time_scale"])
            for row in component_points
        ])),
        "validation_gate_passed": revision_summary["gate"]["passed"],
    })
  if revision_rows:
    write_csv(output / "model_revision_validation.csv", revision_rows)
  metrics = {
      "groups": len(comparisons),
      "fine_points": fine_point_count,
      "groups_where_fine_improves_coarse": sum(
          float(row["coarse_penalty_pct"]) > 1e-9 for row in comparisons
      ),
      "current_estimate_penalty": percentile_summary([
          float(row["current_penalty_pct"]) for row in comparisons
      ]),
      "corrected_estimate_penalty": percentile_summary([
          float(row["corrected_penalty_pct"]) for row in comparisons
      ]),
      "coarse_grid_penalty": percentile_summary([
          float(row["coarse_penalty_pct"]) for row in comparisons
      ]),
      "phase_model_selection_penalty": percentile_summary([
          float(row["phase_model_selection_penalty_pct"]) for row in comparisons
      ]),
      "phase_model_fine_point_e2e_absolute_error": percentile_summary(
          [abs(value) for value in fine_prediction_errors]
      ),
  }
  (output / "estimator_vs_simulator_summary.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
  )

  report = [
      "# HSTU SSD IR 重计算比例模型复核",
      "",
      f"- 覆盖配置/方法组：{metrics['groups']}；新增细粒度仿真点：{fine_point_count}",
      f"- 0.2 粗网格并非最终最优的组数："
      f"{metrics['groups_where_fine_improves_coarse']}/{metrics['groups']}",
      f"- 当前估计器相对实测最优的 median/p90/max penalty："
      f"{metrics['current_estimate_penalty']['median_pct']:.2f}% / "
      f"{metrics['current_estimate_penalty']['p90_pct']:.2f}% / "
      f"{metrics['current_estimate_penalty']['max_pct']:.2f}%",
      f"- phase-aware 模型选择点相对实测最优的 median/p90/max penalty："
      f"{metrics['phase_model_selection_penalty']['median_pct']:.2f}% / "
      f"{metrics['phase_model_selection_penalty']['p90_pct']:.2f}% / "
      f"{metrics['phase_model_selection_penalty']['max_pct']:.2f}%",
      f"- phase-aware 模型在新增细点上的 E2E 绝对误差 median/p90/max："
      f"{metrics['phase_model_fine_point_e2e_absolute_error']['median_pct']:.2f}% / "
      f"{metrics['phase_model_fine_point_e2e_absolute_error']['p90_pct']:.2f}% / "
      f"{metrics['phase_model_fine_point_e2e_absolute_error']['max_pct']:.2f}%",
      "",
      "## 建模结论",
      "",
      "1. 优化目标必须是调度后的端到端事件 DAG，而不是独立的 "
      "Tmem/Tnpu 平衡点；candidate embedding、history embedding、逐层 KV preload、"
      "layer-0 与 repeated layers 的先后和重叠均需显式保留。",
      "2. 算术基线必须与 Meta HSTU trace 一致：输入投影为 8H^2、输出投影为 6H^2 "
      "FLOPs/token；attention point-wise 是 MUL+SWISH+DIV+MUL，LayerNorm 还包含两次 "
      "reduction。不能再用 8H^2 和单次 Vector pass 代表整层。",
      "3. AR 默认只减少远端 KV 行数，不应减少逻辑 QK/AV 计算；只有显式打开 "
      "AR compute reduction 时才改变 NPU attention workload。",
      "4. phase-aware 模型可作为搜索启发式，但留一配置验证未过门限，不能替换默认模型。"
      "下一步应减少自由拟合系数，直接从 simulator 导出各 phase 的 service time 和重叠边，"
      "仅校准少量芯片相关 utilization 参数。",
      "5. 离线最优比例应采用模型提议加局部实测闭环：先粗搜，再在候选两侧做 0.05/0.01 "
      "细搜；不能把 0.2 网格点称为全局时间最优。",
      "6. 当前 9 个 context 只保证 chip/model/seq/bs 各水平至少出现一次，变量彼此耦合，"
      "不足以独立识别 core 数、hidden、batch 和 sequence 的 scaling。若要让模型跨配置泛化，"
      "下一批实验应固定三个维度、逐一扫描第四个维度，而不是继续增加回归交叉项。",
      "",
      "逐配置明细见 `estimator_vs_simulator_optimum.csv`。",
  ]
  if revision_rows:
    first_revision = revision_rows[0]
    last_revision = revision_rows[-1]
    report.extend([
        "",
        "## 算术模型修正效果",
        "",
        f"- LOCO p90 E2E 绝对误差："
        f"{float(first_revision['loco_p90_abs_e2e_error_pct']):.2f}% -> "
        f"{float(last_revision['loco_p90_abs_e2e_error_pct']):.2f}%",
        f"- repeated early/late 中位校准倍率："
        f"{float(first_revision['median_repeated_early_scale']):.2f}x/"
        f"{float(first_revision['median_repeated_late_scale']):.2f}x -> "
        f"{float(last_revision['median_repeated_early_scale']):.2f}x/"
        f"{float(last_revision['median_repeated_late_scale']):.2f}x",
        "",
        "完整修订对比见 `model_revision_validation.csv`。",
    ])
  (output / "model_adjustment_report.md").write_text(
      "\n".join(report) + "\n", encoding="utf-8"
  )
  print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()
