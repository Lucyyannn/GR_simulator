#!/usr/bin/env python3
"""Merge coarse/exact/fine HSTU IR points and plan a local 0.01 refinement."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_hstu_ir_ratio_sweep as coarse_analysis


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("source_root", type=Path)
  parser.add_argument("fine_root", type=Path)
  parser.add_argument(
      "--extra-fine-root", type=Path, action="append", default=[],
      help="Additional completed fine-sweep roots to merge, for example stage 2.",
  )
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


def numeric(row: dict, key: str) -> float:
  return float(row[key])


def main() -> None:
  args = parse_args()
  source = args.source_root.resolve()
  fine = args.fine_root.resolve()
  source_rows = read_csv(source / "sweep_cases.csv")
  fine_roots = [fine] + [root.resolve() for root in args.extra_fine_root]
  fine_rows_by_root = [coarse_analysis.load_rows(root) for root in fine_roots]
  fine_rows = [row for rows in fine_rows_by_root for row in rows]
  if len(fine_rows) == 0:
    raise SystemExit("no successful fine-sweep results found")

  combined: list[dict] = []
  for row in source_rows:
    combined.append({
        "context_id": row["context_id"],
        "chip": row["chip"],
        "model": row["model"],
        "method": row["method"],
        "seq_len": int(row["seq_len"]),
        "batch_size": int(row["batch_size"]),
        "point_kind": row["point_kind"],
        "point_label": row["point_label"],
        "history_recompute_len": int(row["history_recompute_len"]),
        "actual_ratio": float(row["actual_ratio"]),
        "sim_time_us": float(row["sim_time_us"]),
        "model_v2_e2e_us": "",
        "model_v2_e2e_error_pct": "",
        "result_source": "coarse_and_exact",
    })
  for root_index, rows in enumerate(fine_rows_by_root, start=1):
    for row in rows:
      prediction = float(row.get("model_e2e_proxy_us", 0.0))
      simulation = float(row["sim_time_us"])
      combined.append({
        "context_id": row["context_id"],
        "chip": row["chip"],
        "model": row["model"],
        "method": row["method"],
        "seq_len": int(row["seq_len"]),
        "batch_size": int(row["batch_size"]),
        "point_kind": row["point_kind"],
        "point_label": row["point_label"],
        "history_recompute_len": int(row["history_recompute_len"]),
        "actual_ratio": float(row["actual_ratio"]),
        "sim_time_us": simulation,
        "model_v2_e2e_us": prediction,
        "model_v2_e2e_error_pct": (
            100.0 * (prediction / simulation - 1.0) if prediction else ""
        ),
          "result_source": f"fine_v2_stage{root_index}",
      })
  combined.sort(key=lambda row: (
      row["context_id"], row["method"], row["actual_ratio"],
      row["result_source"],
  ))
  write_csv(fine / "combined_observed_points.csv", combined)

  groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for row in combined:
    groups[(row["context_id"], row["method"])].append(row)

  best_rows: list[dict] = []
  stage2: list[dict] = []
  for key, rows in sorted(groups.items()):
    # Identical k values from coarse and estimator points are one observation.
    by_k: dict[int, dict] = {}
    for row in rows:
      k = int(row["history_recompute_len"])
      if k not in by_k or float(row["sim_time_us"]) < float(by_k[k]["sim_time_us"]):
        by_k[k] = row
    unique = sorted(by_k.values(), key=lambda row: row["actual_ratio"])
    best = min(unique, key=lambda row: (
        float(row["sim_time_us"]), float(row["actual_ratio"])
    ))
    # Keep the grid identity even when an estimator point has the same k and
    # therefore replaces it in the de-duplicated observation list.
    coarse = min(
        [row for row in rows if row["point_kind"] == "grid"],
        key=lambda row: float(row["sim_time_us"]),
    )
    best_rows.append({
        "context_id": key[0],
        "method": key[1],
        "chip": best["chip"],
        "model": best["model"],
        "seq_len": best["seq_len"],
        "batch_size": best["batch_size"],
        "coarse_best_ratio": coarse["actual_ratio"],
        "coarse_best_sim_time_us": coarse["sim_time_us"],
        "best_observed_ratio": best["actual_ratio"],
        "best_observed_k": best["history_recompute_len"],
        "best_observed_sim_time_us": best["sim_time_us"],
        "best_observed_source": best["result_source"],
        "improvement_over_coarse_pct": 100.0 * (
            float(best["sim_time_us"]) / float(coarse["sim_time_us"]) - 1.0
        ),
        "observed_points": len(unique),
    })

    best_index = unique.index(best)
    if best_index == 0 or best_index == len(unique) - 1:
      continue
    lower = float(unique[best_index - 1]["actual_ratio"])
    upper = float(unique[best_index + 1]["actual_ratio"])
    center = float(best["actual_ratio"])
    nearby = sorted(
        unique, key=lambda row: abs(float(row["actual_ratio"]) - center)
    )[:5]
    nearby.sort(key=lambda row: float(row["actual_ratio"]))
    xs = np.asarray([float(row["actual_ratio"]) for row in nearby])
    ys = np.asarray([float(row["sim_time_us"]) for row in nearby])
    vertex = center
    if len(nearby) >= 3:
      coefficient = np.polyfit(xs, ys, 2)
      if coefficient[0] > 0:
        proposed = -coefficient[1] / (2.0 * coefficient[0])
        if lower < proposed < upper:
          vertex = float(proposed)
    candidate_ratios = {
        round(vertex - 0.01, 2), round(vertex, 2), round(vertex + 0.01, 2)
    }
    item_count = (int(best["seq_len"]) + 1) // 2
    existing_k = {int(row["history_recompute_len"]) for row in unique}
    for ratio in sorted(candidate_ratios):
      if not (lower < ratio < upper) or ratio <= 0.0 or ratio >= 1.0:
        continue
      k = int(np.floor(ratio * item_count + 0.5))
      if k in existing_k:
        continue
      stage2.append({
          "context_id": key[0],
          "method": key[1],
          "ratio": f"{ratio:.2f}",
      })
  write_csv(fine / "best_observed_after_fine.csv", best_rows)
  write_csv(
      fine / "fine_stage2_plan.csv", stage2,
      fields=["context_id", "method", "ratio"],
  )

  fine_errors = [
      float(row["model_v2_e2e_error_pct"])
      for row in combined if str(row["result_source"]).startswith("fine_v2")
  ]
  summary = {
      "source_points": len(source_rows),
      "fine_points": len(fine_rows),
      "best_groups": len(best_rows),
      "stage2_points": len(stage2),
      "fine_model_v2_median_absolute_error_pct": float(np.median(np.abs(fine_errors))),
      "fine_model_v2_p90_absolute_error_pct": float(np.percentile(np.abs(fine_errors), 90)),
  }
  (fine / "fine_summary.json").write_text(
      json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
