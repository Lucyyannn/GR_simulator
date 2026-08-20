#!/usr/bin/env python3
"""Summarize Cube/Vector activity for a completed HSTU matrix run."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ["Full_Recompute", "Full_Cache", "w_AR", "w_IR", "w_both"]


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def mean(values: list[float]) -> float:
  return statistics.fmean(values)


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
  groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
  for row in rows:
    groups[tuple(row[key] for key in keys)].append(row)

  output: list[dict[str, object]] = []
  for group_key, group in groups.items():
    active = [float(row["vector_active_pct"]) for row in group]
    exposed = [float(row["vector_exposed_pct"]) for row in group]
    overlap = [float(row["vector_overlap_pct"]) for row in group]
    cube = [float(row["cube_active_pct"]) for row in group]
    result = {key: value for key, value in zip(keys, group_key)}
    result.update({
        "cases": len(group),
        "vector_active_mean_pct": mean(active),
        "vector_exposed_mean_pct": mean(exposed),
        "vector_exposed_median_pct": statistics.median(exposed),
        "vector_exposed_min_pct": min(exposed),
        "vector_exposed_max_pct": max(exposed),
        "vector_overlap_mean_pct": mean(overlap),
        "cube_active_mean_pct": mean(cube),
    })
    output.append(result)
  return output


def fmt(value: object) -> str:
  return f"{float(value):.2f}"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("result_root", type=Path)
  args = parser.parse_args()
  root = args.result_root.resolve()
  manifest = read_csv(root / "manifest.csv")

  cases: list[dict[str, object]] = []
  op_cycles: dict[tuple[str, str], int] = defaultdict(int)
  method_vector_cycles: dict[str, int] = defaultdict(int)
  missing: list[str] = []

  for manifest_row in manifest:
    chip = manifest_row["chip"]
    model = manifest_row["model"]
    seq_len = int(manifest_row["seq_len"])
    batch_size = int(manifest_row["batch_size"])
    user = manifest_row["user"]
    method = manifest_row["method"]
    case_dir = (
        root / "cases" / chip / method /
        f"HSTU-{model}_seq{seq_len}_bs{batch_size}_{user}"
    )
    activity_path = case_dir / "compute_activity.csv"
    if not activity_path.exists():
      missing.append(str(activity_path))
      continue
    activity = read_csv(activity_path)
    total = next((row for row in activity if row["scope"] == "npu_total"), None)
    if total is None:
      missing.append(f"{activity_path}: missing npu_total")
      continue

    total_core_cycles = int(total["total_core_cycles"])
    cube_cycles = int(total["cube_active_cycles"])
    vector_cycles = int(total["vector_active_cycles"])
    overlap_cycles = int(total["vector_overlap_with_cube_cycles"])
    exposed_cycles = vector_cycles - overlap_cycles
    cases.append({
        "case_id": manifest_row["case_id"],
        "chip": chip,
        "model": model,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "user": user,
        "method": method,
        "sim_time_us": float(total["sim_time_us"]),
        "total_core_cycles": total_core_cycles,
        "cube_active_cycles": cube_cycles,
        "vector_active_cycles": vector_cycles,
        "vector_exposed_cycles": exposed_cycles,
        "vector_overlap_cycles": overlap_cycles,
        "cube_active_pct": 100.0 * cube_cycles / total_core_cycles,
        "vector_active_pct": 100.0 * vector_cycles / total_core_cycles,
        "vector_exposed_pct": 100.0 * exposed_cycles / total_core_cycles,
        "vector_overlap_pct": 100.0 * overlap_cycles / total_core_cycles,
    })
    method_vector_cycles[method] += vector_cycles
    for row in activity:
      if row["scope"] == "npu_op":
        op_cycles[(method, row["op_name"])] += int(row["vector_active_cycles"])

  if missing:
    raise SystemExit("Missing activity data:\n" + "\n".join(missing))
  if len(cases) != len(manifest):
    raise SystemExit(f"Expected {len(manifest)} cases, parsed {len(cases)}")

  method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
  cases.sort(key=lambda row: (
      method_rank.get(str(row["method"]), 999), str(row["user"]),
      int(row["seq_len"]), int(row["batch_size"])))
  case_fields = list(cases[0])
  write_csv(root / "vector_ratio_cases.csv", cases, case_fields)

  by_method = aggregate(cases, ["method"])
  by_method.sort(key=lambda row: method_rank.get(str(row["method"]), 999))
  aggregate_fields = list(by_method[0])
  write_csv(root / "vector_ratio_by_method.csv", by_method, aggregate_fields)

  by_method_user = aggregate(cases, ["method", "user"])
  by_method_user.sort(key=lambda row: (
      method_rank.get(str(row["method"]), 999), str(row["user"])))
  write_csv(root / "vector_ratio_by_method_user.csv", by_method_user,
            list(by_method_user[0]))

  pivot: dict[tuple[int, int, str], dict[str, object]] = {}
  for row in cases:
    key = (int(row["seq_len"]), int(row["batch_size"]), str(row["user"]))
    output = pivot.setdefault(key, {
        "seq_len": key[0], "batch_size": key[1], "user": key[2]})
    output[f"{row['method']}_vector_exposed_pct"] = row["vector_exposed_pct"]
  pivot_rows = [pivot[key] for key in sorted(pivot)]
  pivot_fields = ["seq_len", "batch_size", "user"] + [
      f"{method}_vector_exposed_pct" for method in METHOD_ORDER]
  write_csv(root / "vector_ratio_matrix.csv", pivot_rows, pivot_fields)

  op_rows: list[dict[str, object]] = []
  for (method, op_name), cycles in op_cycles.items():
    total_cycles = method_vector_cycles[method]
    op_rows.append({
        "method": method,
        "op_name": op_name,
        "vector_active_cycles": cycles,
        "share_of_method_vector_cycles_pct": 100.0 * cycles / total_cycles,
    })
  op_rows.sort(key=lambda row: (
      method_rank.get(str(row["method"]), 999),
      -int(row["vector_active_cycles"]), str(row["op_name"])))
  write_csv(root / "vector_op_breakdown.csv", op_rows, list(op_rows[0]))

  top_cases = sorted(cases, key=lambda row: -float(row["vector_exposed_pct"]))[:10]
  lines = [
      "# HSTU-small 910C Vector 占比汇总",
      "",
      "口径：`active = vector_active_cycles / total_core_cycles`；",
      "`exposed = (vector_active_cycles - vector_overlap_with_cube_cycles) / total_core_cycles`。",
      "聚合值为各配置百分比的非加权算术平均。",
      "",
      "## 按方法",
      "",
      "| 方法 | cases | Vector active 均值 | Vector exposed 均值 | exposed 中位数 | exposed 范围 | Cube active 均值 | overlap 均值 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in by_method:
    lines.append(
        f"| {row['method']} | {row['cases']} | {fmt(row['vector_active_mean_pct'])}% | "
        f"{fmt(row['vector_exposed_mean_pct'])}% | {fmt(row['vector_exposed_median_pct'])}% | "
        f"{fmt(row['vector_exposed_min_pct'])}%–{fmt(row['vector_exposed_max_pct'])}% | "
        f"{fmt(row['cube_active_mean_pct'])}% | {fmt(row['vector_overlap_mean_pct'])}% |"
    )
  lines += [
      "",
      "## 按方法与 hot/cold",
      "",
      "| 方法 | 用户 | cases | Vector exposed 均值 | 中位数 | 范围 |",
      "|---|---|---:|---:|---:|---:|",
  ]
  for row in by_method_user:
    lines.append(
        f"| {row['method']} | {row['user']} | {row['cases']} | "
        f"{fmt(row['vector_exposed_mean_pct'])}% | {fmt(row['vector_exposed_median_pct'])}% | "
        f"{fmt(row['vector_exposed_min_pct'])}%–{fmt(row['vector_exposed_max_pct'])}% |"
    )
  lines += [
      "",
      "## exposed Vector 最高的配置",
      "",
      "| 方法 | seq | bs | 用户 | latency (us) | Vector active | overlap | Vector exposed | Cube active |",
      "|---|---:|---:|---|---:|---:|---:|---:|---:|",
  ]
  for row in top_cases:
    lines.append(
        f"| {row['method']} | {row['seq_len']} | {row['batch_size']} | {row['user']} | "
        f"{float(row['sim_time_us']):.2f} | {fmt(row['vector_active_pct'])}% | "
        f"{fmt(row['vector_overlap_pct'])}% | {fmt(row['vector_exposed_pct'])}% | "
        f"{fmt(row['cube_active_pct'])}% |"
    )
  lines += [
      "",
      "## 复现",
      "",
      "```bash",
      f"python3 scripts/analyze_hstu_vector_matrix.py {root}",
      "```",
  ]
  (root / "vector_ratio_summary.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
  main()
