#!/usr/bin/env python3
"""Estimate activity-weighted AR/w_both QPS with a finite DDR KV cache.

Full Recompute (RE) and Full KV Cache (CA) are always read from their SSD
(``cold``) results.  AR and w_both place the most active users in DDR until
the capacity is full, and use the measured hot/cold endpoints to estimate the
interaction-weighted latency:

    T_avg = p_hot * T_hot + (1 - p_hot) * T_cold
    QPS   = batch_size * 1e6 / T_avg_us

The default inputs are the archived 2026-09-03 matrices.  Result roots are
command-line inputs so the same analysis can be applied to the corrected AR
matrix without changing this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ROOT = Path("results/hstu_calibration_matrix_20260901/base_methods")
DEFAULT_BOTH_ROOT = Path("results/hstu_w_both_matrix_20260902")
DEFAULT_ACTIVITY = Path("configs/kuairand_1k_user_activity_distribution.csv")
DEFAULT_OUTPUT = Path("results/analysis/activity_weighted_qps_0903")
DEFAULT_DDR_GIB = 128.0
DEFAULT_USER_SCALE = 100
DEFAULT_BYTES_PER_ELEMENT = 2.0
METHODS = ("w_AR", "w_both")
BASELINES = ("Full_Recompute", "Full_Cache")


@dataclass(frozen=True)
class CaseResult:
    chip: str
    model: str
    seq_len: int
    batch_size: int
    user: str
    method: str
    latency_us: float
    layers: int
    hidden: int
    recompute_items: int
    kv_reuse_ratio: float
    ar_attention_compute: str

    @property
    def workload(self) -> tuple[str, str, int, int]:
        return self.chip, self.model, self.seq_len, self.batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root", action="append", type=Path, dest="base_roots",
        help="Matrix containing RE, CA, and AR; repeat to merge matrices.",
    )
    parser.add_argument(
        "--both-root", action="append", type=Path, dest="both_roots",
        help="Matrix containing w_both; repeat to merge matrices.",
    )
    parser.add_argument("--activity-csv", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument(
        "--ddr-gib", type=float, default=DEFAULT_DDR_GIB,
        help="Capacity available exclusively to persistent KV, in GiB.",
    )
    parser.add_argument(
        "--user-scale", type=int, default=DEFAULT_USER_SCALE,
        help="Number of equal-activity users represented by each CSV row.",
    )
    parser.add_argument(
        "--bytes-per-element", type=float, default=DEFAULT_BYTES_PER_ELEMENT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def command_value(command: list[str], option: str, default: str | None = None) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        if default is None:
            raise ValueError(f"status command is missing {option}")
        return default


def ar_semantics(status: dict, command: list[str]) -> str:
    explicit = status.get("ar_reduce_attention_compute")
    if explicit is not None:
        return "enabled" if bool(explicit) else "disabled"
    if "--enable-ar-reduce-attention-compute" in command:
        return "enabled"
    if "--disable-ar-reduce-attention-compute" in command:
        return "disabled"
    return "unknown"


def npu_latency_us(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("component") == "NPU" and row.get("scope") == "overall":
                return float(row["sim_time_us"])
    return None


def load_successful(roots: Iterable[Path], allowed: set[str]) -> dict[tuple, CaseResult]:
    """Load only successful status records with a complete NPU summary.

    Roots are applied in command-line order; a later root replaces the same
    logical case.  This makes extending the old analysis to refreshed results
    explicit and deterministic.
    """

    values: dict[tuple, CaseResult] = {}
    for raw_root in roots:
        root = resolve(raw_root)
        for status_path in sorted((root / "logs").glob("*.status.json")):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            method = str(status.get("method", ""))
            if status.get("returncode") != 0 or method not in allowed:
                continue
            command = [str(value) for value in status.get("command", [])]
            try:
                chip = str(status["chip"])
                model = str(status["model"])
                seq_len = int(status["seq_len"])
                batch_size = int(status["batch_size"])
                user = str(status["user"])
                layers = int(command_value(command, "--layers"))
                hidden = int(command_value(command, "--hidden"))
                reuse = float(command_value(command, "--kv-reuse-ratio", "0"))
                recompute = int(
                    status.get(
                        "history_recompute_len",
                        command_value(command, "--history-recompute-len", "0"),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            case_name = f"HSTU-{model}_seq{seq_len}_bs{batch_size}_{user}"
            latency = npu_latency_us(
                root / "cases" / chip / method / case_name / "hardware_summary.csv"
            )
            if latency is None or latency <= 0.0:
                continue
            key = (chip, model, seq_len, batch_size, user, method)
            values[key] = CaseResult(
                chip=chip, model=model, seq_len=seq_len, batch_size=batch_size,
                user=user, method=method, latency_us=latency, layers=layers,
                hidden=hidden, recompute_items=recompute,
                kv_reuse_ratio=reuse,
                ar_attention_compute=ar_semantics(status, command),
            )
    return values


def load_activity(path: Path) -> list[int]:
    interactions = []
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            interactions.append(int(row["number_of_interactions"]))
    if not interactions or any(value < 0 for value in interactions):
        raise ValueError("activity CSV must contain non-negative interactions")
    return sorted(interactions, reverse=True)


def retained_action_rows(seq_len: int, total_reuse_ratio: float) -> int:
    """Match the simulator's Action-reuse row-count calculation exactly."""

    actions = seq_len // 2
    if actions <= 0:
        return 0
    total_reuse_ratio = min(1.0, max(0.0, total_reuse_ratio))
    action_reuse_ratio = min(1.0, total_reuse_ratio * seq_len / actions)
    if action_reuse_ratio <= 0.0:
        return actions
    return max(1, int(round(actions * (1.0 - action_reuse_ratio))))


def persistent_kv_bytes(case: CaseResult, bytes_per_element: float) -> tuple[int, float]:
    """Return cached rows and 2*L*H*rows*s persistent K/V bytes per user."""

    item_rows = (case.seq_len + 1) // 2
    if not 0 <= case.recompute_items <= item_rows:
        raise ValueError(
            f"invalid item recompute length {case.recompute_items} for {case.seq_len}"
        )
    cached_rows = (
        item_rows - case.recompute_items
        + retained_action_rows(case.seq_len, case.kv_reuse_ratio)
    )
    kv_bytes = 2.0 * case.layers * case.hidden * cached_rows * bytes_per_element
    return cached_rows, kv_bytes


def ddr_coverage(
    activity: list[int], user_scale: int, capacity_bytes: float, user_bytes: float,
) -> tuple[int, float, float]:
    """Return resident users, user fraction, and interaction fraction."""

    if user_scale <= 0:
        raise ValueError("user-scale must be positive")
    total_users = len(activity) * user_scale
    resident_users = total_users if user_bytes == 0 else min(
        total_users, int(capacity_bytes // user_bytes)
    )
    remaining = resident_users
    resident_interactions = 0
    for interactions in activity:
        count = min(user_scale, remaining)
        resident_interactions += count * interactions
        remaining -= count
        if remaining == 0:
            break
    total_interactions = user_scale * sum(activity)
    interaction_fraction = (
        resident_interactions / total_interactions if total_interactions else 0.0
    )
    return resident_users, resident_users / total_users, interaction_fraction


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze(
    values: dict[tuple, CaseResult], activity: list[int], capacity_bytes: float,
    user_scale: int, bytes_per_element: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    missing: list[dict] = []
    workloads = sorted({case.workload for case in values.values()})
    for chip, model, seq_len, batch_size in workloads:
        for method in METHODS:
            keys = {
                "re": (chip, model, seq_len, batch_size, "cold", "Full_Recompute"),
                "ca": (chip, model, seq_len, batch_size, "cold", "Full_Cache"),
                "hot": (chip, model, seq_len, batch_size, "hot", method),
                "cold": (chip, model, seq_len, batch_size, "cold", method),
            }
            absent = [name for name, key in keys.items() if key not in values]
            if absent:
                missing.append({
                    "chip": chip, "model": model, "seq_len": seq_len,
                    "batch_size": batch_size, "method": method,
                    "missing": ";".join(absent),
                })
                continue
            re_case, ca_case = values[keys["re"]], values[keys["ca"]]
            hot_case, cold_case = values[keys["hot"]], values[keys["cold"]]
            cached_rows, user_bytes = persistent_kv_bytes(
                hot_case, bytes_per_element
            )
            resident, user_fraction, hot_fraction = ddr_coverage(
                activity, user_scale, capacity_bytes, user_bytes
            )
            average_us = (
                hot_fraction * hot_case.latency_us
                + (1.0 - hot_fraction) * cold_case.latency_us
            )
            qps = batch_size * 1e6 / average_us
            re_qps = batch_size * 1e6 / re_case.latency_us
            ca_qps = batch_size * 1e6 / ca_case.latency_us
            rows.append({
                "chip": chip,
                "model": model,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "method": "AR" if method == "w_AR" else "both",
                "ar_attention_compute": hot_case.ar_attention_compute,
                "layers": hot_case.layers,
                "hidden": hot_case.hidden,
                "recompute_item_rows_for_ddr_user": hot_case.recompute_items,
                "kv_reuse_ratio": hot_case.kv_reuse_ratio,
                "persistent_kv_rows_per_user": cached_rows,
                "persistent_kv_mib_per_user": user_bytes / 2**20,
                "ddr_resident_users": resident,
                "ddr_user_fraction": user_fraction,
                "ddr_interaction_fraction": hot_fraction,
                "hot_latency_us": hot_case.latency_us,
                "cold_latency_us": cold_case.latency_us,
                "weighted_latency_us": average_us,
                "weighted_qps": qps,
                "re_ssd_latency_us": re_case.latency_us,
                "re_ssd_qps": re_qps,
                "ca_ssd_latency_us": ca_case.latency_us,
                "ca_ssd_qps": ca_qps,
                "speedup_vs_re_ssd": qps / re_qps,
                "speedup_vs_ca_ssd": qps / ca_qps,
            })
    return rows, missing


def summarize(rows: list[dict], common_only: bool = False) -> list[dict]:
    if common_only:
        methods_by_workload: dict[tuple, set[str]] = defaultdict(set)
        for row in rows:
            workload = (
                row["chip"], row["model"], row["seq_len"], row["batch_size"]
            )
            methods_by_workload[workload].add(str(row["method"]))
        common = {
            workload for workload, methods in methods_by_workload.items()
            if methods == {"AR", "both"}
        }
        rows = [
            row for row in rows
            if (row["chip"], row["model"], row["seq_len"], row["batch_size"])
            in common
        ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["chip"]), str(row["method"]))].append(row)
    summary = []
    for (chip, method), selected in sorted(groups.items()):
        re_speedups = [float(row["speedup_vs_re_ssd"]) for row in selected]
        ca_speedups = [float(row["speedup_vs_ca_ssd"]) for row in selected]
        summary.append({
            "chip": chip,
            "method": method,
            "case_count": len(selected),
            "geomean_speedup_vs_re_ssd": geometric_mean(re_speedups),
            "min_speedup_vs_re_ssd": min(re_speedups),
            "geomean_speedup_vs_ca_ssd": geometric_mean(ca_speedups),
            "min_speedup_vs_ca_ssd": min(ca_speedups),
        })
    return summary


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.ddr_gib <= 0 or args.bytes_per_element <= 0:
        raise SystemExit("--ddr-gib and --bytes-per-element must be positive")
    base_roots = args.base_roots or [DEFAULT_BASE_ROOT]
    both_roots = args.both_roots or [DEFAULT_BOTH_ROOT]
    values = load_successful(base_roots, set(BASELINES) | {"w_AR"})
    values.update(load_successful(both_roots, {"w_both"}))
    activity = load_activity(args.activity_csv)
    rows, missing = analyze(
        values=values,
        activity=activity,
        capacity_bytes=args.ddr_gib * 2**30,
        user_scale=args.user_scale,
        bytes_per_element=args.bytes_per_element,
    )
    if not rows:
        raise SystemExit("no complete matched cases found")
    output = resolve(args.output_root)
    write_csv(output / "qps_speedup_by_case.csv", rows)
    summary = summarize(rows)
    write_csv(output / "qps_speedup_by_chip.csv", summary)
    common_summary = summarize(rows, common_only=True)
    write_csv(output / "qps_speedup_by_chip_common_cases.csv", common_summary)
    write_csv(
        output / "missing_cases.csv", missing,
        ["chip", "model", "seq_len", "batch_size", "method", "missing"],
    )
    metadata = {
        "base_roots": [str(resolve(path)) for path in base_roots],
        "both_roots": [str(resolve(path)) for path in both_roots],
        "activity_csv": str(resolve(args.activity_csv)),
        "source_activity_users": len(activity),
        "user_scale": args.user_scale,
        "scaled_users": len(activity) * args.user_scale,
        "ddr_capacity_gib": args.ddr_gib,
        "bytes_per_element": args.bytes_per_element,
        "qps_formula": "batch_size * 1e6 / interaction_weighted_latency_us",
        "baseline_policy": "Full_Recompute and Full_Cache always use cold/SSD",
        "mixed_policy": "AR and both use activity-weighted hot/DDR and cold/SSD endpoints",
        "complete_rows": len(rows),
        "missing_rows": len(missing),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} case rows and {len(summary)} summaries to {output}")
    for row in summary:
        print(
            f"{row['chip']:5s} {row['method']:4s} n={row['case_count']:2d} "
            f"geo vs RE={row['geomean_speedup_vs_re_ssd']:.4f}x, "
            f"vs CA={row['geomean_speedup_vs_ca_ssd']:.4f}x"
        )
    if missing:
        print(f"warning: {len(missing)} incomplete method/workload rows; see missing_cases.csv")


if __name__ == "__main__":
    main()
