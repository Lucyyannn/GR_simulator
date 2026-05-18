#!/usr/bin/env python3
"""Calibrate effective memory bandwidth with simulator mem_bench mode."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class BenchRun:
    chip: str
    pattern: str
    bench_config: Path
    output_dir: Path


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def deep_merge(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def config_peak_gbps(config_path: Path, medium: str, rw: str) -> float | None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = cfg.get("metadata", {})
    if medium == "hbm":
        value = metadata.get("derived_hbm_bandwidth_GBps") or metadata.get("target_hbm_bandwidth_GBps")
        if value is not None:
            return float(value)
        mem = cfg["hbm"]
        return float(mem["channels"]) * float(mem["req_size"]) * float(mem["freq"]) / 1000.0
    if medium == "ddr":
        value = metadata.get("derived_ddr_bandwidth_GBps") or metadata.get("target_ddr_bandwidth_GBps")
        if value is not None:
            return float(value)
        mem = cfg["ddr"]
        return float(mem["channels"]) * float(mem["req_size"]) * float(mem["freq"]) / 1000.0
    if medium == "ssd":
        key = "read" if rw == "read" else "write"
        value = (
            metadata.get(f"derived_ssd_{key}_bandwidth_GBps")
            or metadata.get(f"target_ssd_{key}_bandwidth_GBps")
            or cfg.get("ssd", {}).get(f"{key}_bandwidth_GBps")
        )
        return float(value) if value is not None else None
    return None


def write_bench_config(run: BenchRun, args: argparse.Namespace) -> None:
    run.bench_config.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "output_dir": str(run.output_dir),
        "media": args.media,
        "access_types": args.access_types,
        "sizes_bytes": args.sizes_bytes,
        "burst_counts": args.burst_counts,
        "issue_modes": args.issue_modes,
        "address_pattern": run.pattern,
        "random_seed": args.random_seed,
        "random_window_bytes": args.random_window_bytes,
    }
    run.bench_config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_runs(args: argparse.Namespace) -> list[BenchRun]:
    runs = []
    for chip in args.chips:
        for pattern in args.patterns:
            output_dir = args.result_root / "cases" / chip / pattern
            bench_config = args.result_root / "bench_configs" / f"{chip}_{pattern}.json"
            run = BenchRun(chip=chip, pattern=pattern, bench_config=bench_config, output_dir=output_dir)
            write_bench_config(run, args)
            runs.append(run)
    return runs


def run_command(run: BenchRun, args: argparse.Namespace) -> list[str]:
    return [
        args.simulator_bin,
        "--config",
        f"configs/{run.chip}.json",
        "--mode",
        "mem_bench",
        "--bench_config",
        str(run.bench_config),
        "--bench_output_dir",
        str(run.output_dir),
        "--log_level",
        args.log_level,
    ]


def run_benchmarks(runs: list[BenchRun], args: argparse.Namespace) -> None:
    logs_dir = args.result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(runs)
    active: list[tuple[subprocess.Popen, object, BenchRun]] = []
    failures: list[BenchRun] = []

    while pending or active:
        while pending and len(active) < args.max_concurrent:
            run = pending.pop(0)
            status_path = logs_dir / f"{run.chip}__{run.pattern}.status"
            if status_path.exists() and status_path.read_text(encoding="utf-8").strip() == "0" and (run.output_dir / "summary.csv").exists():
                continue
            log_path = logs_dir / f"{run.chip}__{run.pattern}.log"
            log_file = log_path.open("w", encoding="utf-8")
            cmd = run_command(run, args)
            print(f"[mem-cal] {quote_cmd(cmd)}")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((proc, log_file, run))

        still_active: list[tuple[subprocess.Popen, object, BenchRun]] = []
        for proc, log_file, run in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, log_file, run))
                continue
            log_file.close()
            status_path = logs_dir / f"{run.chip}__{run.pattern}.status"
            status_path.write_text(str(code) + "\n", encoding="utf-8")
            if code != 0:
                failures.append(run)
        active = still_active
        if pending or active:
            time.sleep(args.poll_interval)

    if failures:
        failed = ", ".join(f"{run.chip}/{run.pattern}" for run in failures)
        raise RuntimeError(f"{len(failures)} mem_bench calibration runs failed: {failed}")


def pick_rows(rows: list[dict], medium: str, rw: str) -> list[dict]:
    return [row for row in rows if row.get("medium") == medium and row.get("rw") == rw]


def summarize_run(run: BenchRun, args: argparse.Namespace) -> dict:
    summary_path = run.output_dir / "summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Missing mem_bench summary: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chip_result: dict = {}
    config_path = REPO_ROOT / "configs" / f"{run.chip}.json"
    for medium in args.media:
        medium_result = chip_result.setdefault(medium, {})
        pattern_result = medium_result.setdefault(run.pattern, {})
        for rw in args.access_types:
            selected = pick_rows(rows, medium, rw)
            by_size: dict[str, dict] = {}
            best_gbps = 0.0
            best_row: dict | None = None
            for row in selected:
                size = row["access_size_bytes"]
                bandwidth = float(row["bandwidth_GBps"])
                current = by_size.get(size)
                if current is None or bandwidth > float(current["effective_GBps"]):
                    by_size[size] = {
                        "effective_GBps": bandwidth,
                        "burst_count": int(row["burst_count"]),
                        "issue_mode": row["issue_mode"],
                        "total_time_ns": float(row["total_time_ns"]),
                    }
                if bandwidth > best_gbps:
                    best_gbps = bandwidth
                    best_row = row
            peak_gbps = config_peak_gbps(config_path, medium, rw)
            pattern_result[rw] = {
                "effective_GBps": best_gbps,
                "peak_GBps": peak_gbps,
                "utilization": (best_gbps / peak_gbps) if peak_gbps else None,
                "best_size_bytes": int(best_row["access_size_bytes"]) if best_row else None,
                "best_burst_count": int(best_row["burst_count"]) if best_row else None,
                "by_size": by_size,
            }
    return chip_result


def write_outputs(runs: list[BenchRun], args: argparse.Namespace) -> None:
    contexts: dict = {}
    for run in runs:
        chip_context = contexts.setdefault(run.chip, {})
        run_context = summarize_run(run, args)
        for medium, medium_value in run_context.items():
            chip_context.setdefault(medium, {}).update(medium_value)

    suggestions = {
        "memory_bandwidth_calibration": {
            "version": 1,
            "source": "scripts/calibrate_memory_bandwidth.py",
            "patterns": args.patterns,
            "media": args.media,
            "access_types": args.access_types,
            "contexts": contexts,
        }
    }

    suggestions_path = args.result_root / "memory_bandwidth_calibration.json"
    suggestions_path.write_text(json.dumps(suggestions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.merged_calibration_output:
        base = json.loads(args.calibration.read_text(encoding="utf-8"))
        # The old fitted DDR/SSD bandwidth model was HSTU-specific. The new
        # calibration should drive memory bandwidth directly by medium/pattern.
        base.pop("effective_bandwidth_bps", None)
        base.pop("preload_bandwidth_model", None)
        merged = deep_merge(base, suggestions)
        args.merged_calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.merged_calibration_output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Memory Bandwidth Calibration Summary",
        "",
        f"- Result root: `{args.result_root}`",
        f"- Calibration JSON: `{suggestions_path}`",
    ]
    if args.merged_calibration_output:
        lines.append(f"- Merged calibration: `{args.merged_calibration_output}`")
    lines.extend(
        [
            "",
            "| Chip | Medium | Pattern | RW | Effective GB/s | Peak GB/s | Utilization | Best size B | Best burst |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for chip in sorted(contexts):
        for medium in args.media:
            for pattern in args.patterns:
                for rw in args.access_types:
                    entry = contexts.get(chip, {}).get(medium, {}).get(pattern, {}).get(rw, {})
                    util = entry.get("utilization")
                    lines.append(
                        f"| {chip} | {medium} | {pattern} | {rw} | "
                        f"{entry.get('effective_GBps', 0.0):.4f} | "
                        f"{entry.get('peak_GBps') or 0.0:.4f} | "
                        f"{util if util is not None else 0.0:.4f} | "
                        f"{entry.get('best_size_bytes') or ''} | "
                        f"{entry.get('best_burst_count') or ''} |"
                    )
    (args.result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Calibration: {suggestions_path}")
    if args.merged_calibration_output:
        print(f"Merged calibration: {args.merged_calibration_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path("results") / f"memory_bandwidth_calibration_{time.strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--result-root", type=Path, default=default_root)
    parser.add_argument("--calibration", type=Path, default=Path("scripts/recompute_ratio_calibration.json"))
    parser.add_argument("--merged-calibration-output", type=Path, default=None)
    parser.add_argument("--chips", type=lambda v: parse_csv_list(v, str), default=["910A", "910B", "910C"])
    parser.add_argument("--patterns", type=lambda v: parse_csv_list(v, str), default=["contiguous", "random_512b_index"])
    parser.add_argument("--media", type=lambda v: parse_csv_list(v, str), default=["hbm", "ddr", "ssd"])
    parser.add_argument("--access-types", type=lambda v: parse_csv_list(v, str), default=["read"])
    parser.add_argument("--sizes-bytes", type=lambda v: parse_csv_list(v, int), default=[512, 1024, 2048])
    parser.add_argument("--burst-counts", type=lambda v: parse_csv_list(v, int), default=[1, 2, 4, 8])
    parser.add_argument("--issue-modes", type=lambda v: parse_csv_list(v, str), default=["back_to_back"])
    parser.add_argument("--random-seed", type=int, default=20260422)
    parser.add_argument("--random-window-bytes", type=int, default=67108864)
    parser.add_argument("--simulator-bin", default="./build/bin/Simulator")
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.calibration.exists():
        raise SystemExit(f"Calibration file not found: {args.calibration}")
    for chip in args.chips:
        config_path = REPO_ROOT / "configs" / f"{chip}.json"
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
    runs = build_runs(args)
    if args.dry_run:
        for run in runs:
            print(quote_cmd(run_command(run, args)))
        print(f"planned_runs={len(runs)}")
        return

    args.result_root.mkdir(parents=True, exist_ok=True)
    run_benchmarks(runs, args)
    write_outputs(runs, args)


if __name__ == "__main__":
    main()
