#!/usr/bin/env python3
"""Summarize HSTU scalability experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEME_ORDER = ["Full_Cache", "Full_Recompute", "w_AR", "w_IR", "w_both"]
SOURCE_ORDER = ["DRAM", "SSD"]
MODEL_ORDER = ["small", "middle", "large"]


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2) -> str:
    number = fnum(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def avg(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_models(path: Path) -> dict:
    return read_json(path).get("metadata", {})


def row_by(rows: list[dict], component: str, scope: str) -> dict:
    for row in rows:
        if row.get("component") == component and row.get("scope") == scope:
            return row
    return {}


def parse_hardware_summary(path: Path) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "sim_time_us": None,
        "npu_util_overall": None,
        "npu_core_util_avg": None,
        "npu_core_util_min": None,
        "npu_core_util_max": None,
        "hbm_util_avg": None,
        "hbm_util_min": None,
        "hbm_util_max": None,
        "hbm_bw_GBps_total": None,
        "hbm_bw_util_avg": None,
        "hbm_bw_util_min": None,
        "hbm_bw_util_max": None,
        "ddr_util": None,
        "ddr_bw_GBps": None,
        "ddr_bw_util": None,
        "ssd_util": None,
        "ssd_read_util": None,
        "ssd_write_util": None,
        "ssd_read_bw_GBps": None,
        "ssd_write_bw_GBps": None,
    }
    if not path.exists():
        return empty

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    npu_overall = row_by(rows, "NPU", "overall")
    core_utils = [
        fnum(r.get("utilization_percent"))
        for r in rows
        if r.get("component") == "NPU" and r.get("scope") == "core"
    ]
    hbm_overall = [
        r
        for r in rows
        if r.get("scope") == "overall"
        and (r.get("component") == "HBM" or r.get("component", "").startswith("HBM.npu"))
    ]
    hbm_utils = [fnum(r.get("utilization_percent")) for r in hbm_overall]
    hbm_bw_utils = [fnum(r.get("bandwidth_utilization_percent")) for r in hbm_overall]
    hbm_bw = [fnum(r.get("bandwidth_GBps")) for r in hbm_overall]
    ddr = row_by(rows, "DDR", "overall")
    ssd = row_by(rows, "SSD", "overall")
    ssd_read = row_by(rows, "SSD", "read")
    ssd_write = row_by(rows, "SSD", "write")

    empty.update(
        {
            "sim_time_us": fnum(npu_overall.get("sim_time_us"))
            or fnum(ddr.get("sim_time_us"))
            or fnum(ssd.get("sim_time_us")),
            "npu_util_overall": fnum(npu_overall.get("utilization_percent")),
            "npu_core_util_avg": avg(core_utils),
            "npu_core_util_min": min([v for v in core_utils if v is not None], default=None),
            "npu_core_util_max": max([v for v in core_utils if v is not None], default=None),
            "hbm_util_avg": avg(hbm_utils),
            "hbm_util_min": min([v for v in hbm_utils if v is not None], default=None),
            "hbm_util_max": max([v for v in hbm_utils if v is not None], default=None),
            "hbm_bw_GBps_total": sum(v for v in hbm_bw if v is not None) if hbm_bw else None,
            "hbm_bw_util_avg": avg(hbm_bw_utils),
            "hbm_bw_util_min": min([v for v in hbm_bw_utils if v is not None], default=None),
            "hbm_bw_util_max": max([v for v in hbm_bw_utils if v is not None], default=None),
            "ddr_util": fnum(ddr.get("utilization_percent")),
            "ddr_bw_GBps": fnum(ddr.get("bandwidth_GBps")),
            "ddr_bw_util": fnum(ddr.get("bandwidth_utilization_percent")),
            "ssd_util": fnum(ssd.get("utilization_percent")),
            "ssd_read_util": fnum(ssd_read.get("read_bandwidth_utilization_percent"))
            or fnum(ssd_read.get("utilization_percent")),
            "ssd_write_util": fnum(ssd_write.get("write_bandwidth_utilization_percent"))
            or fnum(ssd_write.get("utilization_percent")),
            "ssd_read_bw_GBps": fnum(ssd_read.get("read_bandwidth_GBps"))
            or fnum(ssd_read.get("bandwidth_GBps")),
            "ssd_write_bw_GBps": fnum(ssd_write.get("write_bandwidth_GBps"))
            or fnum(ssd_write.get("bandwidth_GBps")),
        }
    )
    return empty


def parse_overlap_summary(path: Path) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "preload_overlap_gap_avg_us": None,
        "preload_overlap_gap_min_us": None,
        "preload_overlap_gap_max_us": None,
        "preload_wait_avg_us": None,
        "preload_wait_min_us": None,
        "preload_wait_max_us": None,
    }
    if not path.exists():
        return empty

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gaps: list[float] = []
    waits: list[float] = []
    layer_ids = sorted({int(r["layer_id"]) for r in rows if r.get("layer_id", "").isdigit()})
    for layer_id in layer_ids:
        if layer_id == 0:
            continue
        # For IR overlap, only the repeated KV-cache preload participates in
        # the steady-state pipeline. The layer-0 bootstrap stage includes
        # candidate/history embedding, and "stage" rows are aggregate envelope
        # events for the subtask rows, so they must not be mixed into this gap.
        preload = [
            r
            for r in rows
            if r.get("layer_id") == str(layer_id)
            and r.get("pipe") == "preload"
            and r.get("phase") == "kvcache"
        ]
        if not preload:
            preload = [
                r
                for r in rows
                if r.get("layer_id") == str(layer_id)
                and r.get("pipe") == "preload"
                and r.get("phase") == "stage"
            ]
        recompute = [
            r
            for r in rows
            if r.get("layer_id") == str(layer_id)
            and r.get("pipe") == "compute"
            and r.get("phase") == "op"
            and r.get("name") == "hstu::attention.recompute_early"
        ]
        cached_late = [
            r
            for r in rows
            if r.get("layer_id") == str(layer_id)
            and r.get("pipe") == "compute"
            and r.get("phase") == "op"
            and r.get("name") == "hstu::attention.cached_late"
        ]
        if not preload:
            continue
        preload_end = max(float(r["end_us"]) for r in preload)
        if recompute and cached_late:
            recompute_end = max(float(r["end_us"]) for r in recompute)
            cached_late_start = min(float(r["start_us"]) for r in cached_late)
        else:
            attention = [
                r
                for r in rows
                if r.get("layer_id") == str(layer_id)
                and r.get("pipe") == "compute"
                and r.get("phase") == "op"
                and r.get("name") == "hstu::attention"
            ]
            if not attention:
                continue
            attention_start = min(float(r["start_us"]) for r in attention)
            pre_attention_ends = [
                float(r["end_us"])
                for r in rows
                if r.get("layer_id") == str(layer_id)
                and r.get("pipe") == "compute"
                and r.get("phase") == "op"
                and r.get("name") != "hstu::attention"
                and float(r["end_us"]) <= attention_start + 1e-6
            ]
            recompute_end = max(pre_attention_ends) if pre_attention_ends else attention_start
            cached_late_start = attention_start
        gaps.append(preload_end - recompute_end)
        waits.append(max(0.0, cached_late_start - recompute_end))

    if not gaps:
        return empty
    empty.update(
        {
            "preload_overlap_gap_avg_us": avg(gaps),
            "preload_overlap_gap_min_us": min(gaps),
            "preload_overlap_gap_max_us": max(gaps),
            "preload_wait_avg_us": avg(waits),
            "preload_wait_min_us": min(waits),
            "preload_wait_max_us": max(waits),
        }
    )
    return empty


def sort_key(row: dict) -> tuple:
    return (
        int(row["npu_count"]),
        SOURCE_ORDER.index(row["source"]) if row["source"] in SOURCE_ORDER else 99,
        row["chip"],
        SCHEME_ORDER.index(row["scheme"]) if row["scheme"] in SCHEME_ORDER else 99,
    )


def collect_rows(cases_root: Path, logs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for case_dir in sorted(cases_root.glob("NPU*/*/*/*")):
        if not case_dir.is_dir():
            continue
        npu_label, source_label, chip, scheme = case_dir.parts[-4:]
        npu_count = npu_label[3:] if npu_label.startswith("NPU") else npu_label
        status_label = f"{npu_label}__{source_label}__{chip}__{scheme}"
        status_path = logs_root / f"{status_label}.status"
        status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "missing"
        metadata = parse_models(case_dir / "models.json")
        hw = parse_hardware_summary(case_dir / "hardware_summary.csv")
        overlap = parse_overlap_summary(case_dir / "layer_breakdown.csv")
        row = {
            "npu_count": npu_count,
            "source": source_label,
            "chip": chip,
            "scheme": scheme,
            "status": status,
            "history_recompute_len": metadata.get("history_recompute_len", ""),
            "kv_reuse_enabled": metadata.get("kv_reuse_enabled", ""),
            "sim_time_us": hw["sim_time_us"],
            "speedup_vs_Full_Cache": None,
            "speedup_vs_Full_Recompute": None,
            "npu_util_overall": hw["npu_util_overall"],
            "npu_core_util_avg": hw["npu_core_util_avg"],
            "npu_core_util_min": hw["npu_core_util_min"],
            "npu_core_util_max": hw["npu_core_util_max"],
            "hbm_util_avg": hw["hbm_util_avg"],
            "hbm_util_min": hw["hbm_util_min"],
            "hbm_util_max": hw["hbm_util_max"],
            "hbm_bw_GBps_total": hw["hbm_bw_GBps_total"],
            "hbm_bw_util_avg": hw["hbm_bw_util_avg"],
            "hbm_bw_util_min": hw["hbm_bw_util_min"],
            "hbm_bw_util_max": hw["hbm_bw_util_max"],
            "ddr_util": hw["ddr_util"],
            "ddr_bw_GBps": hw["ddr_bw_GBps"],
            "ddr_bw_util": hw["ddr_bw_util"],
            "ssd_util": hw["ssd_util"],
            "ssd_read_util": hw["ssd_read_util"],
            "ssd_write_util": hw["ssd_write_util"],
            "ssd_read_bw_GBps": hw["ssd_read_bw_GBps"],
            "ssd_write_bw_GBps": hw["ssd_write_bw_GBps"],
            "preload_overlap_gap_avg_us": overlap["preload_overlap_gap_avg_us"],
            "preload_overlap_gap_min_us": overlap["preload_overlap_gap_min_us"],
            "preload_overlap_gap_max_us": overlap["preload_overlap_gap_max_us"],
            "preload_wait_avg_us": overlap["preload_wait_avg_us"],
            "preload_wait_min_us": overlap["preload_wait_min_us"],
            "preload_wait_max_us": overlap["preload_wait_max_us"],
            "case_dir": str(case_dir),
        }
        rows.append(row)
    rows.sort(key=sort_key)
    add_speedups(rows)
    return rows


def add_speedups(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        groups.setdefault((row["npu_count"], row["source"], row["chip"]), {})[row["scheme"]] = row

    for cases in groups.values():
        cache_time = fnum(cases.get("Full_Cache", {}).get("sim_time_us"))
        recompute_time = fnum(cases.get("Full_Recompute", {}).get("sim_time_us"))
        for row in cases.values():
            case_time = fnum(row.get("sim_time_us"))
            if case_time and cache_time:
                row["speedup_vs_Full_Cache"] = cache_time / case_time
            if case_time and recompute_time:
                row["speedup_vs_Full_Recompute"] = recompute_time / case_time


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_recompute_choices(root: Path, rows: list[dict]) -> None:
    choices = [
        {
            "npu_count": row["npu_count"],
            "source": row["source"],
            "chip": row["chip"],
            "scheme": row["scheme"],
            "history_recompute_len": row["history_recompute_len"],
            "kv_reuse_enabled": row["kv_reuse_enabled"],
            "sim_time_us": row["sim_time_us"],
            "case_dir": row["case_dir"],
        }
        for row in rows
        if row["scheme"] in {"w_IR", "w_both"}
    ]
    fields = [
        "npu_count",
        "source",
        "chip",
        "scheme",
        "history_recompute_len",
        "kv_reuse_enabled",
        "sim_time_us",
        "case_dir",
    ]
    write_csv(root / "recompute_choices.csv", choices, fields)
    (root / "recompute_choices.json").write_text(
        json.dumps(choices, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def speedup_row(row: dict) -> str:
    return (
        f"| {row['npu_count']} | {row['source']} | {row['chip']} | {row['scheme']} | {row['status']} | "
        f"{row['history_recompute_len']} | {fmt(row['sim_time_us'])} | "
        f"{fmt(row['speedup_vs_Full_Cache'], 4)} | {fmt(row['speedup_vs_Full_Recompute'], 4)} |"
    )


def util_row(row: dict) -> str:
    core = "/".join(fmt(row[key]) for key in ("npu_core_util_avg", "npu_core_util_min", "npu_core_util_max"))
    hbm = "/".join(fmt(row[key]) for key in ("hbm_util_avg", "hbm_util_min", "hbm_util_max"))
    hbm_bw = "/".join(fmt(row[key]) for key in ("hbm_bw_util_avg", "hbm_bw_util_min", "hbm_bw_util_max"))
    ddr = f"{fmt(row['ddr_util'])}/{fmt(row['ddr_bw_util'])}/{fmt(row['ddr_bw_GBps'])}"
    ssd = f"{fmt(row['ssd_read_util'])}/{fmt(row['ssd_write_util'])}/{fmt(row['ssd_read_bw_GBps'])}"
    gap = "/".join(fmt(row[key]) for key in ("preload_overlap_gap_avg_us", "preload_overlap_gap_min_us", "preload_overlap_gap_max_us"))
    wait = "/".join(fmt(row[key]) for key in ("preload_wait_avg_us", "preload_wait_min_us", "preload_wait_max_us"))
    return (
        f"| {row['npu_count']} | {row['source']} | {row['chip']} | {row['scheme']} | "
        f"{fmt(row['npu_util_overall'])} | {core} | {hbm} | {hbm_bw} | "
        f"{fmt(row['hbm_bw_GBps_total'])} | {ddr} | {ssd} | {gap} | {wait} |"
    )


def write_summary(root: Path, rows: list[dict], calibration: Path, metadata: dict) -> None:
    completed = sum(1 for row in rows if row["status"] == "0")
    lines = [
        "# HSTU Scalability Summary",
        "",
        f"- Result root: `{root}`",
        f"- Calibration used: `{calibration}`",
        f"- Calibration cache: `{metadata.get('calibration_root', '')}`",
        f"- Completed cases: {completed}/{len(rows)}",
        f"- Full CSV: `{root / 'scalability_summary.csv'}`",
        f"- Recompute choices: `{root / 'recompute_choices.csv'}`",
        "",
        "## Speedups",
        "",
        "| NPU | Source | Chip | Scheme | Status | k | Time us | Speedup vs Full_Cache | Speedup vs Full_Recompute |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(speedup_row(row) for row in rows)
    lines.extend(
        [
            "",
            "## Hardware Utilization",
            "",
            "| NPU | Source | Chip | Scheme | NPU util % | Core avg/min/max % | HBM util avg/min/max % | HBM BW util avg/min/max % | HBM BW total GB/s | DDR util/BW util/GBps | SSD read/write util/read GBps | Preload gap avg/min/max us | Wait avg/min/max us |",
            "|---:|---|---|---|---:|---|---|---|---:|---|---|---|---|",
        ]
    )
    lines.extend(util_row(row) for row in rows)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(rows: list[dict], source: str, scheme: str, key: str) -> float | None:
    return avg([fnum(row.get(key)) for row in rows if row["source"] == source and row["scheme"] == scheme])


def write_analysis(root: Path, rows: list[dict]) -> None:
    lines = [
        "# HSTU Scalability Analysis",
        "",
        "## Average Speedup By Source",
        "",
        "| Source | Scheme | Speedup vs Full_Cache | Speedup vs Full_Recompute | NPU util % | HBM util % | DDR util % | SSD read util % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_ORDER:
        for scheme in SCHEME_ORDER:
            lines.append(
                f"| {source} | {scheme} | {fmt(aggregate(rows, source, scheme, 'speedup_vs_Full_Cache'), 4)} | "
                f"{fmt(aggregate(rows, source, scheme, 'speedup_vs_Full_Recompute'), 4)} | "
                f"{fmt(aggregate(rows, source, scheme, 'npu_util_overall'))} | "
                f"{fmt(aggregate(rows, source, scheme, 'hbm_util_avg'))} | "
                f"{fmt(aggregate(rows, source, scheme, 'ddr_util'))} | "
                f"{fmt(aggregate(rows, source, scheme, 'ssd_read_util'))} |"
            )

    failures = [row for row in rows if row["status"] != "0"]
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- NPU{row['npu_count']} {row['source']} {row['chip']} {row['scheme']}: "
                f"status={row['status']} case={row['case_dir']}"
            )
    else:
        lines.append("- None.")
    (root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def shell_value(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:-="
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_reproduce(root: Path, calibration: Path, metadata: dict) -> None:
    command = metadata.get("expanded_command")
    if not command:
        command = (
            "bash scripts/run_scalability_npus.sh "
            f"--result-root {shell_value(root)} "
            f"--calibration {shell_value(metadata.get('base_calibration', '<schema-v2-calibration.json>'))} "
            f"--calibration-cache-root {shell_value(metadata.get('calibration_cache_root', 'results/MISC/hstu_ir_calibration_cache'))} "
            f"--max-concurrent {shell_value(metadata.get('max_concurrent', 30))}"
        )
    lines = [
        "# Reproduce HSTU Scalability Experiment",
        "",
        "## Main Run",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Calibration Cache",
        "",
        f"- Cache root: `{metadata.get('calibration_root', '')}`",
        f"- Merged calibration: `{calibration}`",
        "- Re-running the same hardware setup reuses the cache unless `--force-calibration` is passed.",
        "",
        "## Important Inputs",
        "",
        f"- Chips: `{','.join(metadata.get('chips', []))}`",
        f"- Sources: `{','.join(metadata.get('sources', []))}`",
        f"- NPU counts: `{','.join(str(v) for v in metadata.get('npu_counts', []))}`",
        f"- Schemes: `{','.join(metadata.get('schemes', []))}`",
        f"- Workload: layers={metadata.get('layers')}, hidden={metadata.get('hidden')}, kv_len={metadata.get('kv_len')}, users_per_batch={metadata.get('users_per_batch')}, candidates_per_user={metadata.get('candidates_per_user')}",
        "",
        "## Regenerate Summary Only",
        "",
        "```bash",
        f"bash scripts/run_scalability_npus.sh --result-root {shell_value(root)} --summary-only",
        "```",
    ]
    (root / "reproduce.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def modelsize_sort_key(row: dict) -> tuple:
    return (
        MODEL_ORDER.index(row["model_size"]) if row["model_size"] in MODEL_ORDER else 99,
        SOURCE_ORDER.index(row["source"]) if row["source"] in SOURCE_ORDER else 99,
        row["chip"],
        SCHEME_ORDER.index(row["scheme"]) if row["scheme"] in SCHEME_ORDER else 99,
    )


def collect_modelsize_rows(cases_root: Path, logs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for case_dir in sorted(cases_root.glob("*/*/*/*")):
        if not case_dir.is_dir():
            continue
        model_size, source_label, chip, scheme = case_dir.parts[-4:]
        status_label = f"{model_size}__{source_label}__{chip}__{scheme}"
        status_path = logs_root / f"{status_label}.status"
        status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "missing"
        metadata = parse_models(case_dir / "models.json")
        hw = parse_hardware_summary(case_dir / "hardware_summary.csv")
        overlap = parse_overlap_summary(case_dir / "layer_breakdown.csv")
        sim_time = hw["sim_time_us"]
        qps = 1_000_000.0 / sim_time if sim_time and sim_time > 0 else None
        rows.append(
            {
                "model_size": model_size,
                "layers": metadata.get("layers", ""),
                "hidden": metadata.get("hidden", ""),
                "source": source_label,
                "npu_count": "1",
                "chip": chip,
                "scheme": scheme,
                "status": status,
                "history_recompute_len": metadata.get("history_recompute_len", ""),
                "kv_reuse_enabled": metadata.get("kv_reuse_enabled", ""),
                "sim_time_us": sim_time,
                "qps": qps,
                "speedup_vs_Full_Cache": None,
                "speedup_vs_Full_Recompute": None,
                "npu_util_overall": hw["npu_util_overall"],
                "npu_core_util_avg": hw["npu_core_util_avg"],
                "npu_core_util_min": hw["npu_core_util_min"],
                "npu_core_util_max": hw["npu_core_util_max"],
                "hbm_util_avg": hw["hbm_util_avg"],
                "hbm_util_min": hw["hbm_util_min"],
                "hbm_util_max": hw["hbm_util_max"],
                "hbm_bw_GBps_total": hw["hbm_bw_GBps_total"],
                "hbm_bw_util_avg": hw["hbm_bw_util_avg"],
                "hbm_bw_util_min": hw["hbm_bw_util_min"],
                "hbm_bw_util_max": hw["hbm_bw_util_max"],
                "ddr_util": hw["ddr_util"],
                "ddr_bw_GBps": hw["ddr_bw_GBps"],
                "ddr_bw_util": hw["ddr_bw_util"],
                "ssd_util": hw["ssd_util"],
                "ssd_read_util": hw["ssd_read_util"],
                "ssd_write_util": hw["ssd_write_util"],
                "ssd_read_bw_GBps": hw["ssd_read_bw_GBps"],
                "ssd_write_bw_GBps": hw["ssd_write_bw_GBps"],
                "preload_overlap_gap_avg_us": overlap["preload_overlap_gap_avg_us"],
                "preload_overlap_gap_min_us": overlap["preload_overlap_gap_min_us"],
                "preload_overlap_gap_max_us": overlap["preload_overlap_gap_max_us"],
                "preload_wait_avg_us": overlap["preload_wait_avg_us"],
                "preload_wait_min_us": overlap["preload_wait_min_us"],
                "preload_wait_max_us": overlap["preload_wait_max_us"],
                "case_dir": str(case_dir),
            }
        )
    rows.sort(key=modelsize_sort_key)
    add_modelsize_speedups(rows)
    return rows


def add_modelsize_speedups(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        groups.setdefault((row["model_size"], row["source"], row["chip"]), {})[row["scheme"]] = row

    for cases in groups.values():
        cache_time = fnum(cases.get("Full_Cache", {}).get("sim_time_us"))
        recompute_time = fnum(cases.get("Full_Recompute", {}).get("sim_time_us"))
        for row in cases.values():
            case_time = fnum(row.get("sim_time_us"))
            if case_time and cache_time:
                row["speedup_vs_Full_Cache"] = cache_time / case_time
            if case_time and recompute_time:
                row["speedup_vs_Full_Recompute"] = recompute_time / case_time


def write_modelsize_recompute_choices(root: Path, rows: list[dict]) -> None:
    choices = [
        {
            "model_size": row["model_size"],
            "source": row["source"],
            "chip": row["chip"],
            "scheme": row["scheme"],
            "history_recompute_len": row["history_recompute_len"],
            "kv_reuse_enabled": row["kv_reuse_enabled"],
            "sim_time_us": row["sim_time_us"],
            "case_dir": row["case_dir"],
        }
        for row in rows
        if row["scheme"] in {"w_IR", "w_both"}
    ]
    fields = [
        "model_size",
        "source",
        "chip",
        "scheme",
        "history_recompute_len",
        "kv_reuse_enabled",
        "sim_time_us",
        "case_dir",
    ]
    write_csv(root / "recompute_choices.csv", choices, fields)
    (root / "recompute_choices.json").write_text(
        json.dumps(choices, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_modelsize_time_qps(root: Path, rows: list[dict]) -> None:
    fields = ["source", "model_size", "chip", "scheme", "time_us", "qps"]
    table = [
        {
            "source": row["source"],
            "model_size": row["model_size"],
            "chip": row["chip"],
            "scheme": row["scheme"],
            "time_us": row["sim_time_us"],
            "qps": row["qps"],
        }
        for row in rows
    ]
    write_csv(root / "time_qps.csv", table, fields)
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(table).to_excel(root / "time_qps.xlsx", index=False)
    except Exception as exc:  # pragma: no cover - optional dependency path
        (root / "time_qps.xlsx.error.txt").write_text(str(exc) + "\n", encoding="utf-8")


def write_modelsize_summary(root: Path, rows: list[dict], metadata: dict) -> None:
    completed = sum(1 for row in rows if row["status"] == "0")
    lines = [
        "# HSTU Model-Size Scalability Summary",
        "",
        f"- Result root: `{root}`",
        f"- Calibration cache: `{metadata.get('calibration_cache_root', '')}`",
        f"- Completed cases: {completed}/{len(rows)}",
        f"- Full CSV: `{root / 'scalability_summary.csv'}`",
        f"- Time/QPS: `{root / 'time_qps.csv'}`",
        "",
        "## Time And Speedup",
        "",
        "| Model | Source | Chip | Scheme | Status | k | Time us | QPS | Speedup vs Full_Cache | Speedup vs Full_Recompute |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_size']} | {row['source']} | {row['chip']} | {row['scheme']} | {row['status']} | "
            f"{row['history_recompute_len']} | {fmt(row['sim_time_us'])} | {fmt(row['qps'], 4)} | "
            f"{fmt(row['speedup_vs_Full_Cache'], 4)} | {fmt(row['speedup_vs_Full_Recompute'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Hardware Utilization",
            "",
            "| Model | Source | Chip | Scheme | NPU util % | Core avg/min/max % | HBM util avg/min/max % | HBM BW util avg/min/max % | HBM BW GB/s | DDR util/BW util/GBps | SSD read/write util/read GBps | Preload gap avg/min/max us |",
            "|---|---|---|---|---:|---|---|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        core = "/".join(fmt(row[key]) for key in ("npu_core_util_avg", "npu_core_util_min", "npu_core_util_max"))
        hbm = "/".join(fmt(row[key]) for key in ("hbm_util_avg", "hbm_util_min", "hbm_util_max"))
        hbm_bw = "/".join(fmt(row[key]) for key in ("hbm_bw_util_avg", "hbm_bw_util_min", "hbm_bw_util_max"))
        ddr = f"{fmt(row['ddr_util'])}/{fmt(row['ddr_bw_util'])}/{fmt(row['ddr_bw_GBps'])}"
        ssd = f"{fmt(row['ssd_read_util'])}/{fmt(row['ssd_write_util'])}/{fmt(row['ssd_read_bw_GBps'])}"
        gap = "/".join(fmt(row[key]) for key in ("preload_overlap_gap_avg_us", "preload_overlap_gap_min_us", "preload_overlap_gap_max_us"))
        lines.append(
            f"| {row['model_size']} | {row['source']} | {row['chip']} | {row['scheme']} | "
            f"{fmt(row['npu_util_overall'])} | {core} | {hbm} | {hbm_bw} | "
            f"{fmt(row['hbm_bw_GBps_total'])} | {ddr} | {ssd} | {gap} |"
        )

    failures = [row for row in rows if row["status"] != "0"]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- {row['model_size']} {row['source']} {row['chip']} {row['scheme']}: status={row['status']}" for row in failures)
    else:
        lines.append("- None.")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_modelsize_reproduce(root: Path, metadata: dict) -> None:
    command = metadata.get("expanded_command", "bash scripts/run_scalability.sh")
    lines = [
        "# Reproduce HSTU Model-Size Scalability",
        "",
        "```bash",
        command,
        "```",
        "",
        f"- Calibration cache root: `{metadata.get('calibration_cache_root', '')}`",
        f"- Result root: `{root}`",
        "- Same hardware/config hash reuses calibration unless `--force-calibration` is passed.",
        "",
        "Regenerate summary only:",
        "",
        "```bash",
        f"bash scripts/run_scalability.sh --result-root {root} --summary-only",
        "```",
    ]
    (root / "reproduce.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def modelsize_fieldnames() -> list[str]:
    return [
        "model_size",
        "layers",
        "hidden",
        "source",
        "npu_count",
        "chip",
        "scheme",
        "status",
        "history_recompute_len",
        "kv_reuse_enabled",
        "sim_time_us",
        "qps",
        "speedup_vs_Full_Cache",
        "speedup_vs_Full_Recompute",
        "npu_util_overall",
        "npu_core_util_avg",
        "npu_core_util_min",
        "npu_core_util_max",
        "hbm_util_avg",
        "hbm_util_min",
        "hbm_util_max",
        "hbm_bw_GBps_total",
        "hbm_bw_util_avg",
        "hbm_bw_util_min",
        "hbm_bw_util_max",
        "ddr_util",
        "ddr_bw_GBps",
        "ddr_bw_util",
        "ssd_util",
        "ssd_read_util",
        "ssd_write_util",
        "ssd_read_bw_GBps",
        "ssd_write_bw_GBps",
        "preload_overlap_gap_avg_us",
        "preload_overlap_gap_min_us",
        "preload_overlap_gap_max_us",
        "preload_wait_avg_us",
        "preload_wait_min_us",
        "preload_wait_max_us",
        "case_dir",
    ]


def summarize_modelsize(root: Path, cases_root: Path, logs_root: Path, metadata: dict) -> None:
    rows = collect_modelsize_rows(cases_root, logs_root)
    write_csv(root / "scalability_summary.csv", rows, modelsize_fieldnames())
    write_modelsize_time_qps(root, rows)
    write_modelsize_recompute_choices(root, rows)
    write_modelsize_summary(root, rows, metadata)
    write_modelsize_reproduce(root, metadata)
    print(f"Summary: {root / 'summary.md'}")
    print(f"CSV: {root / 'scalability_summary.csv'}")
    print(f"Time/QPS: {root / 'time_qps.csv'}")


def detect_layout(requested: str, metadata: dict, cases_root: Path) -> str:
    if requested != "auto":
        return requested
    if metadata.get("kind") == "hstu_modelsize_scalability":
        return "modelsize"
    has_npu_dirs = any(path.is_dir() and path.name.startswith("NPU") for path in cases_root.glob("NPU*"))
    return "npus" if has_npu_dirs else "modelsize"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=Path(""))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--layout", choices=["auto", "npus", "modelsize"], default="auto")
    args = parser.parse_args()

    root = args.result_root
    cases_root = args.cases_root or root / "cases"
    logs_root = args.logs_root or root / "logs"
    metadata = read_json(args.metadata) if args.metadata else read_json(root / "run_metadata.json")
    layout = detect_layout(args.layout, metadata, cases_root)
    if layout == "modelsize":
        summarize_modelsize(root, cases_root, logs_root, metadata)
        return

    rows = collect_rows(cases_root, logs_root)
    fieldnames = [
        "npu_count",
        "source",
        "chip",
        "scheme",
        "status",
        "history_recompute_len",
        "kv_reuse_enabled",
        "sim_time_us",
        "speedup_vs_Full_Cache",
        "speedup_vs_Full_Recompute",
        "npu_util_overall",
        "npu_core_util_avg",
        "npu_core_util_min",
        "npu_core_util_max",
        "hbm_util_avg",
        "hbm_util_min",
        "hbm_util_max",
        "hbm_bw_GBps_total",
        "hbm_bw_util_avg",
        "hbm_bw_util_min",
        "hbm_bw_util_max",
        "ddr_util",
        "ddr_bw_GBps",
        "ddr_bw_util",
        "ssd_util",
        "ssd_read_util",
        "ssd_write_util",
        "ssd_read_bw_GBps",
        "ssd_write_bw_GBps",
        "preload_overlap_gap_avg_us",
        "preload_overlap_gap_min_us",
        "preload_overlap_gap_max_us",
        "preload_wait_avg_us",
        "preload_wait_min_us",
        "preload_wait_max_us",
        "case_dir",
    ]
    write_csv(root / "scalability_summary.csv", rows, fieldnames)
    write_recompute_choices(root, rows)
    write_summary(root, rows, args.calibration, metadata)
    write_analysis(root, rows)
    write_reproduce(root, args.calibration, metadata)
    print(f"Summary: {root / 'summary.md'}")
    print(f"CSV: {root / 'scalability_summary.csv'}")
    print(f"Analysis: {root / 'analysis.md'}")
    print(f"Reproduce: {root / 'reproduce.md'}")


if __name__ == "__main__":
    main()
