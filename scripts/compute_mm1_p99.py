#!/usr/bin/env python3
"""Discover HSTU latency results and compute steady-state M/M/1 P99."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_MODEL = "HSTU-middle"
TARGET_SEQ_LEN = 16384
TARGET_BATCH_SIZE = 1
TARGET_WORKLOAD = "cold"

METHODS = (
    "Full_Recompute",
    "Full_Cache",
    "w_AR",
    "w_IR",
    "w_both",
)

REQUEST_RATES = [16, 32, 48, 64, 80, 96, 114, 128]
OUTPUT_CSV_NAME = "p99_mm1_HSTU_middle_seq16384_bs1_cold.csv"
OUTPUT_PNG_NAME = "p99_mm1_HSTU_middle_seq16384_bs1_cold.png"
PLOT_Y_MAX_MS = 400.0
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".jsonl"}
SEARCH_ROOTS = (
    Path("results/main_task"),
    Path("result/main_task"),
    Path("results.bak/main_task"),
)

METHOD_DIRS = {
    "Recompute": "Full_Recompute",
    "FullCache": "Full_Cache",
    "W_AR": "w_AR",
    "W_IR": "w_IR",
    "W_both": "w_both",
}

CONFIG_COLUMN_ALIASES = {
    "model": ("model", "model_name", "network", "architecture"),
    "seq_len": ("seq_len", "sequence_length", "seqlen", "seq", "history_length"),
    "batch_size": ("batch_size", "batch", "bs"),
    "workload": ("workload", "access_pattern", "mode", "temperature"),
}

METHOD_COLUMN_ALIASES = ("method", "scheme", "variant", "approach", "case")
LATENCY_COLUMN_HINTS = (
    "latency_ms",
    "latency",
    "sim_time_ms",
    "sim_time_us",
    "service_time_ms",
    "mean_latency_ms",
    "avg_latency_ms",
    "duration_ms",
    "duration_us",
)


def log(message: str) -> None:
    print(message)


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def numeric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group(0))
    if np.isnan(number):
        return None
    return number


def get_result_root() -> Path:
    for path in SEARCH_ROOTS:
        if path.is_dir():
            return path
    searched_text = "\n".join(f"  - {path}" for path in SEARCH_ROOTS)
    raise SystemExit(f"Could not find result root. Searched:\n{searched_text}")


def output_dir_for_result_root(root: Path) -> Path:
    return root.parent


def discover_candidate_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def score_candidate_file(path: Path) -> int:
    text = path.as_posix().lower()
    score = 0
    keyword_weights = {
        "hstu-middle": 10,
        "hstu_middle": 10,
        "seq16384": 10,
        "seq_16384": 10,
        "seq-16384": 10,
        "bs1": 8,
        "bs_1": 8,
        "bs-1": 8,
        "cold": 8,
        "hardware_summary": 6,
        "latency": 4,
        "summary": 4,
        "comparison": 2,
        "result": 1,
    }
    for keyword, weight in keyword_weights.items():
        if keyword in text:
            score += weight

    for part in path.parts:
        if normalize_method(part) in METHODS:
            score += 6
            break
    return score


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            for key in ("rows", "data", "results", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.json_normalize(value)
            return pd.json_normalize([payload])
    raise ValueError(f"Unsupported file suffix: {path.suffix}")


def normalize_method(value: object) -> str | None:
    compact = normalize_token(value)
    if compact in {"fullrecompute", "recompute"}:
        return "Full_Recompute"
    if compact in {"fullcache", "cache"}:
        return "Full_Cache"
    if compact in {"war", "withar", "actionreuse", "wactionreuse"}:
        return "w_AR"
    if compact in {"wir", "withir", "itemrecompute", "witemrecompute"}:
        return "w_IR"
    if compact in {"wboth", "withboth", "both"}:
        return "w_both"
    return None


def infer_method_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        method = METHOD_DIRS.get(part)
        if method:
            return method
        method = normalize_method(part)
        if method:
            return method
    return None


def path_target_fields(path: Path) -> dict[str, bool]:
    text = path.as_posix().lower()
    return {
        "model": "hstu-middle" in text or "hstu_middle" in text,
        "seq_len": bool(re.search(r"seq[_-]?16384(?:\D|$)", text)),
        "batch_size": bool(re.search(r"bs[_-]?1(?:\D|$)", text)),
        "workload": bool(re.search(r"(?:^|[/_\-])cold(?:[/_\-.]|$)", text)),
    }


def find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize_column(column): column for column in df.columns}
    for alias in aliases:
        key = normalize_column(alias)
        if key in normalized:
            return normalized[key]
    return None


def find_method_column(df: pd.DataFrame) -> str | None:
    return find_column(df, METHOD_COLUMN_ALIASES)


def find_latency_column(df: pd.DataFrame) -> str | None:
    normalized = {normalize_column(column): column for column in df.columns}
    for hint in LATENCY_COLUMN_HINTS:
        key = normalize_column(hint)
        if key in normalized:
            return normalized[key]
    for key, column in normalized.items():
        if "latency" in key or key in {"sim_time", "sim_time_us", "sim_time_ms"}:
            return column
    return None


def value_matches_model(value: object) -> bool:
    return "hstumiddle" in normalize_token(value)


def value_matches_seq(value: object) -> bool:
    number = numeric_value(value)
    if number is not None and int(number) == TARGET_SEQ_LEN:
        return True
    return bool(re.search(r"(?:^|[^0-9])16384(?:[^0-9]|$)", str(value)))


def value_matches_batch(value: object) -> bool:
    number = numeric_value(value)
    if number is not None and int(number) == TARGET_BATCH_SIZE:
        return True
    return bool(re.search(r"(?:^|[^0-9])1(?:[^0-9]|$)", str(value)))


def value_matches_workload(value: object) -> bool:
    return TARGET_WORKLOAD in str(value).strip().lower()


def filter_target_rows(
    df: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, list[str], list[str]]:
    path_fields = path_target_fields(path)
    matched_fields = [
        f"{field}=path" for field, matched in path_fields.items() if matched
    ]
    if all(path_fields.values()):
        return df.copy(), matched_fields, []

    mask = pd.Series(True, index=df.index)
    missing_fields: list[str] = []
    matchers = {
        "model": value_matches_model,
        "seq_len": value_matches_seq,
        "batch_size": value_matches_batch,
        "workload": value_matches_workload,
    }
    for field, aliases in CONFIG_COLUMN_ALIASES.items():
        column = find_column(df, aliases)
        if column is None:
            if not path_fields[field]:
                missing_fields.append(field)
            continue
        matched_fields.append(f"{field}=column:{column}")
        mask &= df[column].map(matchers[field])

    return df.loc[mask].copy(), matched_fields, missing_fields


def latency_to_ms(value: object, column_name: str) -> float | None:
    number = numeric_value(value)
    if number is None or number <= 0:
        return None
    column = normalize_column(column_name)
    if column.endswith("_us") or column.endswith("time_us") or column == "sim_time_us":
        return number / 1000.0
    return number


def extract_hardware_summary_latency(
    df: pd.DataFrame, path: Path
) -> dict[str, float]:
    method = infer_method_from_path(path)
    if method is None:
        return {}
    if not all(path_target_fields(path).values()):
        return {}
    component_col = find_column(df, ("component",))
    scope_col = find_column(df, ("scope",))
    sim_time_col = find_column(df, ("sim_time_us",))
    if not component_col or not scope_col or not sim_time_col:
        return {}

    rows = df[
        df[component_col].astype(str).str.lower().eq("npu")
        & df[scope_col].astype(str).str.lower().eq("overall")
    ]
    if rows.empty:
        return {}
    latency_ms = latency_to_ms(rows.iloc[0][sim_time_col], sim_time_col)
    if latency_ms is None:
        return {}
    return {method: latency_ms}


def extract_wide_latency(df: pd.DataFrame, path: Path) -> dict[str, float]:
    target_rows, _, _ = filter_target_rows(df, path)
    if target_rows.empty:
        return {}
    result: dict[str, float] = {}
    row = target_rows.iloc[0]
    for column in target_rows.columns:
        method = normalize_method(column)
        if method not in METHODS:
            continue
        latency_ms = latency_to_ms(row[column], str(column))
        if latency_ms is not None:
            result[method] = latency_ms
    return result


def extract_long_latency(df: pd.DataFrame, path: Path) -> dict[str, float]:
    target_rows, _, _ = filter_target_rows(df, path)
    if target_rows.empty:
        return {}
    method_col = find_method_column(target_rows)
    latency_col = find_latency_column(target_rows)
    if method_col is None or latency_col is None:
        return {}

    result: dict[str, float] = {}
    for _, row in target_rows.iterrows():
        method = normalize_method(row[method_col])
        if method not in METHODS:
            continue
        latency_ms = latency_to_ms(row[latency_col], latency_col)
        if latency_ms is not None:
            result[method] = latency_ms
    return result


def extract_target_latency(df: pd.DataFrame, path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for extractor in (
        extract_hardware_summary_latency,
        extract_wide_latency,
        extract_long_latency,
    ):
        result.update(extractor(df, path))
    return {method: result[method] for method in METHODS if method in result}


def inspect_candidate(df: pd.DataFrame, path: Path) -> tuple[list[str], list[str]]:
    _, matched_fields, missing_fields = filter_target_rows(df, path)
    method = infer_method_from_path(path)
    if method:
        matched_fields.append(f"method=path:{method}")
    elif find_method_column(df):
        matched_fields.append(f"method=column:{find_method_column(df)}")
    else:
        missing_fields.append("method")

    if find_latency_column(df):
        matched_fields.append(f"latency=column:{find_latency_column(df)}")
    elif find_column(df, ("sim_time_us",)):
        matched_fields.append("latency=column:sim_time_us")
    else:
        missing_fields.append("latency")
    return matched_fields, sorted(set(missing_fields))


def mm1_p99_ms(latency_ms: object, request_rate: float) -> float:
    latency = numeric_value(latency_ms)
    if latency is None or latency <= 0:
        return float("nan")
    service_time_s = latency / 1000.0
    mu = 1.0 / service_time_s
    lam = request_rate
    rho = lam / mu
    if rho >= 1.0:
        return 9999.0
    return float(np.log(100.0) / (mu - lam) * 1000.0)


def compute_p99_table(
    latency_dict: dict[str, float], request_rates: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for request_rate in request_rates:
        row: dict[str, float] = {"request_rate": request_rate}
        for method in METHODS:
            row[method] = mm1_p99_ms(latency_dict.get(method), request_rate)
        rows.append(row)
    return pd.DataFrame(rows, columns=("request_rate", *METHODS))


def plot_p99_table(df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = {
        "Full_Recompute": "#4C566A",
        "Full_Cache": "#5E81AC",
        "w_AR": "#A3BE8C",
        "w_IR": "#EBCB8B",
        "w_both": "#BF616A",
    }
    request_rates = df["request_rate"].astype(int).tolist()
    x = np.arange(len(request_rates), dtype=float)
    for method in METHODS:
        values = df[method].astype(float).to_numpy()
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=method,
            color=colors.get(method),
        )

    ax.set_xlabel("request rate")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_ylim(0, PLOT_Y_MAX_MS)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(rate) for rate in request_rates])
    ax.grid(axis="both", linestyle="--", alpha=0.28)
    ax.set_title("M/M/1 steady-state P99 latency")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_debug_report(reports: list[dict[str, Any]], limit: int | None = None) -> None:
    rows = reports if limit is None else reports[:limit]
    for report in rows:
        log(
            "  "
            f"score={report['score']:>3} "
            f"path={report['path']} "
            f"matched={report.get('matched_fields', [])} "
            f"missing={report.get('missing_fields', [])}"
        )
        if report.get("error"):
            log(f"    error={report['error']}")
        if report.get("latencies"):
            log(f"    extracted={report['latencies']}")


def unstable_methods_for_row(row: pd.Series) -> list[str]:
    return [method for method in METHODS if float(row[method]) == 9999.0]


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("This script does not accept arguments; use auto-discovery.")

    root = get_result_root()
    output_dir = output_dir_for_result_root(root)
    output_csv = output_dir / OUTPUT_CSV_NAME
    output_png = output_dir / OUTPUT_PNG_NAME
    candidates = discover_candidate_files(root)
    scored_candidates = sorted(
        ((score_candidate_file(path), path) for path in candidates),
        key=lambda item: (-item[0], item[1].as_posix()),
    )

    log(f"Using result root for input data: {root}")
    log(f"Output directory: {output_dir}")
    log(f"Discovered candidate files: {len(candidates)}")
    log("Top scored candidates:")
    for score, path in scored_candidates[:20]:
        log(f"  score={score:>3} path={path}")

    selected: dict[str, tuple[float, int, Path]] = {}
    reports: list[dict[str, Any]] = []
    scanned_all = False

    for index, (score, path) in enumerate(scored_candidates):
        try:
            df = load_table(path)
            matched_fields, missing_fields = inspect_candidate(df, path)
            latencies = extract_target_latency(df, path)
            report = {
                "path": path,
                "score": score,
                "matched_fields": matched_fields,
                "missing_fields": missing_fields,
                "latencies": latencies,
            }
        except Exception as exc:
            report = {
                "path": path,
                "score": score,
                "matched_fields": [],
                "missing_fields": ["unreadable"],
                "error": f"{type(exc).__name__}: {exc}",
                "latencies": {},
            }
            latencies = {}
        reports.append(report)

        for method, latency_ms in latencies.items():
            existing = selected.get(method)
            if existing is None or score > existing[1]:
                selected[method] = (latency_ms, score, path)

        if all(method in selected for method in METHODS):
            break
        if index == len(scored_candidates) - 1:
            scanned_all = True

    missing_methods = [method for method in METHODS if method not in selected]
    if missing_methods and not scanned_all:
        for score, path in scored_candidates[len(reports) :]:
            try:
                df = load_table(path)
                matched_fields, missing_fields = inspect_candidate(df, path)
                latencies = extract_target_latency(df, path)
                report = {
                    "path": path,
                    "score": score,
                    "matched_fields": matched_fields,
                    "missing_fields": missing_fields,
                    "latencies": latencies,
                }
            except Exception as exc:
                report = {
                    "path": path,
                    "score": score,
                    "matched_fields": [],
                    "missing_fields": ["unreadable"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "latencies": {},
                }
                latencies = {}
            reports.append(report)
            for method, latency_ms in latencies.items():
                existing = selected.get(method)
                if existing is None or score > existing[1]:
                    selected[method] = (latency_ms, score, path)

    missing_methods = [method for method in METHODS if method not in selected]
    log("Candidate files that produced latency values:")
    extracted_reports = [report for report in reports if report.get("latencies")]
    print_debug_report(extracted_reports)

    if missing_methods:
        searched_paths = "\n".join(f"  - {path}" for path in SEARCH_ROOTS)
        log("Failed to find all target latencies.")
        log(f"Missing methods: {missing_methods}")
        log(f"Searched paths:\n{searched_paths}")
        log(f"Candidate file count: {len(candidates)}")
        log("Per-candidate debug:")
        print_debug_report(reports)
        raise SystemExit(1)

    latency_dict = {method: selected[method][0] for method in METHODS}
    log("Selected latency source files:")
    for method in METHODS:
        latency_ms, score, path = selected[method]
        log(f"  {method}: {path} (score={score}, latency_ms={latency_ms:.6g})")

    log("Raw latency:")
    for method in METHODS:
        log(f"  {method} latency_ms = {latency_dict[method]:.6g}")

    p99_df = compute_p99_table(latency_dict, REQUEST_RATES)
    output_dir.mkdir(parents=True, exist_ok=True)
    p99_df.to_csv(output_csv, index=False)
    plot_p99_table(p99_df, output_png)

    log("P99 table (ms):")
    log(p99_df.to_string(index=False))
    log("Methods set to 9999 by request rate:")
    for _, row in p99_df.iterrows():
        rate = int(row["request_rate"])
        unstable = unstable_methods_for_row(row)
        if unstable:
            log(f"  request_rate={rate}: {', '.join(unstable)}")
        else:
            log(f"  request_rate={rate}: none")
    log(f"Wrote {output_csv}")
    log(f"Wrote {output_png} (values > {PLOT_Y_MAX_MS:g} ms are clipped by the y-axis)")


if __name__ == "__main__":
    main()
