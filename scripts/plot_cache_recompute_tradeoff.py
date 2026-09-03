#!/usr/bin/env python3
"""Plot utilization and item-KV recomputation trade-off for one HSTU case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/hstu_calibration_matrix_20260901"),
    )
    parser.add_argument("--chip", default="910B")
    parser.add_argument("--model", default="middle")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--user", choices=("hot", "cold"), default="cold")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "results/figures/cache_recompute_tradeoff_"
            "910B_middle_seq4096_bs4_cold"
        ),
    )
    return parser.parse_args()


def monotone_cubic_curve(
    x: np.ndarray, y: np.ndarray, sample_count: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth, shape-preserving cubic Hermite interpolation."""

    spacing = np.diff(x)
    secant = np.diff(y) / spacing
    slope = np.zeros_like(y, dtype=float)
    for index in range(1, len(y) - 1):
        left, right = secant[index - 1], secant[index]
        if left * right > 0:
            weight_left = 2 * spacing[index] + spacing[index - 1]
            weight_right = spacing[index] + 2 * spacing[index - 1]
            slope[index] = (weight_left + weight_right) / (
                weight_left / left + weight_right / right
            )

    slope[0] = secant[0]
    slope[-1] = secant[-1]
    smooth_x = np.linspace(x[0], x[-1], sample_count)
    smooth_y = np.empty_like(smooth_x)
    for index in range(len(x) - 1):
        mask = (smooth_x >= x[index]) & (smooth_x <= x[index + 1])
        position = (smooth_x[mask] - x[index]) / spacing[index]
        h00 = 2 * position**3 - 3 * position**2 + 1
        h10 = position**3 - 2 * position**2 + position
        h01 = -2 * position**3 + 3 * position**2
        h11 = position**3 - position**2
        smooth_y[mask] = (
            h00 * y[index]
            + h10 * spacing[index] * slope[index]
            + h01 * y[index + 1]
            + h11 * spacing[index] * slope[index + 1]
        )
    return smooth_x, smooth_y


def read_hardware_summary(case_dir: Path) -> dict[str, float]:
    path = case_dir / "hardware_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    npu = next(
        row for row in rows
        if row["component"] == "NPU" and row["scope"] == "overall"
    )
    ssd = next(
        row for row in rows
        if row["component"] == "SSD" and row["scope"] == "overall"
    )
    return {
        "latency_ms": float(npu["sim_time_us"]) / 1000.0,
        "npu_utilization": float(npu["utilization_percent"]) / 100.0,
        "ssd_utilization": float(ssd["bandwidth_utilization_percent"]) / 100.0,
    }


def ratio_record(root: Path, args: argparse.Namespace, ratio_index: int) -> dict:
    case_id = (
        f"{args.chip}__{args.model}__seq{args.seq_len}__bs{args.batch_size}"
        f"__{args.user}__AR_IR__r{ratio_index * 10:03d}"
    )
    path = root / "w_both_ratio" / "logs" / f"{case_id}.status.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("returncode") != 0:
        raise RuntimeError(f"ratio point is not complete: {case_id}")
    return record


def ratio_case_dir(root: Path, args: argparse.Namespace, ratio_index: int) -> Path:
    return (
        root / "w_both_ratio" / "cases" / args.chip / "AR_IR_ratio"
        / f"r{ratio_index * 10:03d}"
        / f"HSTU-{args.model}_seq{args.seq_len}_bs{args.batch_size}_{args.user}"
    )


def write_data(
    path: Path,
    base_metrics: dict[str, dict[str, float]],
    normalized_metrics: dict[str, dict[str, float]],
    ratio_rows: list[dict[str, float]],
) -> None:
    fields = [
        "series", "method", "requested_ratio", "recomputed_item_tokens",
        "latency_ms", "normalized_npu_utilization",
        "normalized_ssd_bandwidth_utilization",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, metrics in base_metrics.items():
            writer.writerow({
                "series": "utilization",
                "method": method,
                "latency_ms": metrics["latency_ms"],
                "normalized_npu_utilization": normalized_metrics[method]["npu"],
                "normalized_ssd_bandwidth_utilization": normalized_metrics[method]["ssd"],
            })
        for row in ratio_rows:
            writer.writerow({
                "series": "recompute_tradeoff",
                "method": "AR+recompute",
                "requested_ratio": row["ratio"],
                "recomputed_item_tokens": int(row["tokens"]),
                "latency_ms": row["latency_ms"],
            })


def main() -> None:
    args = parse_args()
    root = args.result_root
    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)

    ratio_rows = []
    for index in range(11):
        record = ratio_record(root, args, index)
        metrics = read_hardware_summary(ratio_case_dir(root, args, index))
        ratio_rows.append({
            "ratio": index / 10.0,
            "tokens": record["history_recompute_len"],
            "latency_ms": metrics["latency_ms"],
        })

    # Both endpoints include AR.  Only the item-token recomputation ratio
    # changes: ratio=0 is Full KV Cache and ratio=1 is Full Recompute.
    base_metrics = {
        "Full recompute": read_hardware_summary(ratio_case_dir(root, args, 10)),
        "Full KV cache": read_hardware_summary(ratio_case_dir(root, args, 0)),
    }
    npu_reference = max(metrics["npu_utilization"] for metrics in base_metrics.values())
    ssd_reference = max(metrics["ssd_utilization"] for metrics in base_metrics.values())
    normalized_metrics = {
        method: {
            "npu": metrics["npu_utilization"] / npu_reference,
            "ssd": metrics["ssd_utilization"] / ssd_reference,
        }
        for method, metrics in base_metrics.items()
    }

    write_data(
        output.with_suffix(".csv"), base_metrics, normalized_metrics, ratio_rows
    )

    colors = {
        "blue": "#4E79A7",
        "orange": "#D99536",
        "teal": "#3F8F88",
        "red": "#C44E52",
        "grid": "#D9D9D9",
        "text": "#262626",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "axes.edgecolor": colors["text"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": colors["text"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": colors["text"],
        "ytick.color": colors["text"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 7.5,
        "legend.handlelength": 1.8,
        "hatch.linewidth": 1.0,
        "text.color": colors["text"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(6.34, 2.55),
        gridspec_kw={"width_ratios": (4, 4.8), "wspace": 0.28},
    )

    methods = list(base_metrics)
    method_labels = ["Full recompute", "Full KV cache"]
    x = np.arange(len(methods))
    width = 0.32
    npu = [normalized_metrics[name]["npu"] for name in methods]
    ssd = [normalized_metrics[name]["ssd"] for name in methods]
    bars_npu = ax0.bar(
        x - width / 2, npu, width, label="NPU",
        color="#DCE6F1", edgecolor=colors["blue"], linewidth=1.0,
        hatch="...",
    )
    bars_ssd = ax0.bar(
        x + width / 2, ssd, width, label="SSD bandwidth",
        color="#F7E6CD", edgecolor=colors["orange"], linewidth=1.0,
        hatch="///",
    )
    ax0.set_ylabel("Normalized utilization")
    ax0.set_xticks(x, method_labels)
    ax0.set_ylim(0, 1.22)
    ax0.set_yticks(np.arange(0, 1.01, 0.2))
    ax0.grid(axis="y", color=colors["grid"], linewidth=0.55, alpha=0.75)
    ax0.set_axisbelow(True)
    ax0.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, columnspacing=1.2, handletextpad=0.5,
    )
    ax0.set_title("(a) Resource utilization", loc="center", pad=7)

    tokens = np.array([row["tokens"] for row in ratio_rows])
    latency = np.array([row["latency_ms"] for row in ratio_rows])
    best = int(np.argmin(latency))
    smooth_tokens, smooth_latency = monotone_cubic_curve(tokens, latency)
    ax1.plot(
        smooth_tokens, smooth_latency, color=colors["blue"], linewidth=1.8,
        solid_capstyle="round",
    )
    regular_points = np.arange(len(tokens)) != best
    ax1.scatter(
        tokens[regular_points], latency[regular_points],
        s=16, color="white", edgecolor=colors["blue"],
        linewidth=1.0, zorder=3,
    )
    ax1.scatter(
        [tokens[best]], [latency[best]], s=88, marker="*",
        color="#E5B84B", edgecolor=colors["text"], linewidth=0.6, zorder=4,
    )
    ax1.axhline(
        latency[0], color=colors["orange"], linewidth=0.95,
        linestyle=(0, (5, 3)), zorder=1,
    )
    ax1.axhline(
        latency[-1], color="#888888", linewidth=0.95,
        linestyle=(0, (2, 3)), zorder=1,
    )
    label_x = tokens[0] + 0.025 * (tokens[-1] - tokens[0])
    ax1.annotate(
        "Full KV Cache", xy=(label_x, latency[0]), xytext=(0, 4),
        textcoords="offset points", ha="left", va="bottom",
        color=colors["orange"], fontsize=7.3, fontweight="semibold",
    )
    ax1.annotate(
        "Full Recompute", xy=(label_x, latency[-1]), xytext=(0, -4),
        textcoords="offset points", ha="left", va="top",
        color="#777777", fontsize=7.3, fontweight="semibold",
    )
    ax1.margins(y=0.08)
    ax1.set_xlabel("Recomputed item tokens")
    ax1.set_ylabel("End-to-end latency (ms)")
    ax1.set_xticks(tokens[::2])
    ax1.grid(color=colors["grid"], linewidth=0.55, alpha=0.75)
    ax1.set_axisbelow(True)
    ax1.set_title("(b) Recompute trade-off", loc="center", pad=7)
    for axis in (ax0, ax1):
        for side in ("left", "right", "top", "bottom"):
            axis.spines[side].set_visible(True)
            axis.spines[side].set_linewidth(0.8)
    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.20, top=0.86, wspace=0.30
    )
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(json.dumps({
        "case": {
            "chip": args.chip, "model": args.model, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "user": args.user,
        },
        "full_cache": base_metrics["Full KV cache"],
        "full_recompute": base_metrics["Full recompute"],
        "normalized_utilization": normalized_metrics,
        "ratio_zero_latency_ms": ratio_rows[0]["latency_ms"],
        "minimum_latency_ms": ratio_rows[best]["latency_ms"],
        "minimum_ratio": ratio_rows[best]["ratio"],
        "minimum_recomputed_item_tokens": ratio_rows[best]["tokens"],
        "outputs": [str(output.with_suffix(s)) for s in (".pdf", ".png", ".csv")],
    }, indent=2))


if __name__ == "__main__":
    main()
