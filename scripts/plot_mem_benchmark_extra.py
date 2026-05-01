#!/usr/bin/env python3

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


FIG_WIDTH = 9.8
FIG_HEIGHT = 5.2
BURST_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
COLORS = [
    "#0b7285",
    "#c92a2a",
    "#2b8a3e",
    "#e67700",
    "#5f3dc4",
    "#364fc7",
]
BACKGROUND = "#fbf8ef"
GRID_COLOR = "#d0d0d0"
TEXT_COLOR = "#111111"


def load_summary_rows(summary_csv_path):
    with open(summary_csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_detail_rows(detail_csv_path):
    with open(detail_csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def legend_label(size_bytes):
    return f"{size_bytes // 1024}KB" if size_bytes >= 1024 else f"{size_bytes}B"


def configure_axes(ax, title, ylabel):
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_xlabel("Burst count", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BURST_VALUES)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)


def group_rows(rows, metric_key):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["medium"], row["rw"])
        size = int(row["access_size_bytes"])
        grouped[key][size].append(
            {
                "burst_count": int(row["burst_count"]),
                "metric_value": float(row[metric_key]),
            }
        )
    return grouped


def build_avg_device_latency_rows(summary_rows, detail_rows):
    case_device_latencies = defaultdict(list)
    for row in detail_rows:
        case_device_latencies[row["case_id"]].append(float(row["device_latency_ns"]))

    aggregated_rows = []
    for row in summary_rows:
        latencies = case_device_latencies.get(row["case_id"], [])
        avg_device_latency_ns = sum(latencies) / len(latencies) if latencies else 0.0
        aggregated_rows.append(
            {
                "medium": row["medium"],
                "rw": row["rw"],
                "access_size_bytes": row["access_size_bytes"],
                "burst_count": row["burst_count"],
                "avg_device_latency_ns": avg_device_latency_ns,
            }
        )
    return aggregated_rows


def plot_metric(output_dir, medium, rw, size_rows, title_suffix, ylabel, suffix):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor(BACKGROUND)

    for index, size_bytes in enumerate(sorted(size_rows)):
        rows = sorted(size_rows[size_bytes], key=lambda row: row["burst_count"])
        bursts = [row["burst_count"] for row in rows]
        metric_values = [row["metric_value"] for row in rows]
        ax.plot(
            bursts,
            metric_values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=COLORS[index % len(COLORS)],
            label=legend_label(size_bytes),
        )

    configure_axes(ax, f"{medium.upper()} {rw} {title_suffix}", ylabel)
    legend = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    base_path = os.path.join(output_dir, f"{medium}_{rw}_{suffix}")
    fig.savefig(f"{base_path}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: plot_mem_benchmark_extra.py <summary.csv> <detail.csv> <output_dir>"
        )
        return 1

    summary_csv_path = sys.argv[1]
    detail_csv_path = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = load_summary_rows(summary_csv_path)
    detail_rows = load_detail_rows(detail_csv_path)

    total_time_grouped = group_rows(summary_rows, "total_time_ns")
    for (medium, rw), size_rows in total_time_grouped.items():
        plot_metric(
            output_dir,
            medium,
            rw,
            size_rows,
            "Total End-to-End Time",
            "Total Time (ns)",
            "total_time",
        )

    avg_device_latency_rows = build_avg_device_latency_rows(summary_rows, detail_rows)
    avg_device_grouped = group_rows(avg_device_latency_rows, "avg_device_latency_ns")
    for (medium, rw), size_rows in avg_device_grouped.items():
        plot_metric(
            output_dir,
            medium,
            rw,
            size_rows,
            "Average Device Latency",
            "Average Device Latency (ns)",
            "avg_device_latency",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
