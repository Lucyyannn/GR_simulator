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


def load_rows(summary_csv_path):
    with open(summary_csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["medium"], row["rw"])
        size = int(row["access_size_bytes"])
        grouped[key][size].append(row)
    return grouped


def legend_label(size_bytes):
    return f"{size_bytes // 1024}KB" if size_bytes >= 1024 else f"{size_bytes}B"


def sorted_case_rows(size_rows, size_bytes):
    return sorted(size_rows[size_bytes], key=lambda row: int(row["burst_count"]))


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


def plot_metric(output_dir, medium, rw, size_rows, metric_key, title_suffix, ylabel, suffix):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor(BACKGROUND)

    for index, size_bytes in enumerate(sorted(size_rows)):
        rows = sorted_case_rows(size_rows, size_bytes)
        bursts = [int(row["burst_count"]) for row in rows]
        metric_values = [float(row[metric_key]) for row in rows]
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
    if len(sys.argv) != 3:
        print("Usage: plot_mem_benchmark.py <summary.csv> <output_dir>")
        return 1

    summary_csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    rows = load_rows(summary_csv_path)
    grouped = group_rows(rows)
    for (medium, rw), size_rows in grouped.items():
        plot_metric(
            output_dir,
            medium,
            rw,
            size_rows,
            "macro_avg_latency_ns",
            "Average Latency",
            "Latency (ns)",
            "latency",
        )
        plot_metric(
            output_dir,
            medium,
            rw,
            size_rows,
            "bandwidth_GBps",
            "Bandwidth",
            "Bandwidth (GB/s)",
            "bandwidth",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
