#!/usr/bin/env python3

import csv
import math
import os
import sys
from collections import defaultdict
from xml.sax.saxutils import escape


WIDTH = 980
HEIGHT = 520
MARGIN_LEFT = 90
MARGIN_RIGHT = 180
MARGIN_TOP = 60
MARGIN_BOTTOM = 70
PLOT_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_HEIGHT = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
BURST_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
COLORS = [
    "#0b7285",
    "#c92a2a",
    "#2b8a3e",
    "#e67700",
    "#5f3dc4",
    "#364fc7",
]


def load_rows(summary_csv_path):
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["medium"], row["rw"])
        size = int(row["access_size_bytes"])
        grouped[key][size].append(row)
    return grouped


def svg_line(x1, y1, x2, y2, color="#333", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
    )


def svg_text(x, y, text, size=14, color="#111", anchor="start", weight="normal",
             rotate=None):
    rotate_attr = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{rotate_attr}>'
        f'{escape(str(text))}</text>'
    )


def x_position(burst):
    domain_min = math.log2(BURST_VALUES[0])
    domain_max = math.log2(BURST_VALUES[-1])
    value = math.log2(burst)
    ratio = (value - domain_min) / (domain_max - domain_min)
    return MARGIN_LEFT + ratio * PLOT_WIDTH


def y_position(value, y_max):
    ratio = 0.0 if y_max <= 0 else value / y_max
    return MARGIN_TOP + PLOT_HEIGHT * (1.0 - ratio)


def legend_label(size_bytes):
    return f"{size_bytes // 1024}KB" if size_bytes >= 1024 else f"{size_bytes}B"


def build_chart_svg(medium, rw, size_rows, metric_key, metric_title, metric_unit):
    y_max = 0.0
    normalized = {}
    for size in sorted(size_rows):
        rows = sorted(size_rows[size], key=lambda item: int(item["burst_count"]))
        normalized[size] = rows
        for row in rows:
            y_max = max(y_max, float(row[metric_key]))
    if y_max <= 0:
        y_max = 1.0

    y_tick_count = 5
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fbf8ef" />',
        svg_text(WIDTH / 2, 30, f"{medium.upper()} {rw} {metric_title}",
                 size=24, anchor="middle", weight="bold"),
    ]

    x0 = MARGIN_LEFT
    y0 = MARGIN_TOP + PLOT_HEIGHT
    x1 = MARGIN_LEFT + PLOT_WIDTH
    y1 = MARGIN_TOP
    parts.append(svg_line(x0, y0, x1, y0, width=2))
    parts.append(svg_line(x0, y0, x0, y1, width=2))

    for burst in BURST_VALUES:
        x = x_position(burst)
        parts.append(svg_line(x, y0, x, y0 + 6, width=1))
        parts.append(svg_line(x, y0, x, y1, color="#d0d0d0", width=1, dash="3,4"))
        parts.append(svg_text(x, y0 + 24, burst, size=12, anchor="middle"))

    for tick in range(y_tick_count + 1):
        value = y_max * tick / y_tick_count
        y = y_position(value, y_max)
        parts.append(svg_line(x0 - 6, y, x0, y, width=1))
        if tick > 0:
            parts.append(svg_line(x0, y, x1, y, color="#d0d0d0", width=1, dash="3,4"))
        label = f"{value:.1f}" if value < 1000 else f"{value:.0f}"
        parts.append(svg_text(x0 - 10, y + 4, label, size=12, anchor="end"))

    parts.append(svg_text((x0 + x1) / 2, HEIGHT - 20, "Burst count",
                          size=15, anchor="middle", weight="bold"))
    parts.append(svg_text(24, (y0 + y1) / 2, metric_unit,
                          size=15, anchor="middle", weight="bold", rotate=-90))

    for index, size in enumerate(sorted(normalized)):
        rows = normalized[size]
        color = COLORS[index % len(COLORS)]
        points = []
        for row in rows:
            burst = int(row["burst_count"])
            value = float(row[metric_key])
            points.append((x_position(burst), y_position(value, y_max)))

        polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'points="{polyline_points}" />'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />'
            )

        legend_x = x1 + 24
        legend_y = MARGIN_TOP + 20 + index * 24
        parts.append(svg_line(legend_x, legend_y - 5, legend_x + 18, legend_y - 5,
                              color=color, width=3))
        parts.append(svg_text(legend_x + 26, legend_y, legend_label(size), size=13))

    parts.append("</svg>")
    return "\n".join(parts)


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
        latency_svg = build_chart_svg(
            medium, rw, size_rows, "macro_avg_latency_ns", "Average Latency", "Latency (ns)"
        )
        bandwidth_svg = build_chart_svg(
            medium, rw, size_rows, "bandwidth_GBps", "Bandwidth", "Bandwidth (GB/s)"
        )

        with open(os.path.join(output_dir, f"{medium}_{rw}_latency.svg"), "w",
                  encoding="utf-8") as f:
            f.write(latency_svg)
        with open(os.path.join(output_dir, f"{medium}_{rw}_bandwidth.svg"), "w",
                  encoding="utf-8") as f:
            f.write(bandwidth_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
