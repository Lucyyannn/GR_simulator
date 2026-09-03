#!/usr/bin/env python3
"""Plot the paper's IKMHitRate QPS and DRAM-hit-rate comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("results/analysis/user_activity_placement/summary.csv")
DEFAULT_OUTPUT = Path("results/figures/IKMHitRate/IKMHitRate")
METHODS = (
    ("re_qps", "re_dram_hit_rate", "RE"),
    ("ca_qps", "ca_dram_hit_rate", "CA"),
    ("reforge_random_qps", "reforge_random_dram_hit_rate", "w/o IKM"),
    ("reforge_aa_ir_qps", "reforge_aa_ir_dram_hit_rate", "w/ IKM"),
)
COLORS = ("#A8A8A8", "#6B9AC4", "#D8905F", "#67A99A")
HATCHES = ("..", "///", "xx", "--")
HIT_COLORS = ("#7851A9", "#D1495B", "#3977B7", "#2A9D8F")
MARKERS = ("o", "s", "D", "s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load(path: Path) -> list[dict[str, float]]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("placement summary is empty")
    numeric = []
    for row in rows:
        converted = {"users": int(row["users"])}
        for field, _, _ in METHODS:
            converted[field] = float(row[field])
        for _, field, _ in METHODS:
            converted[field] = float(row[field])
        if any(not math.isfinite(value) for value in converted.values()):
            raise ValueError(f"non-finite value in row: {row}")
        numeric.append(converted)
    return sorted(numeric, key=lambda row: row["users"])


def draw(rows: list[dict[str, float]], output: Path) -> None:
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "font.weight": "bold", "axes.labelsize": 8.2,
        "axes.labelweight": "bold", "axes.titlesize": 8.4,
        "axes.titleweight": "bold", "axes.linewidth": 0.72,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 8.0, "hatch.linewidth": 0.85,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    x = np.arange(len(rows), dtype=float)
    width = 0.19
    labels = [
        f"{row['users'] // 1000}K"
        for row in rows
    ]
    with plt.rc_context(style):
        figure, qps_ax = plt.subplots(figsize=(4.18, 2.18))
        hit_ax = qps_ax.twinx()
        qps_handles = []
        hit_handles = []
        for index, (
            (qps_field, hit_field, label), color, hatch, hit_color, marker,
        ) in enumerate(
            zip(METHODS, COLORS, HATCHES, HIT_COLORS, MARKERS)
        ):
            offset_x = x + (index - 1.5) * width
            bars = qps_ax.bar(
                offset_x, [row[qps_field] for row in rows], width=width,
                color=color, edgecolor="#424242", linewidth=0.55,
                hatch=hatch, zorder=3, label=label,
            )
            qps_handles.append(bars)
            hit_values = [100.0 * row[hit_field] for row in rows]
            hit_line, = hit_ax.plot(
                x, hit_values,
                color=hit_color, marker=marker, markersize=6.5,
                markerfacecolor=hit_color, markeredgecolor="white",
                markeredgewidth=0.9, linewidth=2.25,
                linestyle="-", zorder=5,
                clip_on=True,
            )
            hit_handles.append(hit_line)
        qps_ax.set_ylabel("QPS")
        qps_ax.set_xticks(x, labels)
        qps_ax.set_ylim(0.0, math.ceil(max(
            row[field] for row in rows for field, _, _ in METHODS
        ) * 1.12 / 10) * 10)
        hit_ax.set_ylabel("DRAM Hit Rate (%)")
        hit_ax.set_ylim(0.0, 35.0)
        hit_ax.set_yticks(np.arange(0.0, 35.1, 5.0))

        qps_ax.set_xlabel("Number of Users")
        qps_ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.9)
        qps_ax.set_axisbelow(True)
        for axis in (qps_ax, hit_ax):
            axis.tick_params(direction="out", length=2.4, width=0.62)
            for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                label.set_fontweight("bold")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#555555")
                spine.set_linewidth(0.7)

        legend_labels = [label for _, _, label in METHODS]
        blank = Line2D([], [], linestyle="None", alpha=0.0)
        # Matplotlib fills multi-row legends column by column. Interleaving
        # entries keeps QPS and Hit perfectly aligned in two rows.
        combined_handles = [blank, blank]
        combined_labels = ["QPS", "Hit"]
        for qps_handle, hit_handle, label in zip(
            qps_handles, hit_handles, legend_labels
        ):
            combined_handles.extend((qps_handle, hit_handle))
            combined_labels.extend((label, label))
        figure.legend(
            combined_handles, combined_labels, ncol=5,
            loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False,
            columnspacing=0.72, handlelength=1.2, handletextpad=0.3,
            labelspacing=0.35,
            prop={"size": 7.6, "weight": "bold"},
        )
        figure.subplots_adjust(
            left=0.125, right=0.87, bottom=0.22, top=0.78
        )
        output = resolve(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 500})):
            figure.savefig(
                output.with_suffix(f".{suffix}"), bbox_inches="tight",
                pad_inches=0.025, facecolor="white", **options,
            )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load(args.input)
    draw(rows, args.output)
    print(f"wrote PDF/SVG/PNG to {resolve(args.output).parent}")


if __name__ == "__main__":
    main()
