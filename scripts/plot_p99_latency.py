#!/usr/bin/env python3
"""Plot the 910C HSTU-Large 8K M/M/1 P99 latency figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("configs/figure_data/p99_latency.csv")
DEFAULT_OUTPUT = Path("results/figures/P99Latency/P99Latency_910C_large_8K")
METHODS = ("RE", "CA", "O1", "O1+O2", "REFORGE")
COLORS = {
    "RE": "#777777",
    "CA": "#4C78A8",
    "O1": "#72A89A",
    "O1+O2": "#D28E5E",
    "REFORGE": "#9A75A3",
}
MARKERS = {"RE": "o", "CA": "s", "O1": "^", "O1+O2": "D", "REFORGE": "*"}
LINESTYLES = {
    "RE": (0, (2.2, 1.6)),
    "CA": (0, (4.0, 1.8)),
    "O1": (0, (5.0, 1.6, 1.2, 1.6)),
    "O1+O2": (0, (7.0, 1.8)),
    "REFORGE": "-",
}
Y_MAX_MS = 500.0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rates: list[float] = []
    values = {method: [] for method in METHODS}
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"request_rate", *METHODS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            rates.append(float(row["request_rate"]))
            for method in METHODS:
                values[method].append(float(row[method]))
    if not rates:
        raise ValueError("P99 input table is empty")
    rate_array = np.asarray(rates, dtype=float)
    if np.any(np.diff(rate_array) <= 0.0):
        raise ValueError("request_rate must be strictly increasing")
    return rate_array, {
        method: np.asarray(method_values, dtype=float)
        for method, method_values in values.items()
    }


def visible_prefix(values: np.ndarray) -> np.ndarray:
    """Keep real values through the first finite point above the display range."""

    exceed = np.flatnonzero(~np.isfinite(values) | (values > Y_MAX_MS))
    if not len(exceed):
        return values.copy()
    first = int(exceed[0])
    if np.isfinite(values[first]):
        return values[:first + 1].copy()
    return values[:first].copy()


def draw(
    rates: np.ndarray, values: dict[str, np.ndarray], output: Path
) -> None:
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "font.weight": "bold", "axes.labelsize": 8.3,
        "axes.labelweight": "bold", "axes.linewidth": 0.72,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.35, 2.05))
        for method in METHODS:
            shown = visible_prefix(values[method])
            shown_rates = rates[:len(shown)]
            axis.plot(
                shown_rates, shown, label=method, color=COLORS[method],
                linestyle=LINESTYLES[method], linewidth=1.35,
                marker=MARKERS[method], markersize=4.0,
                markerfacecolor="white", markeredgewidth=0.9, zorder=3,
            )

        axis.set_xlabel("Request Rate (query/s)")
        axis.set_ylabel("P99 Latency (ms)")
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, Y_MAX_MS)
        axis.set_xticks(np.arange(0.0, 101.0, 20.0))
        axis.set_xticklabels([str(value) for value in range(0, 101, 20)])
        axis.set_yticks(np.arange(0.0, Y_MAX_MS + 1.0, 100.0))
        axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=2.4, width=0.62)
        for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            label.set_fontweight("bold")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.7)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles, labels, ncol=5, loc="upper center",
            bbox_to_anchor=(0.54, 0.97), frameon=False,
            columnspacing=0.40, handlelength=1.12, handletextpad=0.22,
            prop={"size": 7.0, "weight": "bold"},
        )
        figure.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.84)

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
    draw(*load(args.input), args.output)
    print(f"wrote P99 PDF/SVG/PNG to {resolve(args.output).parent}")


if __name__ == "__main__":
    main()
