#!/usr/bin/env python3
"""Plot the paper's per-chip SystemThroughput comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "configs/figure_data/system_throughput.csv"
)
DEFAULT_OUTPUT_ROOT = Path("results/figures/SystemThroughput")
CHIPS = ("910A", "910B", "910C", "MTIA2")
MODELS = ("small", "middle", "large")
MODEL_LABELS = {
    "small": "HSTU-Small", "middle": "HSTU-Middle", "large": "HSTU-Large",
}
SEQUENCES = (4096, 6144, 8192)
BATCHES = (1, 2, 4)
METHODS = ("RE", "CA", "O1", "O1+O2", "REFORGE")
METHOD_LABELS = {
    "RE": "RE",
    "CA": "CA",
    "O1": "O1",
    "O1+O2": "O1+O2",
    "REFORGE": "REFORGE",
}
COLORS = {
    "RE": "#A8A8A8", "CA": "#6B9AC4", "O1": "#67A99A",
    "O1+O2": "#D8905F", "REFORGE": "#A07AA8",
}
HATCHES = {
    "RE": "..", "CA": "///", "O1": "--", "O1+O2": "xx",
    "REFORGE": "++",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--chips", nargs="+", choices=CHIPS, default=list(CHIPS),
        help="Chips to plot; defaults to all four.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load(path: Path) -> dict[tuple, dict[str, float]]:
    """Load complete plot values without fitting or transforming them."""

    values: dict[tuple, dict[str, float]] = {}
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "chip", "model", "seq_len", "batch_size", *METHODS,
            "reforge_qps",
        }
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "missing CSV columns: " + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            workload = (
                str(row["chip"]), str(row["model"]), int(row["seq_len"]),
                int(row["batch_size"]),
            )
            if workload in values:
                raise ValueError(f"duplicate workload row: {workload}")
            entry = {method: float(row[method]) for method in METHODS}
            entry["reforge_qps"] = float(row["reforge_qps"])
            values[workload] = entry

    expected = {
        (chip, model, sequence, batch)
        for chip in CHIPS for model in MODELS
        for sequence in SEQUENCES for batch in BATCHES
    }
    missing_rows = expected.difference(values)
    extra_rows = set(values).difference(expected)
    if missing_rows:
        raise ValueError(f"missing workload rows: {sorted(missing_rows)}")
    if extra_rows:
        raise ValueError(f"unexpected workload rows: {sorted(extra_rows)}")
    for workload, entry in values.items():
        for field, value in entry.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid {field} for {workload}: {value}")
    return values


def nice_upper(values: list[float]) -> float:
    maximum = max([1.0, *values]) * 1.10
    if maximum <= 3.0:
        return math.ceil(maximum * 5.0) / 5.0
    if maximum <= 6.0:
        return math.ceil(maximum * 2.0) / 2.0
    return float(math.ceil(maximum))


def draw_chip(values: dict[tuple, dict[str, float]], chip: str, output: Path) -> None:
    style = {
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "font.weight": "bold",
        "axes.labelsize": 8.8, "axes.titlesize": 9.2,
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.linewidth": 0.72, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "hatch.linewidth": 0.85, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    }
    positions = np.arange(9, dtype=float)
    positions += np.repeat((0.0, 0.36, 0.72), 3)
    width = 0.135
    offsets = tuple((index - 2) * width for index in range(5))

    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.28))
        legend_handles = []
        for model_index, (axis, model) in enumerate(zip(axes, MODELS)):
            finite_values = [
                values[(chip, model, sequence, batch)][method]
                for sequence in SEQUENCES for batch in BATCHES
                if (chip, model, sequence, batch) in values
                for method in METHODS
                if method in values[(chip, model, sequence, batch)]
            ]
            upper = nice_upper(finite_values)
            for method_index, method in enumerate(METHODS):
                heights = np.asarray([
                    values.get((chip, model, sequence, batch), {})
                    .get(method, math.nan)
                    for sequence in SEQUENCES for batch in BATCHES
                ])
                bars = axis.bar(
                    positions + offsets[method_index], heights, width=width,
                    color=COLORS[method], edgecolor="#424242", linewidth=0.56,
                    hatch=HATCHES[method], zorder=3, label=METHOD_LABELS[method],
                )
                if model_index == 0:
                    legend_handles.append(bars)
            qps = np.asarray([
                values.get((chip, model, sequence, batch), {})
                .get("reforge_qps", math.nan)
                for sequence in SEQUENCES for batch in BATCHES
            ])
            qps_mask = np.isfinite(qps)
            finite_qps = qps[qps_mask]
            qps_axis = axis.twinx()
            qps_axis.scatter(
                positions[qps_mask], qps[qps_mask],
                marker="D", s=17,
                facecolor="white", edgecolor="#C44E52", linewidth=0.9,
                zorder=5,
            )
            if len(finite_qps):
                qps_max = float(np.max(finite_qps))
                tick_step = max(100, math.ceil(qps_max / 400) * 100)
                qps_upper = math.ceil(qps_max / tick_step) * tick_step
                if qps_max > 0.92 * qps_upper:
                    qps_upper += tick_step
                qps_axis.set_ylim(0.0, qps_upper)
                qps_axis.set_yticks(np.arange(0, qps_upper + tick_step, tick_step))
            qps_axis.set_ylabel("QPS" if model_index == 2 else "")
            qps_axis.tick_params(
                axis="y", direction="out", length=2.3, width=0.6,
                colors="#555555", labelsize=6.2, pad=1.5,
            )
            qps_axis.spines["right"].set_color("#555555")
            qps_axis.spines["right"].set_linewidth(0.7)
            qps_axis.spines["top"].set_visible(False)
            qps_axis.spines["left"].set_visible(False)
            qps_axis.spines["bottom"].set_visible(False)
            for label in qps_axis.get_yticklabels():
                label.set_fontweight("heavy")
                label.set_fontsize(6.6)
                label.set_color("#333333")
            axis.axhline(
                1.0, color="#707070", linewidth=0.78,
                linestyle=(0, (3.0, 2.2)), zorder=2,
            )
            for boundary in (
                (positions[2] + positions[3]) / 2,
                (positions[5] + positions[6]) / 2,
            ):
                axis.axvline(
                    boundary, color="#BEBEBE", linewidth=0.7,
                    linestyle="-", zorder=1,
                )
            axis.set_xticks(
                positions,
                [f"b{batch}" for _ in SEQUENCES for batch in BATCHES],
            )
            for group, sequence in enumerate(SEQUENCES):
                group_x = float(np.mean(positions[group * 3:(group + 1) * 3]))
                axis.text(
                    group_x, -0.205,
                    "length=6K" if sequence == 6144
                    else f"length={sequence // 1024}K",
                    transform=axis.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    fontweight="bold",
                )
            axis.set_ylim(0.0, upper * 1.12)
            axis.set_xlim(positions[0] - 0.50, positions[-1] + 0.58)
            axis.set_ylabel(
                r"Normalized Speedup ($\times$)" if model_index == 0 else ""
            )
            axis.set_title(MODEL_LABELS[model], pad=5)
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.9)
            axis.set_axisbelow(True)
            axis.tick_params(direction="out", length=2.5, width=0.62)
            for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                label.set_fontweight("bold")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#555555")
                spine.set_linewidth(0.7)

        qps_handle = Line2D(
            [], [], marker="D", linestyle="None", markersize=4.4,
            markerfacecolor="white", markeredgecolor="#C44E52",
            markeredgewidth=0.9,
        )
        figure.legend(
            [*legend_handles, qps_handle],
            [
                *[METHOD_LABELS[method] for method in METHODS],
                "Throughput of REFORGE (query per second)",
            ],
            ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.005),
            frameon=False, handlelength=1.2, columnspacing=0.9,
            handletextpad=0.4, prop={"size": 7.2, "weight": "bold"},
        )
        figure.subplots_adjust(
            left=0.06, right=0.975, bottom=0.22, top=0.79, wspace=0.20,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 500})):
            figure.savefig(
                output.with_suffix(f".{suffix}"), bbox_inches="tight",
                pad_inches=0.025, facecolor="white", **options,
            )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    values = load(args.input)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for chip in args.chips:
        draw_chip(
            values, chip,
            output_root / f"SystemThroughput_{chip}",
        )
    print(f"wrote {len(args.chips)} per-chip figures to {output_root}")


if __name__ == "__main__":
    main()
