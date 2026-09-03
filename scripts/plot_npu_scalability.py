#!/usr/bin/env python3
"""Plot HSTU-Large NPU scalability directly from a complete QPS table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "configs/figure_data/npu_scalability.csv"
)
DEFAULT_OUTPUT = Path("results/figures/NPUScalability/NPUScalability")
CHIPS = ("910A", "910B", "910C", "MTIA2")
SEQUENCES = (4096, 6144, 8192)
BATCHES = (1, 2, 4)
METHODS = ("RE", "CA", "REFORGE")
COLORS = {"RE": "#A8A8A8", "CA": "#6B9AC4", "REFORGE": "#A07AA8"}
HATCHES = {"RE": "..", "CA": "///", "REFORGE": "++"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Complete CSV with chip, model, seq_len, batch_size, RE, CA, REFORGE.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_qps(path: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    """Load values without fitting, scaling, interpolation, or normalization."""

    values: dict[tuple[str, int, int], dict[str, float]] = {}
    required_columns = {
        "chip", "model", "seq_len", "batch_size", *METHODS,
    }
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "missing CSV columns: " + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            if row["model"] != "large":
                continue
            key = (row["chip"], int(row["seq_len"]), int(row["batch_size"]))
            if key in values:
                raise ValueError(f"duplicate workload row: {key}")
            values[key] = {method: float(row[method]) for method in METHODS}

    expected = {
        (chip, sequence, batch)
        for chip in CHIPS for sequence in SEQUENCES for batch in BATCHES
    }
    missing_rows = expected.difference(values)
    extra_rows = set(values).difference(expected)
    if missing_rows:
        raise ValueError(f"missing workload rows: {sorted(missing_rows)}")
    if extra_rows:
        raise ValueError(f"unexpected workload rows: {sorted(extra_rows)}")
    for key, methods in values.items():
        for method, qps in methods.items():
            if not math.isfinite(qps) or qps <= 0.0:
                raise ValueError(f"invalid {method} QPS for {key}: {qps}")
    return values


def nice_qps_axis(maximum: float) -> tuple[float, float]:
    rough_step = maximum / 4.0
    magnitude = 10 ** math.floor(math.log10(rough_step))
    step = next(
        unit * magnitude for unit in (1, 2, 5, 10)
        if unit * magnitude >= rough_step
    )
    upper = math.ceil(maximum * 1.10 / step) * step
    return upper, step


def draw_batch(
    qps: dict[tuple[str, int, int], dict[str, float]],
    batch: int,
    output: Path,
) -> None:
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "font.weight": "bold", "axes.labelsize": 8.2,
        "axes.labelweight": "bold", "axes.linewidth": 0.75,
        "axes.titlesize": 7.2, "axes.titleweight": "bold",
        "xtick.labelsize": 5.8, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "hatch.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    x = np.arange(len(CHIPS), dtype=float)
    width = 0.235
    maximum = max(
        qps[(chip, sequence, batch)][method]
        for chip in CHIPS for sequence in SEQUENCES for method in METHODS
    )
    if batch == 1:
        upper, step = 200.0, 50.0
        if maximum > upper:
            raise ValueError(
                f"bs=1 QPS exceeds the fixed 200-QPS axis limit: {maximum}"
            )
    else:
        upper, step = nice_qps_axis(maximum)

    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(3.35, 1.78), sharey=True)
        legend_handles = []
        for sequence_index, (axis, sequence) in enumerate(zip(axes, SEQUENCES)):
            for offset, method in zip((-width, 0.0, width), METHODS):
                bars = axis.bar(
                    x + offset,
                    [qps[(chip, sequence, batch)][method] for chip in CHIPS],
                    width=width, color=COLORS[method], edgecolor="#424242",
                    linewidth=0.52, hatch=HATCHES[method], label=method, zorder=3,
                )
                if sequence_index == 0:
                    legend_handles.append(bars)
            length_label = "6K" if sequence == 6144 else f"{sequence // 1024}K"
            axis.set_title(f"length={length_label}", pad=3.0)
            axis.set_xticks(
                x, CHIPS, rotation=35, ha="right", rotation_mode="anchor"
            )
            axis.set_xlim(-0.55, len(CHIPS) - 0.45)
            axis.set_ylim(0.0, upper)
            axis.set_yticks(np.arange(0.0, upper + step * 0.5, step))
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.48, alpha=0.9)
            axis.set_axisbelow(True)
            axis.tick_params(direction="out", length=2.2, width=0.6, pad=1.5)
            for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                label.set_fontweight("bold")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#555555")
                spine.set_linewidth(0.68)
        axes[0].set_ylabel("QPS")
        figure.legend(
            legend_handles, METHODS, ncol=3, loc="upper center",
            bbox_to_anchor=(0.5, 0.965), frameon=False,
            columnspacing=0.8, handlelength=1.25, handletextpad=0.35,
            prop={"size": 6.5, "weight": "bold"},
        )
        figure.subplots_adjust(
            left=0.14, right=0.992, bottom=0.25, top=0.77, wspace=0.10
        )

        output = resolve(output).with_name(f"{resolve(output).name}_b{batch}")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 500})):
            figure.savefig(
                output.with_suffix(f".{suffix}"), bbox_inches="tight",
                pad_inches=0.025, facecolor="white", **options,
            )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    qps = load_qps(args.input)
    for batch in BATCHES:
        draw_batch(qps, batch, args.output)
    print(f"wrote NPU scalability PDF/SVG/PNG to {resolve(args.output).parent}")


if __name__ == "__main__":
    main()
