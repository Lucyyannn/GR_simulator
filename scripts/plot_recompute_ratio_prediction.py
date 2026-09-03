#!/usr/bin/env python3
"""Plot placeholder recompute-ratio data and the predicted ratio."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("configs/figure_data/recompute_ratio_prediction.csv")
DEFAULT_OUTPUT = Path(
    "results/figures/ItemKVRecomputeRatio/ItemKVRecomputeRatio"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def smooth_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving cubic Hermite interpolation through all data points."""

    spacing = np.diff(x)
    secant = np.diff(y) / spacing
    slope = np.zeros_like(y)
    for index in range(1, len(y) - 1):
        left, right = secant[index - 1], secant[index]
        if left * right > 0.0:
            wl = 2.0 * spacing[index] + spacing[index - 1]
            wr = spacing[index] + 2.0 * spacing[index - 1]
            slope[index] = (wl + wr) / (wl / left + wr / right)
    slope[0], slope[-1] = secant[0], secant[-1]

    dense_x = np.linspace(x[0], x[-1], 400)
    dense_y = np.empty_like(dense_x)
    for index in range(len(x) - 1):
        mask = (dense_x >= x[index]) & (dense_x <= x[index + 1])
        position = (dense_x[mask] - x[index]) / spacing[index]
        dense_y[mask] = (
            (2 * position**3 - 3 * position**2 + 1) * y[index]
            + (position**3 - 2 * position**2 + position) * spacing[index] * slope[index]
            + (-2 * position**3 + 3 * position**2) * y[index + 1]
            + (position**3 - position**2) * spacing[index] * slope[index + 1]
        )
    return dense_x, dense_y


def load(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grid = sorted(
        (row for row in rows if row["kind"] == "ratio_sweep"),
        key=lambda row: float(row["ratio"]),
    )
    prediction = [row for row in rows if row["kind"] == "predicted_ratio"]
    if len(grid) != 11 or len(prediction) != 1:
        raise ValueError("input must contain 11 ratio points and one prediction")
    return (
        np.asarray([float(row["ratio"]) for row in grid]),
        np.asarray([float(row["without_embedding_opt_qps"]) for row in grid]),
        np.asarray([float(row["with_embedding_opt_qps"]) for row in grid]),
        float(prediction[0]["ratio"]),
        float(prediction[0]["without_embedding_opt_qps"]),
        float(prediction[0]["with_embedding_opt_qps"]),
    )


def draw(input_path: Path, output_path: Path) -> None:
    (
        ratios, measured, optimized, predicted_ratio,
        predicted_measured_qps, predicted_optimized_qps,
    ) = load(input_path)
    if predicted_optimized_qps <= float(np.max(optimized)):
        raise ValueError(
            "predicted ratio must outperform all 11 embedding-optimized grid points"
        )
    fit_x = np.append(ratios, predicted_ratio)
    order = np.argsort(fit_x)
    fit_x = fit_x[order]
    measured_fit = np.append(measured, predicted_measured_qps)[order]
    optimized_fit = np.append(optimized, predicted_optimized_qps)[order]
    measured_x, measured_y = smooth_curve(fit_x, measured_fit)
    optimized_x, optimized_y = smooth_curve(fit_x, optimized_fit)
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.8,
        "font.weight": "bold", "axes.labelsize": 8.5,
        "axes.labelweight": "bold", "axes.linewidth": 0.75,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    }
    blue, orange, yellow = "#4E79A7", "#D8905F", "#E3B341"
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.55, 2.05))
        measured_line, = axis.plot(
            measured_x, measured_y, color=blue, linewidth=1.65,
            label="w/o Embedding Opt.", zorder=2,
        )
        axis.scatter(
            ratios, measured, s=20, facecolor="white", edgecolor=blue,
            linewidth=1.15, zorder=4,
        )
        optimized_line, = axis.plot(
            optimized_x, optimized_y, color=orange, linewidth=1.75,
            linestyle=(0, (4, 1.8)), label="w/ Embedding Opt.",
            zorder=3,
        )
        axis.scatter(
            ratios, optimized, marker="s", s=18, facecolor="white",
            edgecolor=orange, linewidth=1.1, zorder=4,
        )
        prediction = axis.scatter(
            [predicted_ratio], [predicted_optimized_qps], marker="*", s=85,
            facecolor=yellow, edgecolor="#5A4A16", linewidth=0.7,
            label="Cost-Model Prediction", zorder=5,
        )
        axis.axvline(
            predicted_ratio, color="#8A8A8A", linewidth=0.7,
            linestyle=(0, (2, 2)), zorder=1,
        )
        axis.set_xlabel("Item Recompute Ratio")
        axis.set_ylabel("QPS")
        axis.set_xlim(-0.025, 1.025)
        axis.set_xticks(np.arange(0.0, 1.01, 0.2))
        lower = np.floor(min(measured.min(), optimized.min()) / 50.0) * 50.0
        upper = np.ceil(max(measured.max(), optimized.max()) * 1.04 / 50.0) * 50.0
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=2.4, width=0.62)
        for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            label.set_fontweight("bold")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.7)
        figure.legend(
            [measured_line, optimized_line, prediction],
            ["w/o Embedding Opt.", "w/ Embedding Opt.", "Predicted Ratio"],
            ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
            frameon=False, handlelength=1.35, columnspacing=0.7,
            handletextpad=0.35, borderpad=0.1,
            prop={"size": 6.2, "weight": "bold"},
        )
        figure.subplots_adjust(left=0.15, right=0.985, bottom=0.22, top=0.84)
        output = resolve(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 500})):
            figure.savefig(
                output.with_suffix(f".{suffix}"), bbox_inches="tight",
                pad_inches=0.025, facecolor="white", **options,
            )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    draw(args.input, args.output)
    print(f"wrote PDF/SVG/PNG to {resolve(args.output).parent}")


if __name__ == "__main__":
    main()
