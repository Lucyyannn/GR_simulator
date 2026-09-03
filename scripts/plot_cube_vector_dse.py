#!/usr/bin/env python3
"""Analytical Cube/Vector DSE for the HSTU resource-allocation study.

The script deliberately uses transparent workload equations rather than
simulator cycles.  It enumerates NPU configurations near a fixed compute-area
budget and plots latency against the paper's Cube/Vector pressure ratio R.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CUBE_AREA_MM2 = 2.57                 # One 8-TFLOPS reference Cube core.
VECTOR_AREA_2048B_MM2 = 0.70         # One 2048-bit Vector core.
CUBE_FLOPS_PER_CORE_CYCLE = 2 * 64 * 64
FP16_BITS = 16

# 910C analytical Vector primitive costs (cycles), matching configs/910C.json.
# These are architecture-level operation costs, not fitted workload factors.
VECTOR_ADD_CYCLES = 1
VECTOR_MUL_CYCLES = 1
VECTOR_SWISH_CYCLES = 8
VECTOR_DIV_CYCLES = 4
VECTOR_ADD_TREE_CYCLES = 7


@dataclass(frozen=True)
class Workload:
    layers: int = 4
    hidden: int = 256
    heads: int = 4
    batch: int = 4
    history_tokens: int = 8192
    candidate_tokens: int = 128


def hstu_compute(workload: Workload) -> dict[str, float]:
    """Return ordinary-HSTU Cube FLOPs and Vector-equivalent FLOPs.

    Cube work per layer:
      projection = (8 + 6) * tokens * H^2
      attention  = 4 * score_elements * H       (QK and AV)

    Vector latency-weighted work per layer:
      LayerNorm = (2*l_tree + 2*l_add + 3*l_mul) * tokens * H
      input path  = (LayerNorm + l_swish) * tokens * H
      output path = (LayerNorm + l_mul) * tokens * H
      attention   = (2*l_mul + l_swish + l_div) * heads * scores

    Vector scalar operations are multiplied by two so that they can be
    divided by the architecture's two-FLOP-per-FP16-lane throughput.
    """

    L, H, N = workload.layers, workload.hidden, workload.batch
    S, C = workload.history_tokens, workload.candidate_tokens
    tokens = N * (S + C)
    score_elements = N * (S * (S + 1) / 2 + C * S + C)

    cube_projection_flops = L * 14.0 * tokens * H * H
    cube_attention_flops = L * 4.0 * score_elements * H
    layernorm_weight = (
        2 * VECTOR_ADD_TREE_CYCLES
        + 2 * VECTOR_ADD_CYCLES
        + 3 * VECTOR_MUL_CYCLES
    )
    token_weight = (
        layernorm_weight + VECTOR_SWISH_CYCLES
        + layernorm_weight + VECTOR_MUL_CYCLES
    )
    attention_weight = (
        2 * VECTOR_MUL_CYCLES + VECTOR_SWISH_CYCLES + VECTOR_DIV_CYCLES
    )
    vector_token_ops = L * token_weight * tokens * H
    vector_attention_ops = L * attention_weight * workload.heads * score_elements
    vector_ops = vector_token_ops + vector_attention_ops

    return {
        "tokens": tokens,
        "score_elements": score_elements,
        "cube_projection_flops": cube_projection_flops,
        "cube_attention_flops": cube_attention_flops,
        "cube_flops": cube_projection_flops + cube_attention_flops,
        "vector_token_ops": vector_token_ops,
        "vector_attention_ops": vector_attention_ops,
        "vector_ops": vector_ops,
        "vector_equivalent_flops": 2.0 * vector_ops,
    }


def area_mm2(nc: int, nv: int, vector_width_bits: int) -> tuple[float, float, float]:
    cube = nc * CUBE_AREA_MM2
    vector = nv * VECTOR_AREA_2048B_MM2 * vector_width_bits / 2048.0
    return cube, vector, cube + vector


def evaluate_point(
    nc: int,
    nv: int,
    vector_width_bits: int,
    frequency_hz: float,
    compute: dict[str, float],
) -> dict[str, float]:
    cube_area, vector_area, total_area = area_mm2(nc, nv, vector_width_bits)
    cube_flops_per_cycle = nc * CUBE_FLOPS_PER_CORE_CYCLE
    vector_flops_per_cycle = nv * 2.0 * (vector_width_bits / FP16_BITS)
    cube_cycles = compute["cube_flops"] / cube_flops_per_cycle
    vector_cycles = compute["vector_equivalent_flops"] / vector_flops_per_cycle
    latency_ms = (cube_cycles + vector_cycles) / frequency_hz * 1e3
    pressure_ratio = (cube_cycles / cube_area) / (vector_cycles / vector_area)
    return {
        "nc": nc,
        "nv": nv,
        "vector_width_bits": vector_width_bits,
        "cube_area_mm2": cube_area,
        "vector_area_mm2": vector_area,
        "total_area_mm2": total_area,
        "cube_flops_per_cycle": cube_flops_per_cycle,
        "vector_flops_per_cycle": vector_flops_per_cycle,
        "cube_cycles": cube_cycles,
        "vector_cycles": vector_cycles,
        "R": pressure_ratio,
        "latency_ms": latency_ms,
    }


def lower_envelope(rows: list[dict[str, float]], bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["R"] for row in rows])
    y = np.asarray([row["normalized_latency"] for row in rows])
    edges = np.geomspace(x.min(), x.max(), bins + 1)
    centers, minima = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x < right)
        if mask.any():
            index = np.flatnonzero(mask)[np.argmin(y[mask])]
            centers.append(x[index])
            minima.append(y[index])
    return np.asarray(centers), np.asarray(minima)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/cube_vector_dse"))
    parser.add_argument("--area-min-ratio", type=float, default=0.99)
    parser.add_argument("--area-max-ratio", type=float, default=1.00)
    parser.add_argument("--frequency-ghz", type=float, default=1.8)
    parser.add_argument("--nc-min", type=int, default=1)
    parser.add_argument("--nc-max", type=int, default=64)
    parser.add_argument("--nv-min", type=int, default=1)
    parser.add_argument("--nv-max", type=int, default=128)
    parser.add_argument("--width-min", type=int, default=1024)
    parser.add_argument("--width-max", type=int, default=8192)
    parser.add_argument("--width-step", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = Workload()
    compute = hstu_compute(workload)
    frequency_hz = args.frequency_ghz * 1e9

    baseline = evaluate_point(48, 48, 4096, frequency_hz, compute)
    area_low = args.area_min_ratio * baseline["total_area_mm2"]
    area_high = args.area_max_ratio * baseline["total_area_mm2"]

    rows: list[dict[str, float]] = []
    for nc in range(args.nc_min, args.nc_max + 1):
        for nv in range(args.nv_min, args.nv_max + 1):
            for width in range(args.width_min, args.width_max + 1, args.width_step):
                row = evaluate_point(nc, nv, width, frequency_hz, compute)
                if area_low <= row["total_area_mm2"] <= area_high:
                    row["normalized_latency"] = row["latency_ms"] / baseline["latency_ms"]
                    rows.append(row)
    if not rows:
        raise RuntimeError("No feasible points in the requested area window")

    baseline["normalized_latency"] = 1.0
    optimum = min(rows, key=lambda row: row["latency_ms"])
    for row in rows:
        row["is_baseline"] = int(
            row["nc"] == 48 and row["nv"] == 48 and row["vector_width_bits"] == 4096
        )
        row["is_optimal"] = int(row is optimum)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cube_vector_dse_910c_hstu_small_bs4_seq8k.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    x = np.asarray([row["R"] for row in rows])
    y = np.asarray([row["normalized_latency"] for row in rows])
    ax.scatter(x, y, s=12, facecolors="white", edgecolors="#5B83B4", linewidths=0.65,
               alpha=0.70, label="Feasible Designs", zorder=1)
    env_x, env_y = lower_envelope(rows)
    ax.plot(env_x, env_y, color="#4C78A8", linewidth=1.8, label="Lower envelope", zorder=2)
    ax.scatter(baseline["R"], 1.0, marker="D", s=48, facecolor="#8B78A8",
               edgecolor="#4D3C68", linewidth=0.85, label="Baseline NPU", zorder=4)
    ax.scatter(optimum["R"], optimum["normalized_latency"], marker="*", s=125,
               facecolor="#F2C14E", edgecolor="#7A5A00", linewidth=0.7,
               label="Optimal configuration", zorder=5)
    ax.axvline(1.0, color="#B8B8B8", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cube-to-Vector pressure ratio $R$")
    ax.set_ylabel("Normalized analytical latency")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.65)
    ax.tick_params(direction="out", length=3, width=0.7)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#6F6F6F")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.0, handletextpad=0.45)
    fig.tight_layout(pad=0.7)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"cube_vector_dse_910c_hstu_small_bs4_seq8k.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Linear paper view: retain the performance-relevant neighborhood while
    # leaving the complete, many-orders-of-magnitude DSE in the log plot.
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    visible = [row for row in rows if row["R"] <= 3.0 and row["normalized_latency"] <= 1.5]
    vx = np.asarray([row["R"] for row in visible])
    vy = np.asarray([row["normalized_latency"] for row in visible])
    ax.scatter(vx, vy, s=13, facecolors="white", edgecolors="#5B83B4", linewidths=0.65,
               alpha=0.70, label="Feasible Designs", zorder=1)
    ax.scatter(baseline["R"], 1.0, marker="D", s=48, facecolor="#8B78A8",
               edgecolor="#4D3C68", linewidth=0.85, label="Baseline NPU", zorder=4)
    ax.scatter(optimum["R"], optimum["normalized_latency"], marker="*", s=115,
               facecolor="#F2C14E", edgecolor="#7A5A00", linewidth=0.7,
               label="Optimal Design", zorder=5)
    ax.axvline(1.0, color="#B8B8B8", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.90, 1.5)
    ax.set_xlabel(r"$R$", fontsize=12)
    ax.set_ylabel("Normalized Latency", fontsize=11)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.65)
    ax.tick_params(direction="out", length=3, width=0.7, labelsize=9.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#6F6F6F")
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(0.985, 0.995),
              handletextpad=0.45, labelspacing=0.3, fontsize=8.1,
              borderaxespad=0.25)
    fig.tight_layout(pad=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.output_dir / f"cube_vector_dse_910c_hstu_small_bs4_seq8k_linear.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    print(f"Feasible configurations: {len(rows)}")
    print(f"HSTU Cube work: {compute['cube_flops'] / 1e12:.6f} TFLOP")
    print(f"HSTU Vector work: {compute['vector_ops'] / 1e9:.6f} Gop")
    print(
        "Baseline: Nc=48 Nv=48 Wv=4096, "
        f"R={baseline['R']:.6f}, latency={baseline['latency_ms']:.6f} ms"
    )
    print(
        f"Optimal: Nc={optimum['nc']} Nv={optimum['nv']} "
        f"Wv={optimum['vector_width_bits']}, area={optimum['total_area_mm2']:.3f} mm^2, "
        f"R={optimum['R']:.6f}, latency={optimum['latency_ms']:.6f} ms, "
        f"speedup={1.0 / optimum['normalized_latency']:.6f}x"
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
