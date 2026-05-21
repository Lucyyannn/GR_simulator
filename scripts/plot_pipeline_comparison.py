#!/usr/bin/env python3
"""Compare HSTU pipeline timelines across recompute/cache schemes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_pipeline_timeline import load_plot_context, load_rows, render_segments


@dataclass(frozen=True)
class SchemeSpec:
    key: str
    label: str
    aliases: tuple[str, ...]


@dataclass
class ResourceSegment:
    lane: str
    start_us: float
    end_us: float
    layer_id: int | None

    @property
    def duration_us(self) -> float:
        return max(0.001, self.end_us - self.start_us)


@dataclass
class SchemeData:
    spec: SchemeSpec
    case_dir: Path
    segments: list[ResourceSegment]
    sim_time_us: float
    max_end_us: float


SCHEMES = (
    SchemeSpec("Full_Recompute", "Full recompute", ("Full_Recompute",)),
    SchemeSpec("Full_Cache", "Full cache", ("Full_Cache",)),
    SchemeSpec("w_AR", "w/ AR", ("w_AR",)),
    SchemeSpec("w_IR", "w/ IR", ("w_IR",)),
    SchemeSpec("w_both", "w/ both", ("w_both", "w_Both")),
)

NPU_OP_LANE = "NPU op span"
ALL_LANES = ("SSD", "DRAM", "HBM", NPU_OP_LANE)

RESOURCE_COLORS = {
    "SSD": "#0072B2",
    "DRAM": "#009E73",
    "HBM": "#D55E00",
    NPU_OP_LANE: "#E69F00",
}


def fnum(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def normalize_source_label(source: str) -> str:
    value = source.strip()
    lowered = value.lower()
    if lowered in {"dram", "ddr"}:
        return "DRAM"
    if lowered == "ssd":
        return "SSD"
    return value


def source_to_medium(source: str) -> str:
    label = normalize_source_label(source)
    if label == "DRAM":
        return "ddr"
    if label == "SSD":
        return "ssd"
    return label.lower()


def resolve_child(root: Path, requested: str) -> Path:
    exact = root / requested
    if exact.exists():
        return exact
    requested_folded = requested.casefold()
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == requested_folded:
                return child
    return exact


def resolve_scheme_dir(base_dir: Path, spec: SchemeSpec) -> Path:
    for alias in spec.aliases:
        path = base_dir / alias
        if path.exists():
            return path
    if base_dir.exists():
        aliases = {alias.casefold() for alias in spec.aliases}
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.casefold() in aliases:
                return child
    return base_dir / spec.key


def parse_hardware_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row.get("component") == "NPU" and row.get("scope") == "overall":
            return fnum(row.get("sim_time_us"))
    for row in rows:
        value = fnum(row.get("sim_time_us"))
        if value is not None:
            return value
    return None


def event_end_us(row: dict) -> float:
    end = fnum(row.get("end_us"), 0.0) or 0.0
    if end > 0:
        return end
    start = fnum(row.get("start_us"), 0.0) or 0.0
    duration = fnum(row.get("duration_us"), 0.0) or 0.0
    return start + duration


def parse_layer_id(row: dict) -> int | None:
    try:
        layer_id = int(float(row.get("layer_id", "")))
    except (TypeError, ValueError):
        return None
    if layer_id < 0:
        return None
    return layer_id


def ensure_context_source_defaults(context: dict, source_label: str) -> None:
    fallback = source_to_medium(source_label)
    for key in (
        "source_medium",
        "embedding_source_medium",
        "history_recompute_source_medium",
    ):
        if not context.get(key) or context.get(key) == "unknown":
            context[key] = fallback


def medium_to_lane(medium: str) -> str:
    value = (medium or "").strip().lower()
    if value == "ssd":
        return "SSD"
    if value in {"ddr", "dram"}:
        return "DRAM"
    if value == "hbm":
        return "HBM"
    return "DRAM"


def lanes_for_source(source_label: str) -> tuple[str, ...]:
    if normalize_source_label(source_label) == "SSD":
        return ("SSD", "HBM", NPU_OP_LANE)
    return ("DRAM", "HBM", NPU_OP_LANE)


def preload_phase_medium(phase: str, context: dict) -> str:
    if phase == "candidate_embedding":
        return context.get("embedding_source_medium") or context.get("source_medium")
    if phase == "history_embedding":
        return context.get("history_recompute_source_medium") or context.get(
            "source_medium"
        )
    return context.get("source_medium") or "ddr"


def resource_segments(rows: list[dict], context: dict) -> list[ResourceSegment]:
    segments: list[ResourceSegment] = []
    for row in rows:
        pipe = row.get("pipe", "")
        phase = row.get("phase", "")
        duration = fnum(row.get("duration_us"), 0.0) or 0.0
        if duration < 0:
            continue

        layer_id = parse_layer_id(row)
        if pipe == "preload":
            if phase == "stage":
                continue
            for start_us, duration_us, rendered_phase in render_segments(row, context):
                lane = medium_to_lane(preload_phase_medium(rendered_phase, context))
                segments.append(
                    ResourceSegment(lane, start_us, start_us + duration_us, layer_id)
                )
        elif pipe == "compute" and phase == "movin":
            start_us = fnum(row.get("start_us"), 0.0) or 0.0
            segments.append(
                ResourceSegment("HBM", start_us, event_end_us(row), layer_id)
            )
        elif pipe == "compute" and phase == "op":
            if (
                row.get("name") == "aten::embedding"
                and (fnum(row.get("duration_us"), 0.0) or 0.0) < 0.01
            ):
                continue
            start_us = fnum(row.get("start_us"), 0.0) or 0.0
            segments.append(
                ResourceSegment(NPU_OP_LANE, start_us, event_end_us(row), layer_id)
            )
    return merge_segments(segments)


def merge_segments(segments: list[ResourceSegment], gap_us: float = 0.002) -> list[ResourceSegment]:
    merged: list[ResourceSegment] = []
    for lane in ALL_LANES:
        layer_ids = sorted(
            {
                segment.layer_id
                for segment in segments
                if segment.lane == lane and segment.layer_id is not None
            }
        )
        if any(segment.lane == lane and segment.layer_id is None for segment in segments):
            layer_ids.append(None)
        for layer_id in layer_ids:
            lane_segments = sorted(
                (
                    segment
                    for segment in segments
                    if segment.lane == lane and segment.layer_id == layer_id
                ),
                key=lambda segment: (segment.start_us, segment.end_us),
            )
            if not lane_segments:
                continue
            current = lane_segments[0]
            for segment in lane_segments[1:]:
                if segment.start_us <= current.end_us + gap_us:
                    current.end_us = max(current.end_us, segment.end_us)
                else:
                    merged.append(current)
                    current = segment
            merged.append(current)
    return merged


def shaded_resource_color(lane: str, layer_id: int | None) -> str:
    base_color = RESOURCE_COLORS.get(lane, "#888888")
    if layer_id is None:
        return base_color

    rgb = mcolors.to_rgb(base_color)
    shade_steps = (-0.18, -0.06, 0.08, 0.20, 0.32)
    shade = shade_steps[layer_id % len(shade_steps)]
    if shade < 0:
        adjusted = tuple(channel * (1.0 + shade) for channel in rgb)
    else:
        adjusted = tuple(channel + (1.0 - channel) * shade for channel in rgb)
    return mcolors.to_hex(adjusted)


def filter_segments_for_lanes(
    segments: list[ResourceSegment], lanes: tuple[str, ...]
) -> list[ResourceSegment]:
    allowed = set(lanes)
    return [segment for segment in segments if segment.lane in allowed]


def scheme_origin_us(scheme: SchemeData, lanes: tuple[str, ...]) -> float:
    segments = filter_segments_for_lanes(scheme.segments, lanes)
    if not segments:
        return 0.0
    return min(segment.start_us for segment in segments)


def shifted_max_end_us(data: list[SchemeData], lanes: tuple[str, ...]) -> float:
    max_end_us = 0.0
    for scheme in data:
        origin_us = scheme_origin_us(scheme, lanes)
        lane_segments = sorted(
            filter_segments_for_lanes(scheme.segments, lanes),
            key=lambda segment: segment.end_us,
        )
        if lane_segments:
            max_end_us = max(max_end_us, lane_segments[-1].end_us - origin_us)
        max_end_us = max(max_end_us, scheme.sim_time_us - origin_us)
    return max_end_us


def load_scheme_data(base_dir: Path, spec: SchemeSpec, source_label: str) -> SchemeData:
    case_dir = resolve_scheme_dir(base_dir, spec)
    breakdown = case_dir / "layer_breakdown.csv"
    hardware = case_dir / "hardware_summary.csv"
    if not breakdown.exists():
        raise SystemExit(f"missing pipeline CSV for {spec.key}: {breakdown}")

    rows = load_rows(breakdown)
    if not rows:
        raise SystemExit(f"empty pipeline CSV for {spec.key}: {breakdown}")

    context = load_plot_context(breakdown)
    ensure_context_source_defaults(context, source_label)
    segments = resource_segments(rows, context)
    if not segments:
        raise SystemExit(f"no resource pipeline events found for {spec.key}: {breakdown}")

    max_end_us = max(max(event_end_us(row) for row in rows), max(s.end_us for s in segments))
    sim_time_us = parse_hardware_summary(hardware) or max_end_us
    return SchemeData(spec, case_dir, segments, sim_time_us, max_end_us)


def resolve_case_base(result_root: Path, model_size: str, source: str, chip: str) -> Path:
    cases = result_root / "cases"
    size_dir = resolve_child(cases, model_size)
    source_dir = resolve_child(size_dir, normalize_source_label(source))
    chip_dir = resolve_child(source_dir, chip)
    return chip_dir


def default_output_path(
    result_root: Path, model_size: str, source: str, chip: str
) -> Path:
    filename = f"{model_size}_{normalize_source_label(source)}_{chip}_resources.png"
    return result_root / "pipeline_compare" / filename


def format_us(value: float) -> str:
    if value >= 10_000:
        return f"{value / 1000.0:.2f} ms"
    if value >= 100:
        return f"{value:.1f} us"
    return f"{value:.2f} us"


def plot_scheme_axis(
    ax,
    scheme: SchemeData,
    x_max: float,
    baseline_us: float,
    lanes: tuple[str, ...],
) -> None:
    lane_positions = {lane: index for index, lane in enumerate(lanes)}
    origin_us = scheme_origin_us(scheme, lanes)
    for segment in filter_segments_for_lanes(scheme.segments, lanes):
        lane = lane_positions.get(segment.lane)
        if lane is None:
            continue
        ax.broken_barh(
            [(segment.start_us - origin_us, segment.duration_us)],
            (lane - 0.32, 0.64),
            facecolors=shaded_resource_color(segment.lane, segment.layer_id),
            edgecolors="white",
            linewidth=0.16,
            alpha=0.92,
        )

    ax.axvline(
        max(0.0, scheme.sim_time_us - origin_us),
        color="#222222",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.65, len(lanes) - 0.35)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=8)
    speedup = baseline_us / scheme.sim_time_us if scheme.sim_time_us > 0 else 0.0
    origin_note = ""
    if origin_us > 1.0:
        origin_note = f"  |  origin +{format_us(origin_us)}"
    ax.set_title(
        f"{scheme.spec.label}  |  {format_us(scheme.sim_time_us)}"
        f"  |  {speedup:.2f}x vs Full recompute{origin_note}",
        loc="left",
        fontsize=9,
        pad=3,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def plot_comparison(
    data: list[SchemeData],
    output_path: Path,
    title: str,
    dpi: int,
    source_label: str,
) -> None:
    baseline = next(
        (scheme for scheme in data if scheme.spec.key == "Full_Recompute"), data[0]
    )
    baseline_us = baseline.sim_time_us
    lanes = lanes_for_source(source_label)
    x_max = shifted_max_end_us(data, lanes)
    if x_max <= 0:
        raise SystemExit("cannot plot comparison with non-positive timeline range")

    fig_height = min(8.2, max(5.8, 1.0 + len(data) * 1.12))
    fig = plt.figure(figsize=(13.2, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        len(data),
        1,
        hspace=0.18,
    )

    axes = []
    for index, scheme in enumerate(data):
        ax = fig.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        plot_scheme_axis(ax, scheme, x_max, baseline_us, lanes)
        if index != len(data) - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)

    axes[-1].set_xlabel("simulation time (us)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    handles = [
        Patch(facecolor=RESOURCE_COLORS[lane], edgecolor="none", label=lane)
        for lane in lanes
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(lanes),
        frameon=False,
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one shared-axis resource timeline comparing Full_Recompute, "
            "Full_Cache, w_AR, w_IR, and w_both pipeline results."
        )
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--source", required=True, help="DRAM/DDR or SSD")
    parser.add_argument("--chip", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root
    source_label = normalize_source_label(args.source)
    case_base = resolve_case_base(
        result_root, args.model_size, source_label, args.chip
    )
    if not case_base.exists():
        raise SystemExit(f"case directory not found: {case_base}")

    data = [
        load_scheme_data(case_base, spec, source_label)
        for spec in SCHEMES
    ]
    output = args.output or default_output_path(
        result_root, args.model_size, source_label, args.chip
    )
    title = args.title or (
        f"Pipeline resource comparison: {args.model_size} / "
        f"{source_label} / {args.chip}"
    )
    plot_comparison(data, output, title, args.dpi, source_label)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
