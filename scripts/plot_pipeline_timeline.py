#!/usr/bin/env python3

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def macro_label(row):
    model = row.get("model", "")
    if model:
        return model
    batch_id = row.get("batch_id", "")
    macro_batch_id = row.get("macro_batch_id", "")
    if batch_id or macro_batch_id:
        return f"b{batch_id or '?'}_m{macro_batch_id or '?'}"
    return "macro"


def display_phase(row):
    phase = row.get("phase", "") or "other"
    if row.get("pipe") != "preload":
        return phase
    if phase in {"pre_attention", "candidate_embedding"}:
        return "candidate_embedding"
    if phase in {
        "history_sequence",
        "history_embedding",
        "history_recompute_embedding",
    }:
        return "history_embedding"
    if phase == "kvcache":
        return "kvcache"
    if phase in {
        "post_attention",
        "post_attention_weights",
        "post_attention_kvcache",
        "weights",
    }:
        return "post_attention"
    return "other_preload"


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_compute_intervals(csv_path):
    """Load instruction-level resource intervals emitted by the simulator."""
    interval_path = Path(csv_path).with_name("compute_activity_intervals.csv")
    if not interval_path.exists():
        return []
    try:
        with interval_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_plot_context(csv_path):
    path = Path(csv_path)
    context = {
        "candidate_tokens": 128,
        "precision_bytes": 2,
        "models": {},
        "hidden": infer_hidden_from_path(path),
        "batch_size": infer_batch_from_path(path),
        "source_medium": "unknown",
        "embedding_source_medium": "unknown",
        "history_recompute_source_medium": "unknown",
    }
    models_json = path.with_name("models.json")
    if not models_json.exists():
        return context
    try:
        payload = json.loads(models_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return context

    metadata = payload.get("metadata", {})
    source_medium = metadata.get("source_medium") or "unknown"
    embedding_source_medium = metadata.get("embedding_source_medium") or source_medium
    history_recompute_source_medium = (
        metadata.get("history_recompute_source_medium") or source_medium
    )
    context["source_medium"] = str(source_medium).lower()
    context["embedding_source_medium"] = str(embedding_source_medium).lower()
    context["history_recompute_source_medium"] = str(
        history_recompute_source_medium
    ).lower()
    context["candidate_tokens"] = parse_int(
        metadata.get("candidates_per_user"), context["candidate_tokens"]
    )
    for model in payload.get("models", []):
        name = model.get("name", "")
        if not name:
            continue
        weight_key = model.get("weight_key", "")
        hidden = infer_hidden_from_weight_key(weight_key) or context["hidden"]
        context["models"][name] = {
            "batch_size": parse_int(model.get("batch_size"), context["batch_size"]),
            "hidden": hidden,
        }
    return context


def infer_hidden_from_weight_key(weight_key):
    match = re.search(r"_h(\d+)(?:_|$)", weight_key or "")
    return parse_int(match.group(1)) if match else 0


def infer_hidden_from_path(path):
    text = str(path)
    if "HSTU-large" in text:
        return 1024
    if "HSTU-middle" in text:
        return 512
    return 256


def infer_batch_from_path(path):
    match = re.search(r"_bs(\d+)(?:_|/|$)", str(path))
    return parse_int(match.group(1), 1) if match else 1


def infer_batch_from_detail(row):
    detail = row.get("detail", "")
    match = re.search(r"(?:^|;)movements=(\d+)(?:;|$)", detail)
    return parse_int(match.group(1), 0) if match else 0


def candidate_embedding_bytes(row, context):
    model_ctx = context["models"].get(row.get("model", ""), {})
    batch_size = (
        model_ctx.get("batch_size")
        or infer_batch_from_detail(row)
        or context["batch_size"]
        or 1
    )
    hidden = model_ctx.get("hidden") or context["hidden"] or 256
    return (
        batch_size
        * context["candidate_tokens"]
        * hidden
        * context["precision_bytes"]
    )


def render_segments(row, context):
    start = float(row["start_us"])
    duration = float(row["duration_us"])
    if duration <= 0:
        duration = 0.001

    if (
        row.get("pipe") == "preload"
        and row.get("phase") == "pre_attention"
        and row.get("name", "").endswith(".pre_attention")
    ):
        total_bytes = parse_int(row.get("bytes"))
        candidate_bytes = min(candidate_embedding_bytes(row, context), total_bytes)
        history_bytes = max(total_bytes - candidate_bytes, 0)
        if total_bytes > 0 and history_bytes > 0:
            candidate_duration = duration * candidate_bytes / total_bytes
            history_duration = duration - candidate_duration
            return [
                (start, max(candidate_duration, 0.001), "candidate_embedding"),
                (
                    start + candidate_duration,
                    max(history_duration, 0.001),
                    "history_embedding",
                ),
            ]

    return [(start, duration, display_phase(row))]


def op_id_from_row(row):
    match = re.search(r"(?:^|;)op_id=(\d+)(?:;|$)", row.get("detail", ""))
    return match.group(1) if match else ""


def render_resource_segments(row, intervals):
    """Split a top-level op span by actual Cube/Vector active intervals.

    Intervals are unioned across cores over the shared simulation clock.  The
    resulting colors describe the resource activity visible during the op
    span; gaps remain the regular gray op-span color.
    """
    start = float(row["start_us"])
    end = float(row["end_us"])
    if end <= start or not intervals:
        return [(start, max(end - start, 0.001), "op")]

    op_id = op_id_from_row(row)
    selected = [
        item for item in intervals
        if (op_id and item.get("op_id") == op_id)
        or (not op_id and item.get("op_name") == row.get("name"))
    ]
    if not selected:
        return [(start, max(end - start, 0.001), "op")]

    clipped = []
    boundaries = {start, end}
    for item in selected:
        left = max(start, float(item["start_us"]))
        right = min(end, float(item["end_us"]))
        if right <= left:
            continue
        clipped.append((left, right, item.get("resource", "")))
        boundaries.add(left)
        boundaries.add(right)
    if not clipped:
        return [(start, max(end - start, 0.001), "op")]

    points = sorted(boundaries)
    segments = []
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        cube = any(
            resource == "cube" and begin <= midpoint < finish
            for begin, finish, resource in clipped
        )
        vector = any(
            resource == "vector" and begin <= midpoint < finish
            for begin, finish, resource in clipped
        )
        phase = "overlap" if cube and vector else "cube" if cube else "vector" if vector else "op"
        if segments and segments[-1][2] == phase and abs(
            segments[-1][0] + segments[-1][1] - left
        ) < 1e-9:
            old_start, old_duration, old_phase = segments[-1]
            segments[-1] = (old_start, old_duration + right - left, old_phase)
        else:
            segments.append((left, right - left, phase))
    return segments or [(start, max(end - start, 0.001), "op")]


def preload_source_medium(row, context):
    if row.get("pipe") != "preload":
        return ""
    phase = display_phase(row)
    if phase in {"candidate_embedding", "history_embedding"}:
        if phase == "candidate_embedding":
            return context.get("embedding_source_medium") or "unknown"
        return context.get("history_recompute_source_medium") or "unknown"
    if phase == "kvcache":
        return context.get("source_medium") or "unknown"
    if phase == "post_attention":
        return context.get("source_medium") or "unknown"
    return "unknown"


def label_for(row, context):
    pipe = row["pipe"]
    if pipe == "preload":
        medium = preload_source_medium(row, context)
        return f"{macro_label(row)}:preload:{medium}"
    layer = row.get("layer_id", "")
    if layer and layer != "-1":
        return f"{macro_label(row)}:L{layer}:compute"
    return f"{macro_label(row)}:compute"


def plot_timeline(csv_path, output_path):
    rows = load_rows(csv_path)
    context = load_plot_context(csv_path)
    compute_intervals = load_compute_intervals(csv_path)
    rows = [
        row
        for row in rows
        if row.get("pipe") in {"preload", "compute"}
        and not (row.get("pipe") == "preload" and row.get("phase") == "stage")
        and not (
            row.get("pipe") == "compute"
            and row.get("phase") == "op"
            and row.get("name") == "aten::embedding"
            and float(row.get("duration_us", "0") or 0) < 0.01
        )
        and float(row.get("duration_us", "0") or 0) >= 0
    ]
    if not rows:
        raise SystemExit(f"no pipeline rows found in {csv_path}")

    labels = []
    by_label = defaultdict(list)
    for row in rows:
        label = label_for(row, context)
        if label not in by_label:
            labels.append(label)
        by_label[label].append(row)

    height = max(3.0, min(0.38 * len(labels) + 1.4, 18.0))
    fig, ax = plt.subplots(figsize=(12, height))
    colors = {
        "stage": "#4477aa",
        "candidate_embedding": "#dd8452",
        "history_embedding": "#55a868",
        "kvcache": "#4c72b0",
        "post_attention": "#c44e52",
        "other_preload": "#8172b2",
        "other": "#8172b2",
        "op": "#999999",
        "movin": "#44aa99",
        "cube": "#d95f02",
        # Keep Vector visually distinct from the existing teal MOVIN color.
        "vector": "#6baed6",
        "overlap": "#7570b3",
    }

    for lane, label in enumerate(labels):
        for row in by_label[label]:
            if (
                compute_intervals
                and row.get("pipe") == "compute"
                and row.get("phase") == "op"
            ):
                segments = render_resource_segments(row, compute_intervals)
            else:
                segments = render_segments(row, context)
            for start, duration, phase in segments:
                ax.broken_barh(
                    [(start, duration)],
                    (lane - 0.35, 0.7),
                    facecolors=colors.get(phase, "#888888"),
                    edgecolors="black",
                    linewidth=0.35,
                )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("simulation time (us)")
    ax.set_title("Macrobatch Preload / Layer Compute Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colors["candidate_embedding"],
            label="pre-attention candidate embedding",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colors["history_embedding"],
            label="pre-attention history embedding",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colors["kvcache"],
            label="KV cache read",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colors["post_attention"],
            label="post-attention preload",
        ),
        plt.Rectangle((0, 0), 1, 1, color=colors["op"], label="op span"),
        plt.Rectangle((0, 0), 1, 1, color=colors["movin"], label="MOVIN"),
    ]
    if compute_intervals:
        handles.extend([
            plt.Rectangle((0, 0), 1, 1, color=colors["cube"],
                          label="Cube GEMM compute"),
            plt.Rectangle((0, 0), 1, 1, color=colors["vector"],
                          label="Vector compute"),
            plt.Rectangle((0, 0), 1, 1, color=colors["overlap"],
                          label="Cube/Vector overlap"),
        ])
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: plot_pipeline_timeline.py <pipeline_breakdown.csv> <output.png>"
        )
    plot_timeline(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
