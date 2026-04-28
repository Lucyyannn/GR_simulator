#!/usr/bin/env python3

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def label_for(row):
    model = row["model"]
    layer = row["layer_id"]
    pipe = row["pipe"]
    phase = row.get("phase", "")
    if pipe == "preload":
        if phase and phase != "stage":
            return f"{model}:preload:{phase}"
        return f"{model}:preload:stage"
    if phase == "movin":
        if layer and layer != "-1":
            return f"{model}:L{layer}:{phase}"
        return f"{model}:{phase}"
    if layer and layer != "-1":
        return f"{model}:compute-total:L{layer}"
    return f"{model}:compute"


def plot_timeline(csv_path, output_path):
    rows = load_rows(csv_path)
    rows = [
        row
        for row in rows
        if row.get("pipe") in {"preload", "compute"}
        and float(row.get("duration_us", "0") or 0) >= 0
    ]
    if not rows:
        raise SystemExit(f"no pipeline rows found in {csv_path}")

    labels = []
    by_label = defaultdict(list)
    for row in rows:
        label = label_for(row)
        if label not in by_label:
            labels.append(label)
        by_label[label].append(row)

    height = max(3.0, min(0.38 * len(labels) + 1.4, 18.0))
    fig, ax = plt.subplots(figsize=(12, height))
    colors = {
        "stage": "#4477aa",
        "candidate_embedding": "#dd8452",
        "kvcache": "#55a868",
        "weights": "#c44e52",
        "other": "#8172b2",
        "op": "#999999",
        "movin": "#44aa99",
    }

    for lane, label in enumerate(labels):
        for row in by_label[label]:
            start = float(row["start_us"])
            duration = float(row["duration_us"])
            if duration <= 0:
                duration = 0.001
            ax.broken_barh(
                [(start, duration)],
                (lane - 0.35, 0.7),
                facecolors=colors.get(row.get("phase", ""), "#888888"),
                edgecolors="black",
                linewidth=0.35,
            )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("simulation time (us)")
    ax.set_title("Layer Preload / Core Phase Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["stage"], label="preload stage"),
        plt.Rectangle((0, 0), 1, 1, color=colors["candidate_embedding"], label="candidate embedding"),
        plt.Rectangle((0, 0), 1, 1, color=colors["kvcache"], label="KVCache"),
        plt.Rectangle((0, 0), 1, 1, color=colors["weights"], label="weights"),
        plt.Rectangle((0, 0), 1, 1, color=colors["op"], label="op span"),
        plt.Rectangle((0, 0), 1, 1, color=colors["movin"], label="MOVIN"),
    ]
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
