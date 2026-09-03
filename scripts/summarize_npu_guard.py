#!/usr/bin/env python3
"""Summarize candidate w_both runs and select one robust design per NPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from summarize_hstu_qps import collect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--guard-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=54)
    parser.add_argument("--min-speedup", type=float, default=0.95)
    parser.add_argument("--selected-config-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = collect(args.baseline_root)
    rows = []
    for candidate_root in sorted(path for path in args.guard_root.iterdir() if path.is_dir()):
        candidate = candidate_root.name
        config = json.loads((args.config_root / f"{candidate}.json").read_text())
        design = config["metadata"]["npu_reconfiguration"]["physical_design"]
        values = collect(candidate_root)
        speedups = []
        for key, latency in values.items():
            if key[-1] != "both" or key not in baseline:
                continue
            speedups.append((key, baseline[key] / latency))
        completed = len(speedups)
        geomean = math.prod(value for _, value in speedups) ** (1 / completed) if completed else None
        worst_key, worst = min(speedups, key=lambda item: item[1]) if speedups else (None, None)
        rows.append({
            "candidate": candidate,
            **design,
            "chip": candidate.split("_", 1)[0],
            "completed_cases": completed,
            "geomean_speedup_vs_w_both": geomean,
            "min_speedup_vs_w_both": worst,
            "worst_case": "/".join(map(str, worst_key[:-1])) if worst_key else "",
            "passes_guard": (
                completed == args.expected_cases
                and worst is not None and worst >= args.min_speedup
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} candidate summaries to {args.output}")
    selected = {}
    for chip in sorted({row["chip"] for row in rows}):
        eligible = [
            row for row in rows
            if row["chip"] == chip and row["passes_guard"]
        ]
        if not eligible:
            continue
        winner = max(
            eligible,
            key=lambda row: (
                row["geomean_speedup_vs_w_both"],
                row["min_speedup_vs_w_both"],
            ),
        )
        selected[chip] = winner
        if args.selected_config_root is not None:
            args.selected_config_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                args.config_root / f"{winner['candidate']}.json",
                args.selected_config_root / f"{chip}_w_both_optimal.json",
            )
    selection_path = args.output.with_name("selected_npu_configs.json")
    selection_path.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"selected robust designs for {len(selected)} NPUs")


if __name__ == "__main__":
    main()
