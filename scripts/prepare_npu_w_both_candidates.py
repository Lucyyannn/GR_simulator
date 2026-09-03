#!/usr/bin/env python3
"""Select a compact, diverse simulator-validation set from analytic NPU DSE."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import npu_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-chip", type=int, default=30)
    parser.add_argument("--top-overall", type=int, default=10)
    parser.add_argument("--top-vector-multiple", type=int, default=10)
    parser.add_argument("--balance-bins", type=int, default=9)
    parser.add_argument("--cube-compression", type=int, default=1)
    return parser.parse_args()


def append_unique(selected: list[int], candidates, limit: int) -> None:
    for raw in candidates:
        index = int(raw)
        if index not in selected:
            selected.append(index)
        if len(selected) >= limit:
            return


def select_indices(data, args: argparse.Namespace, baseline: npu_config.DesignPoint):
    nc = data["cube_cores"]
    nv = data["vector_cores"]
    wv = data["vector_width_bits"]
    score = data["geomean_request_latency_s"]
    valid = np.flatnonzero(np.isfinite(score))
    ranked = valid[np.argsort(score[valid], kind="stable")]
    selected: list[int] = []

    append_unique(selected, ranked[: args.top_overall], args.per_chip)
    multiples = ranked[(nv[ranked] % nc[ranked]) == 0]
    append_unique(selected, multiples[: args.top_vector_multiple], args.per_chip)

    # Cover the Cube/Vector throughput trade-off without fitting workload-specific
    # correction terms: choose the best analytic point in each log-ratio bin.
    balance = (
        nv[valid].astype(np.float64) * wv[valid].astype(np.float64)
        / nc[valid].astype(np.float64)
    )
    edges = np.quantile(np.log(balance), np.linspace(0.0, 1.0, args.balance_bins + 1))
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = valid[(np.log(balance) >= left) & (np.log(balance) <= right)]
        if len(in_bin):
            append_unique(selected, [in_bin[np.argmin(score[in_bin])]], args.per_chip)

    baseline_distance = (
        np.abs(nc - baseline.cube_cores) / baseline.cube_cores
        + np.abs(
            nv.astype(np.float64) * wv.astype(np.float64)
            - baseline.vector_cores * baseline.vector_width_bits
        )
        / (baseline.vector_cores * baseline.vector_width_bits)
    )
    append_unique(selected, valid[np.argsort(baseline_distance[valid])], args.per_chip)
    append_unique(selected, ranked, args.per_chip)
    return selected


def main() -> None:
    args = parse_args()
    if args.per_chip <= 0:
        raise SystemExit("--per-chip must be positive")
    config_root = args.output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "method": "w_both",
        "selection": {
            "per_chip": args.per_chip,
            "top_overall": args.top_overall,
            "top_vector_multiple": args.top_vector_multiple,
            "balance_bins": args.balance_bins,
            "cube_compression": args.cube_compression,
        },
        "candidates": [],
    }
    for chip_dir in sorted(p for p in args.search_root.iterdir() if p.is_dir()):
        score_path = chip_dir / "candidate_scores.npz"
        baseline_path = Path("configs") / f"{chip_dir.name}.json"
        if not score_path.is_file() or not baseline_path.is_file():
            continue
        baseline_config = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = npu_config.baseline_design(
            baseline_config, args.cube_compression
        )
        with np.load(score_path) as data:
            indices = select_indices(data, args, baseline)
            for rank, index in enumerate(indices, start=1):
                point = npu_config.DesignPoint(
                    int(data["cube_cores"][index]),
                    int(data["vector_cores"][index]),
                    int(data["vector_width_bits"][index]),
                )
                name = f"{chip_dir.name}_dse_{rank:02d}"
                generated, mapping = npu_config.materialize_config(
                    baseline_config, point,
                    cube_compression=args.cube_compression,
                )
                path = config_root / f"{name}.json"
                path.write_text(
                    json.dumps(generated, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                manifest["candidates"].append({
                    "candidate": name,
                    "chip": chip_dir.name,
                    "analytic_rank_in_validation_set": rank,
                    "design": asdict(point),
                    "area_mm2": float(data["area_mm2"][index]),
                    "power_w": float(data["power_w"][index]),
                    "predicted_geomean_latency_us": float(
                        data["geomean_request_latency_s"][index] * 1e6
                    ),
                    "vector_is_cube_multiple": point.vector_cores % point.cube_cores == 0,
                    "config": str(path),
                    "simulation_mapping": mapping,
                })
    (args.output_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest['candidates'])} candidates to {config_root}")


if __name__ == "__main__":
    main()
