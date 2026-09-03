#!/usr/bin/env python3
"""Combine HSTU method matrices into an exact E2E-QPS comparison table.

QPS is derived from the simulator's end-to-end NPU latency:
``QPS = batch_size / (E2E_latency_us * 1e-6)``.  Existing RE/CA/AR/both
results are reused; GRACE is supplied as a separate matrix with final NPU
configuration overrides.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CHIPS = ("910A", "910B", "910C")
MODELS = ("small", "middle", "large")
SEQUENCES = (4096, 8192, 16384)
BATCHES = (1, 4, 8)
USERS = ("hot", "cold")
METHODS = {
    "Full_Recompute": "RE",
    "Full_Cache": "CA",
    "w_AR": "AR",
    "w_both": "both",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--both-root", type=Path, required=True)
    parser.add_argument("--grace-root", type=Path, required=True)
    parser.add_argument(
        "--extra-root", action="append", type=Path, default=[],
        help="Additional matrix roots, used for completed gap-fill cases.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def npu_latency_us(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("component") == "NPU" and row.get("scope") == "overall":
                return float(row["sim_time_us"])
    return None


def collect(root: Path) -> dict[tuple, float]:
    values = {}
    for path in root.glob("cases/*/*/HSTU-*_seq*_bs*_*/*"):
        # The glob above intentionally does not assume a method directory
        # spelling.  Parse the stable path components below instead.
        if path.name != "hardware_summary.csv":
            continue
        relative = path.relative_to(root / "cases")
        chip, method, case_name = relative.parts[:3]
        if method not in METHODS:
            continue
        prefix, user = case_name.rsplit("_", 1)
        model, sequence, batch = prefix[len("HSTU-"):].split("_", 2)
        key = (chip, model, int(sequence[len("seq"):]),
               int(batch[len("bs"):]), user, METHODS[method])
        latency = npu_latency_us(path)
        if latency is not None:
            values[key] = latency
    return values


def collect_grace(root: Path) -> dict[tuple, float]:
    values = collect(root)
    return {
        (*key[:-1], "GRACE"): latency
        for key, latency in values.items()
        if key[-1] == "both"
    }


def main() -> None:
    args = parse_args()
    latency = collect(args.base_root)
    latency.update(collect(args.both_root))
    for root in args.extra_root:
        latency.update(collect(root))
    latency.update(collect_grace(args.grace_root))
    expected = {
        (chip, model, sequence, batch, user, method)
        for chip in CHIPS for model in MODELS for sequence in SEQUENCES
        for batch in BATCHES for user in USERS
        for method in (*METHODS.values(), "GRACE")
    }
    missing = sorted(expected - set(latency))
    if missing:
        example = ", ".join(map(str, missing[:3]))
        raise SystemExit(f"missing {len(missing)} QPS cases, e.g. {example}")
    rows = []
    for key in sorted(expected):
        chip, model, sequence, batch, user, method = key
        value = latency[key]
        rows.append({
            "chip": chip,
            "model": model,
            "seq_len": sequence,
            "batch_size": batch,
            "user": user,
            "method": method,
            "e2e_latency_us": value,
            "qps": batch * 1e6 / value,
        })
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    with (output / "qps_by_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for chip in CHIPS:
        for method in (*METHODS.values(), "GRACE"):
            values = [row["qps"] for row in rows if row["chip"] == chip and row["method"] == method]
            summary.append({
                "chip": chip,
                "method": method,
                "case_count": len(values),
                "geomean_qps": math.prod(values) ** (1.0 / len(values)),
            })
    with (output / "qps_geomean_by_chip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(rows)} QPS rows to {output}", flush=True)


if __name__ == "__main__":
    main()
