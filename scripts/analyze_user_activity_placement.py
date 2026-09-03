#!/usr/bin/env python3
"""Estimate QPS and DRAM hit rate for user-activity-aware KV placement.

CA randomly places users in DRAM, so its expected interaction hit rate equals
the resident-user fraction. REFORGE places the most active users first. Two
REFORGE capacity policies are reported: both account for Action KV reuse, while
only ``REFORGE-AA+IR`` also removes recomputed Item KV from persistent storage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("configs/figure_data/user_activity_placement_inputs.json")
DEFAULT_ACTIVITY = Path("configs/kuairand_1k_user_activity_distribution.csv")
DEFAULT_OUTPUT = Path("results/analysis/user_activity_placement/summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--activity-csv", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_activity(path: Path) -> list[int]:
    with resolve(path).open(newline="", encoding="utf-8") as handle:
        values = [int(row["number_of_interactions"]) for row in csv.DictReader(handle)]
    if not values or any(value < 0 for value in values):
        raise ValueError("activity CSV must contain non-negative interactions")
    return sorted(values, reverse=True)


def retained_action_rows(sequence_length: int, total_reuse_ratio: float) -> int:
    """Match the simulator's conversion from total reuse to Action reuse."""

    action_rows = sequence_length // 2
    action_reuse_ratio = min(
        1.0, max(0.0, total_reuse_ratio) * sequence_length / action_rows
    )
    if action_reuse_ratio <= 0.0:
        return action_rows
    return max(1, int(round(action_rows * (1.0 - action_reuse_ratio))))


def kv_bytes_per_user(
    layers: int, hidden: int, cached_rows: int, bytes_per_element: float,
) -> float:
    # K/V factor 2; all layers retain their own persistent item-KV state.
    return 2.0 * layers * hidden * cached_rows * bytes_per_element


def activity_aware_hit_rate(
    activity: list[int], user_count: int, resident_users: int,
) -> float:
    """Replicate the empirical 1K distribution to the requested population."""

    if user_count % len(activity) != 0:
        raise ValueError(
            f"user count {user_count} must be divisible by {len(activity)}"
        )
    copies = user_count // len(activity)
    remaining = min(user_count, resident_users)
    resident_interactions = 0
    for interactions in activity:
        count = min(copies, remaining)
        resident_interactions += count * interactions
        remaining -= count
        if remaining == 0:
            break
    total_interactions = copies * sum(activity)
    return resident_interactions / total_interactions if total_interactions else 0.0


def weighted_qps(batch_size: int, hit_rate: float, dram_us: float, ssd_us: float) -> float:
    latency_us = hit_rate * dram_us + (1.0 - hit_rate) * ssd_us
    return batch_size * 1e6 / latency_us


def main() -> None:
    args = parse_args()
    config = json.loads(resolve(args.input).read_text(encoding="utf-8"))
    activity = load_activity(args.activity_csv)

    sequence_length = int(config["sequence_length"])
    item_rows = (sequence_length + 1) // 2
    action_rows = retained_action_rows(
        sequence_length, float(config["ar_total_reuse_ratio"])
    )
    recomputed_items = int(config["ir_recomputed_item_rows"])
    if not 0 <= recomputed_items <= item_rows:
        raise ValueError("ir_recomputed_item_rows is outside the Item-token range")

    rows_by_policy = {
        "CA": sequence_length,
        "REFORGE-AA": item_rows + action_rows,
        "REFORGE-AA+IR": item_rows - recomputed_items + action_rows,
    }
    capacity_bytes = float(config["ddr_capacity_gib"]) * 2**30
    bytes_per_user = {
        policy: kv_bytes_per_user(
            int(config["layers"]), int(config["hidden_size"]), cached_rows,
            float(config["bytes_per_element"]),
        )
        for policy, cached_rows in rows_by_policy.items()
    }
    resident_users = {
        policy: int(capacity_bytes // storage)
        for policy, storage in bytes_per_user.items()
    }
    latency = {key: float(value) for key, value in config["latency_us"].items()}
    batch_size = int(config["batch_size"])

    output_rows = []
    for user_count in (int(value) for value in config["user_counts"]):
        ca_resident = min(user_count, resident_users["CA"])
        ca_hit = ca_resident / user_count  # Expected hit rate under random placement.
        no_ir_resident = min(user_count, resident_users["REFORGE-AA"])
        no_ir_random_hit = no_ir_resident / user_count
        with_ir_resident = min(user_count, resident_users["REFORGE-AA+IR"])
        no_ir_hit = activity_aware_hit_rate(activity, user_count, no_ir_resident)
        with_ir_hit = activity_aware_hit_rate(activity, user_count, with_ir_resident)

        output_rows.append({
            "chip": config["chip"], "model": config["model"],
            "seq_len": sequence_length, "batch_size": batch_size,
            "users": user_count,
            "re_qps": batch_size * 1e6 / latency["re_ssd"],
            "ca_qps": weighted_qps(
                batch_size, ca_hit, latency["ca_dram"], latency["ca_ssd"]
            ),
            "reforge_ssd_qps": batch_size * 1e6 / latency["reforge_ssd"],
            "reforge_random_qps": weighted_qps(
                batch_size, no_ir_random_hit,
                latency["reforge_dram"], latency["reforge_ssd"],
            ),
            "reforge_aa_qps": weighted_qps(
                batch_size, no_ir_hit,
                latency["reforge_dram"], latency["reforge_ssd"],
            ),
            "reforge_aa_ir_qps": weighted_qps(
                batch_size, with_ir_hit,
                latency["reforge_dram"], latency["reforge_ssd"],
            ),
            "re_dram_hit_rate": 0.0,
            "ca_dram_hit_rate": ca_hit,
            "reforge_ssd_dram_hit_rate": 0.0,
            "reforge_random_dram_hit_rate": no_ir_random_hit,
            "reforge_aa_dram_hit_rate": no_ir_hit,
            "reforge_aa_ir_dram_hit_rate": with_ir_hit,
            "ca_resident_users": ca_resident,
            "reforge_aa_resident_users": no_ir_resident,
            "reforge_aa_ir_resident_users": with_ir_resident,
            "ca_mib_per_user": bytes_per_user["CA"] / 2**20,
            "reforge_aa_mib_per_user": bytes_per_user["REFORGE-AA"] / 2**20,
            "reforge_aa_ir_mib_per_user": bytes_per_user["REFORGE-AA+IR"] / 2**20,
            "retained_action_rows": action_rows,
            "recomputed_item_rows": recomputed_items,
        })

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} population points to {output}")


if __name__ == "__main__":
    main()
