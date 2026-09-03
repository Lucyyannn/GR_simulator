#!/usr/bin/env python3
"""Run prioritized RE/CA/AR followed by overlapping w_both ratio sweeps.

All RE/CA/AR cases are submitted first.  Once the last base case has been
submitted, completed slots immediately admit ratio-sweep work; slow base cases
may therefore overlap the sweep.  One scheduler owns both phases, so the CPU
and memory limits apply to their combined execution.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import run_hstu_recompute_ratio_sweep as ratio
import run_meta_hstu_full_matrix as base


CHIPS = ("910A", "910B", "910C", "MTIA2")
MODELS = ("small", "middle", "large")
SEQUENCES = (4096, 6144, 8192)
BATCHES = (1, 2, 4)
USERS = ("hot", "cold")
METHODS = ("Full_Recompute", "Full_Cache", "w_AR")
RATIOS = tuple(index / 10 for index in range(11))
MODEL_PRIORITY = {"small": 0, "middle": 1, "large": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("configs/item_kv_calib.json"))
    parser.add_argument("--max-concurrent", type=int, default=196)
    parser.add_argument("--max-simulator-rss-gib", type=float, default=460.0)
    parser.add_argument("--memory-headroom-gib", type=float, default=10.0)
    parser.add_argument("--log-level", default="warn")
    parser.add_argument(
        "--base-methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Select base methods; use only w_AR when refreshing AR semantics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep successful status records and run only unfinished cases",
    )
    return parser.parse_args()


def reservation_gib(case) -> float:
    return base.MemoryAdmissionController.case_reservation_gib(case)


def run_base(repo, root, calibration, log_level, case):
    return base.run_case(
        repo, root, base.CHIP_CONFIGS[case.chip], calibration, log_level,
        case, False, None, f"calibration_base__{case.case_id}",
    )


def run_ratio(repo, root, log_level, case):
    return ratio.run_case(
        repo, root, ratio.CHIP_CONFIGS[case.chip], log_level, case, False
    )


def load_successful_records(log_dir: Path) -> list[dict]:
    """Load completed cases for an explicit, same-result-root resume."""

    records = []
    for status_path in sorted(log_dir.glob("*.status.json")):
        try:
            record = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("returncode") == 0 and record.get("case_id"):
            records.append(record)
    return records


def write_state(path: Path, *, base_total, base_pending, ratio_pending, active, base_records, ratio_records) -> None:
    path.write_text(json.dumps({
        "base_total": base_total,
        "base_completed": len(base_records),
        "base_pending_submission": len(base_pending),
        "ratio_total": 2376,
        "ratio_completed": len(ratio_records),
        "ratio_pending_submission": len(ratio_pending),
        "active": len(active),
        "failures": sum(r.get("returncode") != 0 for r in (*base_records, *ratio_records)),
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_concurrent <= 196:
        raise SystemExit("--max-concurrent must be in [1, 196]")
    usable_memory = args.max_simulator_rss_gib - args.memory_headroom_gib
    if usable_memory <= 0:
        raise SystemExit("memory limit must exceed its headroom")

    repo = Path(__file__).resolve().parents[1]
    root = args.result_root
    base_root = root / "base_methods"
    ratio_root = root / "w_both_ratio"
    base_root.mkdir(parents=True, exist_ok=True)
    ratio_root.mkdir(parents=True, exist_ok=True)

    base_pending = base.matrix_cases(
        CHIPS, MODELS, SEQUENCES, BATCHES, USERS, args.base_methods
    )
    base_total = len(base_pending)
    total_cases = base_total + 2376
    ratio_pending = ratio.matrix_cases(
        CHIPS, MODELS, SEQUENCES, BATCHES, USERS, RATIOS
    )
    base_records = load_successful_records(base_root / "logs") if args.resume else []
    ratio_records = load_successful_records(ratio_root / "logs") if args.resume else []
    completed_base_ids = {record["case_id"] for record in base_records}
    completed_ratio_ids = {record["case_id"] for record in ratio_records}
    base_pending = [case for case in base_pending if case.case_id not in completed_base_ids]
    ratio_pending = [case for case in ratio_pending if case.case_id not in completed_ratio_ids]

    base_pending.sort(key=lambda case: (reservation_gib(case), case.case_id))
    # Keep model priority explicit: all small and middle sweeps are submitted
    # before any large sweep, while retaining memory-friendly ordering within
    # each model size.
    ratio_pending.sort(key=lambda case: (
        MODEL_PRIORITY[case.model], reservation_gib(case), case.case_id
    ))
    (root / "matrix_definition.json").write_text(json.dumps({
        "scheduling": "submit_all_base_before_ratio; overlap_ratio_with_base_stragglers",
        "chips": CHIPS,
        "models": MODELS,
        "seq_lens": SEQUENCES,
        "batch_sizes": BATCHES,
        "users": USERS,
        "base_methods": args.base_methods,
        "recompute_ratios": RATIOS,
        "base_case_count": base_total,
        "ratio_case_count": 2376,
        "total_case_count": total_cases,
        "max_concurrent": args.max_concurrent,
        "max_simulator_rss_gib": args.max_simulator_rss_gib,
        "memory_headroom_gib": args.memory_headroom_gib,
        "ratio_model_priority": tuple(MODEL_PRIORITY),
        "resumed": args.resume,
    }, indent=2) + "\n", encoding="utf-8")

    active = {}
    reserved = 0.0
    completed = len(base_records) + len(ratio_records)

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        while base_pending or ratio_pending or active:
            # Strict submission priority gives every base case a running or
            # completed slot before ratio work starts.  It does not wait for
            # slow base cases to finish.
            pending = base_pending if base_pending else ratio_pending
            phase = "base" if base_pending else "ratio"
            while pending and len(active) < args.max_concurrent:
                needed = reservation_gib(pending[0])
                if reserved + needed > usable_memory + 1e-12:
                    break
                case = pending.pop(0)
                reserved += needed
                if phase == "base":
                    future = executor.submit(
                        run_base, repo, base_root, args.calibration,
                        args.log_level, case,
                    )
                else:
                    future = executor.submit(
                        run_ratio, repo, ratio_root, args.log_level, case,
                    )
                active[future] = (phase, case, needed)

            if not active:
                needed = reservation_gib(pending[0]) if pending else 0.0
                raise RuntimeError(
                    f"cannot admit next {phase} case: needs {needed:.2f} GiB, "
                    f"usable limit is {usable_memory:.2f} GiB"
                )

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                finished_phase, case, reservation = active.pop(future)
                reserved = max(0.0, reserved - reservation)
                record = future.result()
                if finished_phase == "base":
                    base_records.append(record)
                else:
                    ratio_records.append(record)
                completed += 1
                if completed % 10 == 0 or record.get("returncode") != 0:
                    base.write_manifest(base_root, base_records)
                    ratio.write_manifest(ratio_root, ratio_records)
                    write_state(
                        root / "progress.json", base_total=base_total,
                        base_pending=base_pending,
                        ratio_pending=ratio_pending, active=active,
                        base_records=base_records, ratio_records=ratio_records,
                    )
                print(
                    f"[{completed}/{total_cases}] {finished_phase} {case.case_id} "
                    f"rc={record.get('returncode')} active={len(active)}",
                    flush=True,
                )

    base.write_manifest(base_root, base_records)
    ratio.write_manifest(ratio_root, ratio_records)
    write_state(
        root / "progress.json", base_total=base_total,
        base_pending=base_pending,
        ratio_pending=ratio_pending, active=active,
        base_records=base_records, ratio_records=ratio_records,
    )
    failures = sum(r.get("returncode") != 0 for r in (*base_records, *ratio_records))
    (root / "run_complete.json").write_text(json.dumps({
        "case_count": len(base_records) + len(ratio_records),
        "failures": failures,
    }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
