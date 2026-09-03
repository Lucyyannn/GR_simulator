#!/usr/bin/env python3
"""Run NPU-reconfiguration candidates on a pilot or full w_both matrix."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_meta_hstu_full_matrix import (
    CHIP_CONFIGS,
    Case,
    MemoryAdmissionController,
    matrix_cases,
    run_case,
)


PILOT_WORKLOADS = (
    ("small", 4096, 1, "hot"), ("large", 8192, 1, "hot"),
    ("middle", 6144, 1, "cold"), ("large", 8192, 1, "cold"),
    ("small", 8192, 2, "hot"), ("middle", 4096, 2, "hot"),
    ("small", 4096, 2, "cold"), ("large", 6144, 2, "cold"),
    ("middle", 8192, 4, "hot"), ("large", 4096, 4, "hot"),
    ("small", 6144, 4, "cold"), ("middle", 8192, 4, "cold"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--model-result-root", action="append", default=[], metavar="MODEL=PATH",
        help="Store one model's cases under PATH instead of --result-root.",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--workload-set", choices=("pilot", "full"), default="full",
        help="pilot uses 12 balanced workloads; full uses all 54 workloads.",
    )
    parser.add_argument("--models", nargs="+", choices=("small", "middle", "large"),
                        default=["large"])
    parser.add_argument("--max-concurrent", type=int, default=165)
    parser.add_argument("--max-simulator-rss-gib", type=float, default=460.0)
    parser.add_argument("--max-total-simulators", type=int, default=196)
    parser.add_argument("--memory-headroom-gib", type=float, default=10.0)
    parser.add_argument("--log-level", default="warn")
    return parser.parse_args()


def chip_for_candidate(name: str) -> str:
    prefix = name.split("_", 1)[0]
    chip = prefix
    if chip not in CHIP_CONFIGS:
        raise ValueError(f"cannot infer chip from candidate name {name!r}")
    return chip


def cases_for(chip: str, models: list[str], workload_set: str):
    if workload_set == "pilot":
        return [
            Case(chip, model, sequence, batch, user, "w_both")
            for model, sequence, batch, user in PILOT_WORKLOADS
            if model in models
        ]
    return matrix_cases(
        [chip], models, [4096, 6144, 8192], [1, 2, 4],
        ["hot", "cold"], ["w_both"],
    )


def parse_model_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        model, separator, path = value.partition("=")
        if separator != "=" or model not in {"small", "middle", "large"} or not path:
            raise ValueError(f"invalid --model-result-root {value!r}")
        roots[model] = Path(path)
    return roots


def write_manifest(path: Path, records: list[dict]) -> None:
    fields = ["candidate", "case_id", "chip", "model", "seq_len", "batch_size", "user", "method", "returncode", "wall_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_concurrent <= 196:
        raise SystemExit("--max-concurrent must be in [1, 196]")
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    model_roots = parse_model_roots(args.model_result_root)
    candidates = sorted(args.candidate_config_root.glob("*.json"))
    if not candidates:
        raise SystemExit("no candidate configurations found")
    jobs = []
    for config in candidates:
        candidate = config.stem
        chip = chip_for_candidate(candidate)
        for case in cases_for(chip, args.models, args.workload_set):
            candidate_root = model_roots.get(case.model, root) / candidate
            if (candidate_root / "cases" / case.chip / case.method / f"HSTU-{case.model}_seq{case.seq_len}_bs{case.batch_size}_{case.user}" / "hardware_summary.csv").is_file():
                continue
            jobs.append((candidate, candidate_root, config, case))
    jobs.sort(key=lambda job: (
        MemoryAdmissionController.case_reservation_gib(job[3]),
        job[0], job[3].case_id,
    ))
    admission = MemoryAdmissionController(
        args.max_simulator_rss_gib, args.memory_headroom_gib, root,
        args.max_total_simulators,
    )
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(run_case, Path(__file__).resolve().parents[1], candidate_root,
                            config, args.calibration, args.log_level, case, False, admission,
                            f"{candidate}__{case.case_id}"):
            (candidate, case)
            for candidate, candidate_root, config, case in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            candidate, _ = futures[future]
            record = future.result()
            record["candidate"] = candidate
            records.append(record)
            write_manifest(root / "guard_manifest.csv", records)
            print(f"[{completed}/{len(jobs)}] {candidate}/{record['case_id']} rc={record['returncode']}", flush=True)
    failures = sum(record["returncode"] != 0 for record in records)
    (root / "guard_run_complete.json").write_text(
        f'{{"case_count": {len(records)}, "failures": {failures}}}\n', encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
