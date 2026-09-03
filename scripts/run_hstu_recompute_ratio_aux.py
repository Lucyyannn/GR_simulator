#!/usr/bin/env python3
"""Run a disjoint ratio-sweep shard without touching global summary files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_hstu_recompute_ratio_sweep as sweep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--chips", nargs="+", choices=sweep.CHIPS, required=True)
    parser.add_argument("--models", nargs="+", choices=sweep.MODELS, required=True)
    parser.add_argument("--seq-lens", nargs="+", type=int, choices=sweep.SEQS, default=list(sweep.SEQS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=sweep.BATCHES, default=list(sweep.BATCHES))
    parser.add_argument("--users", nargs="+", choices=sweep.USERS, default=list(sweep.USERS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(sweep.RATIOS))
    parser.add_argument("--log-level", default="warn")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.max_concurrent <= sweep.MAX_CONCURRENT:
        raise SystemExit(f"--max-concurrent must be in [1, {sweep.MAX_CONCURRENT}]")
    repo = Path(__file__).resolve().parents[1]
    cases = sweep.matrix_cases(
        args.chips,
        args.models,
        args.seq_lens,
        args.batch_sizes,
        args.users,
        args.ratios,
    )
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                sweep.run_case,
                repo,
                args.result_root,
                sweep.CHIP_CONFIGS[case.chip],
                args.log_level,
                case,
                False,
            ): case
            for case in cases
        }
        failures = 0
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            failures += record["returncode"] != 0
            print(
                f"[{completed}/{len(cases)}] {record['case_id']} "
                f"rc={record['returncode']} wall={record.get('wall_seconds', 0):.1f}s",
                flush=True,
            )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
