#!/usr/bin/env python3
"""Run the corrected Meta HSTU matrix without overwriting earlier results."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


MODELS = {
    "small": (4, 256),
    "middle": (8, 512),
    "large": (12, 1024),
}
CHIPS = ("910C", "910B", "910A")
SEQS = (4096, 8192, 16384)
BATCHES = (1, 4, 8)
USERS = ("hot", "cold")
METHODS = ("Full_Recompute", "Full_Cache", "w_AR", "w_IR", "w_both")
KV_REUSE_RATIO = 0.4360


@dataclass(frozen=True)
class Case:
    chip: str
    model: str
    seq_len: int
    batch_size: int
    user: str
    method: str

    @property
    def case_id(self):
        return (
            f"{self.chip}__{self.model}__seq{self.seq_len}__bs{self.batch_size}"
            f"__{self.user}__{self.method}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=48)
    parser.add_argument("--chips", nargs="+", choices=CHIPS, default=list(CHIPS))
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, choices=SEQS, default=list(SEQS)
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, choices=BATCHES,
        default=list(BATCHES),
    )
    parser.add_argument("--users", nargs="+", choices=USERS, default=list(USERS))
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("scripts/recompute_ratio_calibration.json"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Override the chip config (intended for a single selected chip).",
    )
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def case_dir(root, case):
    return (
        root
        / "cases"
        / case.chip
        / case.method
        / f"HSTU-{case.model}_seq{case.seq_len}_bs{case.batch_size}_{case.user}"
    )


def compute_k(repo, base_config, calibration, case, layers, hidden, enable_ar):
    command = [
        "python3",
        "scripts/recompute_ratio_cost_model_new.py",
        "--config",
        str(base_config),
        "--calibration",
        str(calibration),
        "--user",
        case.user,
        "--layers",
        str(layers),
        "--hidden",
        str(hidden),
        "--kv-len",
        str(case.seq_len),
        "--batch-size",
        str(case.batch_size),
        "--candidates",
        "128",
        "--history-embedding-source",
        "ddr" if case.user == "hot" else "ssd",
        "--candidate-embedding-source",
        "ssd",
        "--field",
        "json",
    ]
    if enable_ar:
        command += ["--enable-kv-reuse", "--kv-reuse-ratio", str(KV_REUSE_RATIO)]
    result = subprocess.run(
        command, cwd=repo, text=True, capture_output=True, check=True
    )
    payload = json.loads(result.stdout)
    return int(payload["history_recompute_len"]), payload


def build_command(repo, root, base_config, calibration, log_level, case):
    layers, hidden = MODELS[case.model]
    source = "ddr" if case.user == "hot" else "ssd"
    output = case_dir(root, case)
    command = [
        "bash",
        "scripts/run_hstu.sh",
        "--base-config",
        str(base_config),
        "--result-dir",
        str(output),
        "--source-medium",
        source,
        "--embedding-source-medium",
        "ssd",
        "--history-recompute-source-medium",
        source,
        "--layers",
        str(layers),
        "--hidden",
        str(hidden),
        "--kv-len",
        str(case.seq_len),
        "--num-users",
        str(case.batch_size),
        "--users-per-batch",
        str(case.batch_size),
        "--candidates-per-user",
        "128",
        "--macro-batch-size",
        "128",
        "--vocab",
        "262144",
        "--attention-modeling",
        "fused",
        "--disable-ar-reduce-attention-compute",
        "--log-level",
        log_level,
    ]
    selection = None
    if case.method == "Full_Recompute":
        command += [
            "--history-recompute-len",
            str(case.seq_len),
            "--history-recompute-index-mode",
            "random",
        ]
    elif case.method == "Full_Cache":
        command += ["--history-recompute-len", "0"]
    elif case.method == "w_AR":
        command += [
            "--history-recompute-len",
            "0",
            "--enable-kv-reuse",
            "--kv-reuse-ratio",
            str(KV_REUSE_RATIO),
        ]
    elif case.method in {"w_IR", "w_both"}:
        enable_ar = case.method == "w_both"
        k, selection = compute_k(
            repo, base_config, calibration, case, layers, hidden, enable_ar
        )
        command += ["--history-recompute-len", str(k)]
        if enable_ar:
            command += [
                "--enable-kv-reuse",
                "--kv-reuse-ratio",
                str(KV_REUSE_RATIO),
            ]
    else:
        raise ValueError(case.method)
    return command, selection


def file_digest(repo, path):
    resolved = path if path.is_absolute() else repo / path
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digests(repo, base_config, calibration):
    paths = {
        "base_config": base_config,
        "calibration": calibration,
        "run_hstu": Path("scripts/run_hstu.sh"),
        "trace_generator": Path("scripts/generate_hstu_baseline_trace.py"),
        "cost_model": Path("scripts/recompute_ratio_cost_model_new.py"),
        "simulator": Path("build/bin/Simulator"),
    }
    return {name: file_digest(repo, path) for name, path in paths.items()}


def run_case(repo, root, base_config, calibration, log_level, case, dry_run):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / f"{case.case_id}.status.json"

    started = time.time()
    record = {**asdict(case), "case_id": case.case_id, "started_unix": started}
    try:
        command, selection = build_command(
            repo, root, base_config, calibration, log_level, case
        )
        record["command"] = command
        record["input_digests"] = input_digests(
            repo, base_config, calibration
        )
        if status_path.exists() and not dry_run:
            old = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                old.get("returncode") == 0
                and not old.get("dry_run", False)
                and old.get("command") == command
                and old.get("input_digests") == record["input_digests"]
            ):
                return old
        if selection is not None and not dry_run:
            selection_path = logs / f"{case.case_id}.ir_selection.json"
            selection_path.write_text(
                json.dumps(selection, indent=2) + "\n", encoding="utf-8"
            )
        if selection is not None:
            record["history_recompute_len"] = selection["history_recompute_len"]
        if dry_run:
            record["returncode"] = 0
            record["dry_run"] = True
        else:
            env = os.environ.copy()
            env["WAIT_IF_SIMULATOR_RUNNING"] = "0"
            env["SIMULATOR_SLOT_LOCK"] = f"/tmp/{case.case_id}.lock"
            env["MPLBACKEND"] = "Agg"
            stdout_path = logs / f"{case.case_id}.stdout.log"
            stderr_path = logs / f"{case.case_id}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, \
                 stderr_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            record["returncode"] = result.returncode
    except Exception as exc:  # keep the rest of the matrix running
        record["returncode"] = -1
        record["error"] = repr(exc)
    record["finished_unix"] = time.time()
    record["wall_seconds"] = record["finished_unix"] - started
    if not dry_run:
        status_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    return record


def write_manifest(root, records):
    fields = [
        "case_id", "chip", "model", "seq_len", "batch_size", "user",
        "method", "history_recompute_len", "returncode", "wall_seconds",
    ]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["case_id"]))


def main():
    args = parse_args()
    if not 1 <= args.max_concurrent <= 48:
        raise SystemExit("--max-concurrent must be in [1, 48]")
    repo = Path(__file__).resolve().parents[1]
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    if args.base_config is not None and len(args.chips) != 1:
        raise SystemExit("--base-config requires exactly one --chips value")
    cases = [
        Case(chip, model, seq, batch, user, method)
        for chip in args.chips
        for model in args.models
        for seq in args.seq_lens
        for batch in args.batch_sizes
        for user in args.users
        for method in args.methods
    ]
    (root / "matrix_definition.json").write_text(
        json.dumps(
            {
                "case_count": len(cases),
                "chip_order": args.chips,
                "models": {name: MODELS[name] for name in args.models},
                "seq_lens": args.seq_lens,
                "batch_sizes": args.batch_sizes,
                "users": args.users,
                "methods": args.methods,
                "max_concurrent": args.max_concurrent,
                "ar_reduce_attention_compute": False,
                "kv_reuse_ratio": KV_REUSE_RATIO,
                "base_config": str(args.base_config) if args.base_config else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    records = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                run_case,
                repo,
                root,
                args.base_config or Path(f"configs/{case.chip}.json"),
                args.calibration,
                args.log_level,
                case,
                args.dry_run,
            ): case
            for case in cases
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            write_manifest(root, records)
            print(
                f"[{completed}/{len(cases)}] {record['case_id']} "
                f"rc={record['returncode']} wall={record.get('wall_seconds', 0):.1f}s",
                flush=True,
            )
    failures = [record for record in records if record["returncode"] != 0]
    (root / "run_complete.json").write_text(
        json.dumps(
            {
                "case_count": len(records),
                "failures": len(failures),
                "dry_run": args.dry_run,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
