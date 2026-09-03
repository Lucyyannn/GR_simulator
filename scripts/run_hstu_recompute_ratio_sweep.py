#!/usr/bin/env python3
"""Run the aligned-NPU HSTU AR+IR recompute-ratio sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


MODELS = {"small": (4, 256), "middle": (8, 512), "large": (12, 1024)}
CHIP_CONFIGS = {
    chip: Path(f"configs/{chip}.json")
    for chip in ("910A", "910B", "910C", "MTIA2")
}
CHIPS = tuple(CHIP_CONFIGS)
DEFAULT_CHIPS = ("910A", "910B", "910C")
SEQS = (4096, 6144, 8192, 16384)
BATCHES = (1, 2, 4, 8)
USERS = ("hot", "cold")
RATIOS = tuple(i / 10 for i in range(11))
KV_REUSE_RATIO = 0.4802
DEFAULT_MAX_CONCURRENT = 48
MAX_CONCURRENT = 196
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def round_half_up(value: float) -> int:
    """Round a nonnegative value to the nearest integer, resolving ties upward."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"expected a finite nonnegative value, got {value}")
    return int(math.floor(value + 0.5))


def ratio_tenths(ratio: float) -> int:
    tenths = round_half_up(ratio * 10)
    if not 0 <= tenths <= 10 or not math.isclose(ratio, tenths / 10, abs_tol=1e-9):
        raise ValueError(f"ratio must be one of 0, 0.1, ..., 1; got {ratio}")
    return tenths


@dataclass(frozen=True)
class Case:
    chip: str
    model: str
    seq_len: int
    batch_size: int
    user: str
    requested_ratio: float

    @property
    def layers(self) -> int:
        return MODELS[self.model][0]

    @property
    def hidden(self) -> int:
        return MODELS[self.model][1]

    @property
    def item_count(self) -> int:
        return (self.seq_len + 1) // 2

    @property
    def ratio_label(self) -> str:
        return f"r{ratio_tenths(self.requested_ratio) * 10:03d}"

    @property
    def history_recompute_len(self) -> int:
        # Do the half-up operation entirely in integers.  This avoids binary
        # floating-point behavior at an exact .5 boundary.
        tenths = ratio_tenths(self.requested_ratio)
        return min(self.item_count, (self.item_count * tenths + 5) // 10)

    @property
    def actual_ratio(self) -> float:
        return self.history_recompute_len / self.item_count

    @property
    def case_id(self) -> str:
        return (
            f"{self.chip}__{self.model}__seq{self.seq_len}__bs{self.batch_size}"
            f"__{self.user}__AR_IR__{self.ratio_label}"
        )


def matrix_cases(chips, models, seq_lens, batch_sizes, users, ratios):
    normalized_ratios = sorted({ratio_tenths(float(ratio)) for ratio in ratios})
    return [
        Case(chip, model, seq_len, batch_size, user, tenths / 10)
        for chip in chips
        for model in models
        for seq_len in seq_lens
        for batch_size in batch_sizes
        for user in users
        for tenths in normalized_ratios
    ]


def case_dir(root: Path, case: Case) -> Path:
    return (
        root / "cases" / case.chip / "AR_IR_ratio" / case.ratio_label
        / f"HSTU-{case.model}_seq{case.seq_len}_bs{case.batch_size}_{case.user}"
    )


def build_command(root: Path, config: Path, log_level: str, case: Case) -> list[str]:
    history_source = "ddr" if case.user == "hot" else "ssd"
    return [
        "bash", "scripts/run_hstu.sh",
        "--base-config", str(config),
        "--result-dir", str(case_dir(root, case)),
        "--source-medium", history_source,
        "--embedding-source-medium", "ssd",
        "--history-recompute-source-medium", history_source,
        "--layers", str(case.layers),
        "--hidden", str(case.hidden),
        "--kv-len", str(case.seq_len),
        "--history-recompute-len", str(case.history_recompute_len),
        "--history-recompute-index-mode", "continuous",
        "--num-users", str(case.batch_size),
        "--users-per-batch", str(case.batch_size),
        "--candidates-per-user", "128",
        "--macro-batch-size", "128",
        "--vocab", "262144",
        "--attention-modeling", "fused",
        "--enable-kv-reuse",
        "--kv-reuse-ratio", str(KV_REUSE_RATIO),
        "--enable-ar-reduce-attention-compute",
        "--log-level", log_level,
    ]


def file_digest(repo: Path, path: Path) -> str | None:
    resolved = path if path.is_absolute() else repo / path
    if not resolved.exists():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digests(repo: Path, config: Path) -> dict[str, str | None]:
    paths = {
        "base_config": config,
        "runner": Path("scripts/run_hstu_recompute_ratio_sweep.py"),
        "run_hstu": Path("scripts/run_hstu.sh"),
        "trace_generator": Path("scripts/generate_hstu_baseline_trace.py"),
        "simulator": Path("build/bin/Simulator"),
    }
    return {name: file_digest(repo, path) for name, path in paths.items()}


def record_for(case: Case) -> dict:
    return {
        **asdict(case),
        "case_id": case.case_id,
        "layers": case.layers,
        "hidden": case.hidden,
        "item_count": case.item_count,
        "ratio_label": case.ratio_label,
        "actual_ratio": case.actual_ratio,
        "history_recompute_len": case.history_recompute_len,
        "kv_reuse_ratio": KV_REUSE_RATIO,
        "ar_reduce_attention_compute": True,
        "source_medium": "ddr" if case.user == "hot" else "ssd",
        "embedding_source_medium": "ssd",
        "history_recompute_source_medium": "ddr" if case.user == "hot" else "ssd",
    }


def completed_outputs(root: Path, case: Case) -> bool:
    output = case_dir(root, case)
    required = all(
        (output / name).is_file()
        for name in (
            "runtime_config.json", "models.json", "layer_breakdown.csv",
            "hardware_summary.csv",
        )
    )
    traces = sorted((output / "traces").glob("*.json"))
    if not required or not traces:
        return False
    try:
        return all(
            json.loads(path.read_text(encoding="utf-8"))["metadata"]
            ["ar_reduce_attention_compute"] is True
            for path in traces
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_case(
    repo: Path,
    root: Path,
    config: Path,
    log_level: str,
    case: Case,
    dry_run: bool,
) -> dict:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / f"{case.case_id}.status.json"
    record = record_for(case)
    record["started_unix"] = time.time()
    command = build_command(root, config, log_level, case)
    record["command"] = command
    record["input_digests"] = input_digests(repo, config)

    if status_path.exists() and not dry_run:
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            old.get("returncode") == 0
            and not old.get("dry_run", False)
            and old.get("command") == command
            and old.get("input_digests") == record["input_digests"]
            and completed_outputs(root, case)
        ):
            return old

    if dry_run:
        record["returncode"] = 0
        record["dry_run"] = True
    else:
        try:
            env = os.environ.copy()
            env.update(SINGLE_THREAD_ENV)
            env["WAIT_IF_SIMULATOR_RUNNING"] = "0"
            env["SIMULATOR_SLOT_LOCK"] = f"/tmp/{case.case_id}.lock"
            env["MPLBACKEND"] = "Agg"
            stdout_path = logs / f"{case.case_id}.stdout.log"
            stderr_path = logs / f"{case.case_id}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, \
                 stderr_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    command, cwd=repo, env=env, stdout=stdout, stderr=stderr,
                    check=False,
                )
            record["returncode"] = result.returncode
            if result.returncode == 0 and not completed_outputs(root, case):
                record["returncode"] = -2
                record["error"] = (
                    "completed output is missing or AR attention reduction "
                    "is not enabled in every trace"
                )
        except Exception as exc:  # keep independent sweep points running
            record["returncode"] = -1
            record["error"] = repr(exc)

    record["finished_unix"] = time.time()
    record["wall_seconds"] = record["finished_unix"] - record["started_unix"]
    if not dry_run:
        status_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


MANIFEST_FIELDS = [
    "case_id", "chip", "model", "layers", "hidden", "seq_len",
    "batch_size", "user", "ratio_label", "requested_ratio", "actual_ratio",
    "item_count", "history_recompute_len", "kv_reuse_ratio",
    "ar_reduce_attention_compute", "source_medium", "embedding_source_medium",
    "history_recompute_source_medium", "returncode", "wall_seconds",
]


def write_manifest(root: Path, records: list[dict]) -> None:
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["case_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--chips", nargs="+", choices=CHIPS, default=list(DEFAULT_CHIPS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seq-lens", nargs="+", type=int, choices=SEQS, default=list(SEQS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=BATCHES, default=list(BATCHES))
    parser.add_argument("--users", nargs="+", choices=USERS, default=list(USERS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument(
        "--chip-config", action="append", default=[], metavar="CHIP=PATH",
        help="Override one selected chip config; repeat for multiple chips.",
    )
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise SystemExit(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    overrides = {}
    for value in args.chip_config:
        chip, separator, path = value.partition("=")
        if separator != "=" or chip not in CHIPS or not path:
            raise SystemExit(
                f"invalid --chip-config {value!r}; expected CHIP=PATH"
            )
        overrides[chip] = Path(path)
    for chip, path in overrides.items():
        if chip not in args.chips:
            raise SystemExit(f"override supplied for unselected chip: {chip}")
        if not path.is_file():
            raise SystemExit(f"chip config not found: {path}")
    configs = {chip: overrides.get(chip, CHIP_CONFIGS[chip]) for chip in args.chips}
    try:
        cases = matrix_cases(
            args.chips, args.models, args.seq_lens, args.batch_sizes,
            args.users, args.ratios,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    repo = Path(__file__).resolve().parents[1]
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix_definition.json").write_text(
        json.dumps({
            "case_count": len(cases),
            "chips": args.chips,
            "chip_configs": {chip: str(configs[chip]) for chip in args.chips},
            "models": {name: MODELS[name] for name in args.models},
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "users": args.users,
            "requested_ratios": sorted({case.requested_ratio for case in cases}),
            "ratio_semantics": "fraction of item tokens; item_count=(seq_len+1)//2; deterministic half-up k",
            "kv_reuse_enabled": True,
            "kv_reuse_ratio": KV_REUSE_RATIO,
            "ar_reduce_attention_compute": True,
            "candidate_embedding_source": "ssd",
            "history_sources": {"hot": "ddr", "cold": "ssd"},
            "max_concurrent": args.max_concurrent,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                run_case, repo, root, configs[case.chip], args.log_level,
                case, args.dry_run,
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
        json.dumps({
            "case_count": len(records),
            "failures": len(failures),
            "dry_run": args.dry_run,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
