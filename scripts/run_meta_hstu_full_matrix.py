#!/usr/bin/env python3
"""Run the aligned-NPU Meta HSTU speedup-comparison matrix."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


MODELS = {
    "small": (4, 256),
    "middle": (8, 512),
    "large": (12, 1024),
}
CHIP_CONFIGS = {
    "910A": Path("configs/910A.json"),
    "910B": Path("configs/910B.json"),
    "910C": Path("configs/910C.json"),
    "MTIA2": Path("configs/MTIA2.json"),
}
CHIPS = tuple(CHIP_CONFIGS)
DEFAULT_CHIPS = ("910A", "910B", "910C")
SEQS = (4096, 6144, 8192, 16384)
BATCHES = (1, 2, 4, 8)
USERS = ("hot", "cold")
# The requested speedup-comparison matrix has four methods.  IR recompute is
# still exercised as part of w_both, but is not a standalone matrix method.
METHODS = ("Full_Recompute", "Full_Cache", "w_AR", "w_both")
KV_REUSE_RATIO = 0.4802
DEFAULT_MAX_CONCURRENT = 48
MAX_CONCURRENT = 196
GIB = 1024 ** 3
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Concurrent cases (default: {DEFAULT_MAX_CONCURRENT}, max: {MAX_CONCURRENT}).",
    )
    parser.add_argument("--chips", nargs="+", choices=CHIPS, default=list(DEFAULT_CHIPS))
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
    parser.add_argument(
        "--chip-config",
        action="append",
        default=[],
        metavar="CHIP=PATH",
        help=(
            "Override one logical chip config; repeat for multiple chips. "
            "This is intended for generated reconfiguration configs."
        ),
    )
    parser.add_argument("--log-level", default="warn")
    parser.add_argument(
        "--lock-prefix", default="",
        help="Optional prefix for simulator slot locks when matrices overlap.",
    )
    parser.add_argument(
        "--max-simulator-rss-gib",
        type=float,
        default=None,
        help=(
            "Optional aggregate Simulator RSS limit. Cases reserve memory "
            "from their model/sequence/batch shape before launch."
        ),
    )
    parser.add_argument(
        "--max-total-simulators",
        type=int,
        default=None,
        help=(
            "Optional global Simulator-process cap, including Simulator jobs "
            "owned by other concurrently running experiment schedulers."
        ),
    )
    parser.add_argument(
        "--memory-headroom-gib",
        type=float,
        default=10.0,
        help="Memory kept below --max-simulator-rss-gib (default: 10 GiB).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class MemoryAdmissionController:
    """Reserve memory and CPU slots across concurrent experiment schedulers."""

    def __init__(
        self, limit_gib, headroom_gib, result_root, max_total_simulators=None
    ):
        self.limit_gib = limit_gib
        self.usable_gib = limit_gib - headroom_gib if limit_gib else None
        self.max_total_simulators = max_total_simulators
        self.result_marker = str(result_root)
        self.reserved_gib = 0.0
        self.active_cases = 0
        self.condition = threading.Condition()
        self.last_sample_time = 0.0
        self.last_external_rss_gib = 0.0
        self.last_external_simulators = 0

    @staticmethod
    def case_reservation_gib(case):
        """Conservative RSS envelope derived from the completed matrix runs."""

        hidden = MODELS[case.model][1]
        scaled_work = (
            (hidden / 1024.0) ** 2
            * (case.seq_len / 16384.0)
            * (case.batch_size / 8.0)
        )
        return 0.7 + 15.6 * scaled_work

    def sample_external_simulators(self):
        now = time.monotonic()
        if now - self.last_sample_time < 2.0:
            return self.last_external_rss_gib, self.last_external_simulators
        result = subprocess.run(
            ["ps", "-eo", "rss=,cmd="],
            text=True,
            capture_output=True,
            check=True,
        )
        rss_kib = 0
        count = 0
        for line in result.stdout.splitlines():
            if "./build/bin/Simulator " not in line:
                continue
            if self.result_marker in line:
                continue
            rss_kib += int(line.strip().split(None, 1)[0])
            count += 1
        self.last_external_rss_gib = rss_kib * 1024 / GIB
        self.last_external_simulators = count
        self.last_sample_time = now
        return self.last_external_rss_gib, self.last_external_simulators

    def acquire(self, case):
        if self.usable_gib is None:
            return 0.0
        reservation = self.case_reservation_gib(case)
        with self.condition:
            while True:
                external_rss, external_count = self.sample_external_simulators()
                memory_ok = (
                    self.usable_gib is None
                    or external_rss + self.reserved_gib + reservation
                    <= self.usable_gib
                )
                cpu_ok = (
                    self.max_total_simulators is None
                    or external_count + self.active_cases
                    < self.max_total_simulators
                )
                if memory_ok and cpu_ok:
                    self.reserved_gib += reservation
                    self.active_cases += 1
                    return reservation
                self.condition.wait(timeout=2.0)

    def release(self, reservation):
        if reservation <= 0.0:
            return
        with self.condition:
            self.reserved_gib = max(0.0, self.reserved_gib - reservation)
            self.active_cases = max(0, self.active_cases - 1)
            self.condition.notify_all()


def case_dir(root, case):
    return (
        root
        / "cases"
        / case.chip
        / case.method
        / f"HSTU-{case.model}_seq{case.seq_len}_bs{case.batch_size}_{case.user}"
    )


def parse_chip_config_overrides(values):
    overrides = {}
    for value in values:
        chip, separator, path = value.partition("=")
        if separator != "=" or chip not in CHIP_CONFIGS or not path:
            raise ValueError(
                f"invalid --chip-config {value!r}; expected CHIP=PATH"
            )
        overrides[chip] = Path(path)
    return overrides


def config_for_case(base_config, case, chip_configs=None):
    """Resolve the explicit override or the logical chip's aligned config."""
    if chip_configs and case.chip in chip_configs:
        return chip_configs[case.chip]
    return base_config or CHIP_CONFIGS[case.chip]


def matrix_cases(chips, models, seq_lens, batch_sizes, users, methods):
    return [
        Case(chip, model, seq, batch, user, method)
        for chip in chips
        for model in models
        for seq in seq_lens
        for batch in batch_sizes
        for user in users
        for method in methods
    ]


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
        command += [
            "--enable-kv-reuse",
            "--kv-reuse-ratio",
            str(KV_REUSE_RATIO),
            "--kv-reuse-reduce-npu",
        ]
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
    if case.method in {"w_AR", "w_both"}:
        command += ["--enable-ar-reduce-attention-compute"]
    else:
        command += ["--disable-ar-reduce-attention-compute"]
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


def completed_outputs_match_method(root, case):
    output = case_dir(root, case)
    if not all(
        (output / name).is_file()
        for name in (
            "runtime_config.json", "models.json", "layer_breakdown.csv",
            "hardware_summary.csv",
        )
    ):
        return False
    traces = sorted((output / "traces").glob("*.json"))
    if not traces:
        return False
    expected = case.method in {"w_AR", "w_both"}
    try:
        return all(
            json.loads(path.read_text(encoding="utf-8"))["metadata"]
            ["ar_reduce_attention_compute"] is expected
            for path in traces
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_case(
    repo, root, base_config, calibration, log_level, case, dry_run,
    admission=None, lock_tag=None,
):
    admission = admission or MemoryAdmissionController(None, 0.0, root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / f"{case.case_id}.status.json"

    started = time.time()
    record = {
        **asdict(case),
        "case_id": case.case_id,
        "started_unix": started,
        "ar_reduce_attention_compute": case.method in {"w_AR", "w_both"},
    }
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
                and completed_outputs_match_method(root, case)
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
            lock_name = lock_tag or case.case_id
            env["SIMULATOR_SLOT_LOCK"] = f"/tmp/{lock_name}.lock"
            env["MPLBACKEND"] = "Agg"
            env.update(SINGLE_THREAD_ENV)
            stdout_path = logs / f"{case.case_id}.stdout.log"
            stderr_path = logs / f"{case.case_id}.stderr.log"
            reservation = admission.acquire(case)
            record["memory_reservation_gib"] = reservation
            try:
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
            finally:
                admission.release(reservation)
            record["returncode"] = result.returncode
            if result.returncode == 0 and not completed_outputs_match_method(root, case):
                record["returncode"] = -2
                record["error"] = (
                    "completed output is missing or trace AR attention mode "
                    "does not match the method"
                )
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
    if not 1 <= args.max_concurrent <= MAX_CONCURRENT:
        raise SystemExit(f"--max-concurrent must be in [1, {MAX_CONCURRENT}]")
    if args.max_simulator_rss_gib is not None and (
        args.max_simulator_rss_gib <= 0
        or not 0 <= args.memory_headroom_gib < args.max_simulator_rss_gib
    ):
        raise SystemExit(
            "memory limit must be positive and larger than its headroom"
        )
    if args.max_total_simulators is not None and not (
        1 <= args.max_total_simulators <= MAX_CONCURRENT
    ):
        raise SystemExit(
            f"--max-total-simulators must be in [1, {MAX_CONCURRENT}]"
        )
    repo = Path(__file__).resolve().parents[1]
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    try:
        chip_config_overrides = parse_chip_config_overrides(args.chip_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.base_config is not None and len(args.chips) != 1:
        raise SystemExit("--base-config requires exactly one --chips value")
    if args.base_config is not None and chip_config_overrides:
        raise SystemExit("--base-config and --chip-config cannot be combined")
    for chip, path in chip_config_overrides.items():
        if chip not in args.chips:
            raise SystemExit(f"--chip-config supplied for unselected chip: {chip}")
        if not path.is_file():
            raise SystemExit(f"chip config not found: {path}")
    cases = matrix_cases(
        args.chips, args.models, args.seq_lens, args.batch_sizes,
        args.users, args.methods,
    )
    if args.max_simulator_rss_gib is not None:
        # Maximize useful CPU concurrency under the memory cap: complete
        # lower-memory shapes first while larger shapes wait for capacity.
        cases.sort(key=lambda case: (
            MemoryAdmissionController.case_reservation_gib(case),
            case.case_id,
        ))
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
                "max_simulator_rss_gib": args.max_simulator_rss_gib,
                "max_total_simulators": args.max_total_simulators,
                "memory_headroom_gib": args.memory_headroom_gib,
                "ar_reduce_attention_compute": False,
                "kv_reuse_ratio": KV_REUSE_RATIO,
                "chip_configs": {
                    chip: str(
                        chip_config_overrides.get(
                            chip, args.base_config or CHIP_CONFIGS[chip]
                        )
                    )
                    for chip in args.chips
                },
                "base_config": str(args.base_config) if args.base_config else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    records = []
    admission = MemoryAdmissionController(
        args.max_simulator_rss_gib,
        args.memory_headroom_gib,
        root,
        args.max_total_simulators,
    )
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                run_case,
                repo,
                root,
                config_for_case(
                    args.base_config, case, chip_config_overrides
                ),
                args.calibration,
                args.log_level,
                case,
                args.dry_run,
                admission,
                f"{args.lock_prefix}__{case.case_id}" if args.lock_prefix else None,
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
