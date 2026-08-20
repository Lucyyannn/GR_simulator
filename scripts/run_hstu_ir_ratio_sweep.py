#!/usr/bin/env python3
"""Run the representative SSD HSTU IR ratio sweep without overwriting results."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


KV_REUSE_RATIO = 0.4360
MODEL_SHAPES = {
    "small": (4, 256),
    "middle": (8, 512),
    "large": (12, 1024),
}

# L9(3^4) orthogonal array. Every pair of chip/model/sequence/batch levels
# occurs exactly once. Level labels are permuted to avoid combining all three
# largest workload dimensions in one case.
DEFAULT_CONTEXTS = (
    ("oa01", "910C", "small", 4096, 4),
    ("oa02", "910C", "middle", 8192, 8),
    ("oa03", "910C", "large", 16384, 1),
    ("oa04", "910B", "small", 8192, 1),
    ("oa05", "910B", "middle", 16384, 4),
    ("oa06", "910B", "large", 4096, 8),
    ("oa07", "910A", "small", 16384, 8),
    ("oa08", "910A", "middle", 4096, 1),
    ("oa09", "910A", "large", 8192, 4),
)


@dataclass(frozen=True)
class Context:
  context_id: str
  chip: str
  model: str
  seq_len: int
  batch_size: int

  @property
  def layers(self) -> int:
    return MODEL_SHAPES[self.model][0]

  @property
  def hidden(self) -> int:
    return MODEL_SHAPES[self.model][1]

  @property
  def item_count(self) -> int:
    return (self.seq_len + 1) // 2

  @property
  def name(self) -> str:
    return (
        f"{self.context_id}_{self.chip}_{self.model}_"
        f"seq{self.seq_len}_bs{self.batch_size}"
    )


@dataclass(frozen=True)
class SweepCase:
  context: Context
  method: str
  point_kind: str
  point_label: str
  history_recompute_len: int
  requested_ratio: float | None
  model_prediction: dict

  @property
  def actual_ratio(self) -> float:
    return self.history_recompute_len / self.context.item_count

  @property
  def case_id(self) -> str:
    return f"{self.context.name}__{self.method}__{self.point_label}"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--result-root", type=Path, required=True)
  parser.add_argument("--max-concurrent", type=int, default=48)
  parser.add_argument(
      "--ratios", nargs="+", type=float,
      default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  )
  parser.add_argument(
      "--methods", nargs="+", choices=["w_IR", "w_both"],
      default=["w_IR", "w_both"],
  )
  parser.add_argument(
      "--contexts", nargs="+",
      choices=[row[0] for row in DEFAULT_CONTEXTS],
      default=[row[0] for row in DEFAULT_CONTEXTS],
  )
  parser.add_argument(
      "--calibration", type=Path,
      default=Path("scripts/recompute_ratio_calibration.json"),
  )
  parser.add_argument(
      "--objective", choices=["balance", "steady", "pipeline", "e2e"],
      default="balance",
      help="Objective used only for estimate points.",
  )
  parser.add_argument(
      "--include-estimate", action="store_true",
      help="Add one exact estimator-selected point per context and method.",
  )
  parser.add_argument(
      "--estimate-only", action="store_true",
      help="Run estimator-selected points without the ratio grid.",
  )
  parser.add_argument("--estimate-label", default="current_estimate")
  parser.add_argument(
      "--plan-csv", type=Path, default=None,
      help=(
          "Run explicit context_id,method,ratio rows instead of the Cartesian "
          "ratio matrix. Intended for local fine sweeps."
      ),
  )
  parser.add_argument("--log-level", default="warn")
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def round_half_up(value: float) -> int:
  return int(math.floor(value + 0.5))


def ratio_label(ratio: float) -> str:
  return "r" + f"{ratio:.4f}".replace(".", "p")


def model_command(
    context: Context,
    method: str,
    calibration: Path,
    objective: str,
    fixed_k: int | None,
) -> list[str]:
  command = [
      "python3",
      "scripts/recompute_ratio_cost_model_new.py",
      "--config",
      f"configs/{context.chip}.json",
      "--calibration",
      str(calibration),
      "--user",
      "cold",
      "--layers",
      str(context.layers),
      "--hidden",
      str(context.hidden),
      "--kv-len",
      str(context.seq_len),
      "--batch-size",
      str(context.batch_size),
      "--candidates",
      "128",
      "--embedding-source",
      "ssd",
      "--objective",
      objective,
      "--field",
      "json",
  ]
  if fixed_k is not None:
    command += ["--fixed-recompute-len", str(fixed_k)]
  if method == "w_both":
    command += [
        "--enable-kv-reuse",
        "--kv-reuse-ratio",
        str(KV_REUSE_RATIO),
        "--no-kv-reuse-reduce-npu",
    ]
  return command


def predict(
    repo: Path,
    context: Context,
    method: str,
    calibration: Path,
    objective: str,
    fixed_k: int | None,
) -> dict:
  result = subprocess.run(
      model_command(context, method, calibration, objective, fixed_k),
      cwd=repo,
      text=True,
      capture_output=True,
      check=True,
  )
  return json.loads(result.stdout)


def selected_contexts(ids: list[str]) -> list[Context]:
  wanted = set(ids)
  return [Context(*row) for row in DEFAULT_CONTEXTS if row[0] in wanted]


def build_cases(args: argparse.Namespace, repo: Path) -> list[SweepCase]:
  contexts = selected_contexts(args.contexts)
  cases: list[SweepCase] = []
  if args.plan_csv is not None:
    context_lookup = {context.context_id: context for context in contexts}
    with args.plan_csv.open(newline="", encoding="utf-8") as handle:
      plan_rows = list(csv.DictReader(handle))
    seen: set[tuple[str, str, int]] = set()
    for row in plan_rows:
      context_id = row["context_id"]
      method = row["method"]
      if context_id not in context_lookup:
        raise SystemExit(f"plan context {context_id} is not enabled")
      if method not in args.methods:
        raise SystemExit(f"plan method {method} is not enabled")
      context = context_lookup[context_id]
      ratio = float(row["ratio"])
      if ratio < 0.0 or ratio > 1.0:
        raise SystemExit(f"ratio must be in [0, 1], got {ratio}")
      k = min(context.item_count, round_half_up(ratio * context.item_count))
      key = (context_id, method, k)
      if key in seen:
        continue
      seen.add(key)
      prediction = predict(
          repo, context, method, args.calibration, "e2e", k
      )
      label = f"fine_r{ratio:.4f}".replace(".", "p")
      cases.append(SweepCase(
          context=context,
          method=method,
          point_kind="fine_grid",
          point_label=label,
          history_recompute_len=k,
          requested_ratio=ratio,
          model_prediction=prediction,
      ))
    return cases
  if not args.estimate_only:
    ratios = sorted(set(args.ratios))
    for ratio in ratios:
      if ratio < 0.0 or ratio > 1.0:
        raise SystemExit(f"ratio must be in [0, 1], got {ratio}")
    for context in contexts:
      for method in args.methods:
        for ratio in ratios:
          k = min(context.item_count, round_half_up(ratio * context.item_count))
          prediction = predict(
              repo, context, method, args.calibration, "e2e", k
          )
          cases.append(SweepCase(
              context=context,
              method=method,
              point_kind="grid",
              point_label=ratio_label(ratio),
              history_recompute_len=k,
              requested_ratio=ratio,
              model_prediction=prediction,
          ))
  if args.include_estimate or args.estimate_only:
    for context in contexts:
      for method in args.methods:
        prediction = predict(
            repo, context, method, args.calibration, args.objective, None
        )
        cases.append(SweepCase(
            context=context,
            method=method,
            point_kind="estimate",
            point_label=args.estimate_label,
            history_recompute_len=int(prediction["history_recompute_len"]),
            requested_ratio=None,
            model_prediction=prediction,
        ))
  return cases


def case_dir(root: Path, case: SweepCase) -> Path:
  return root / "cases" / case.context.name / case.method / case.point_label


def simulator_command(
    root: Path,
    log_level: str,
    case: SweepCase,
) -> list[str]:
  context = case.context
  command = [
      "bash",
      "scripts/run_hstu.sh",
      "--base-config",
      f"configs/{context.chip}.json",
      "--result-dir",
      str(case_dir(root, case)),
      "--source-medium",
      "ssd",
      "--embedding-source-medium",
      "ssd",
      "--history-recompute-source-medium",
      "ssd",
      "--layers",
      str(context.layers),
      "--hidden",
      str(context.hidden),
      "--kv-len",
      str(context.seq_len),
      "--history-recompute-len",
      str(case.history_recompute_len),
      "--history-recompute-index-mode",
      "continuous",
      "--num-users",
      str(context.batch_size),
      "--users-per-batch",
      str(context.batch_size),
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
  if case.method == "w_both":
    command += [
        "--enable-kv-reuse",
        "--kv-reuse-ratio",
        str(KV_REUSE_RATIO),
    ]
  return command


def status_record(case: SweepCase) -> dict:
  return {
      "case_id": case.case_id,
      "context_id": case.context.context_id,
      "chip": case.context.chip,
      "model": case.context.model,
      "layers": case.context.layers,
      "hidden": case.context.hidden,
      "seq_len": case.context.seq_len,
      "batch_size": case.context.batch_size,
      "method": case.method,
      "point_kind": case.point_kind,
      "point_label": case.point_label,
      "requested_ratio": case.requested_ratio,
      "history_recompute_len": case.history_recompute_len,
      "actual_ratio": case.actual_ratio,
      "source_medium": "ssd",
      "embedding_source_medium": "ssd",
      "history_recompute_source_medium": "ssd",
      "kv_reuse_ratio": KV_REUSE_RATIO if case.method == "w_both" else 0.0,
      "ar_reduce_attention_compute": False,
  }


def file_digest(repo: Path, path: Path) -> str | None:
  resolved = path if path.is_absolute() else repo / path
  if not resolved.exists():
    return None
  digest = hashlib.sha256()
  with resolved.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def input_digests(repo: Path, case: SweepCase) -> dict[str, str | None]:
  paths = {
      "base_config": Path(f"configs/{case.context.chip}.json"),
      "run_hstu": Path("scripts/run_hstu.sh"),
      "trace_generator": Path("scripts/generate_hstu_baseline_trace.py"),
      "simulator": Path("build/bin/Simulator"),
  }
  return {name: file_digest(repo, path) for name, path in paths.items()}


def reusable_measurement(
    root: Path,
    case: SweepCase,
    current_digests: dict[str, str | None],
) -> tuple[dict, Path] | None:
  """Find a completed, simulator-equivalent point in this result root.

  Estimate points often land exactly on a grid point.  Re-running that point
  is both expensive and unnecessary: the generated simulator workload is
  identical because the point label only changes output paths.  Keep the new
  estimator prediction/status, but explicitly reference the original
  measurement so analysis retains provenance.
  """
  wanted = status_record(case)
  equivalence_fields = [
      "context_id", "chip", "model", "layers", "hidden", "seq_len",
      "batch_size", "method", "history_recompute_len", "source_medium",
      "embedding_source_medium", "history_recompute_source_medium",
      "kv_reuse_ratio", "ar_reduce_attention_compute",
  ]
  logs = root / "logs"
  for path in sorted(logs.glob("*.status.json")):
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("case_id") == case.case_id:
      continue
    if int(candidate.get("returncode", -1)) != 0:
      continue
    if candidate.get("input_digests") != current_digests:
      continue
    if not all(candidate.get(key) == wanted.get(key) for key in equivalence_fields):
      continue
    candidate_dir = (
        root / "cases" /
        f"{candidate['context_id']}_{candidate['chip']}_{candidate['model']}_"
        f"seq{candidate['seq_len']}_bs{candidate['batch_size']}" /
        candidate["method"] / candidate["point_label"]
    )
    required = [
        candidate_dir / "runtime_config.json",
        candidate_dir / "models.json",
        candidate_dir / "compute_activity.csv",
        candidate_dir / "layer_breakdown.csv",
    ]
    if all(item.exists() for item in required):
      return candidate, candidate_dir
  return None


def run_case(
    repo: Path,
    root: Path,
    log_level: str,
    case: SweepCase,
    dry_run: bool,
) -> dict:
  """Serialize duplicate submissions of one case, while keeping cases parallel."""
  logs = root / "logs"
  logs.mkdir(parents=True, exist_ok=True)
  lock_path = logs / f"{case.case_id}.runner.lock"
  with lock_path.open("w", encoding="utf-8") as lock_handle:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return run_case_locked(repo, root, log_level, case, dry_run)


def run_case_locked(
    repo: Path,
    root: Path,
    log_level: str,
    case: SweepCase,
    dry_run: bool,
) -> dict:
  logs = root / "logs"
  logs.mkdir(parents=True, exist_ok=True)
  status_path = logs / f"{case.case_id}.status.json"
  prediction_path = logs / f"{case.case_id}.model_prediction.json"
  prediction_path.write_text(
      json.dumps(case.model_prediction, indent=2) + "\n", encoding="utf-8"
  )
  record = status_record(case)
  record["started_unix"] = time.time()
  command = simulator_command(root, log_level, case)
  record["command"] = command
  record["input_digests"] = input_digests(repo, case)
  if status_path.exists() and not dry_run:
    old = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        old.get("returncode") == 0
        and not old.get("dry_run", False)
        and old.get("command") == command
        and old.get("input_digests") == record["input_digests"]
    ):
      return old
  reuse = (
      None if dry_run else
      reusable_measurement(root, case, record["input_digests"])
  )
  if reuse is not None:
    source, source_dir = reuse
    record.update({
        "returncode": 0,
        "measurement_reused": True,
        "measurement_source_case_id": source["case_id"],
        "measurement_case_dir": str(source_dir.resolve()),
        "reuse_reason": "identical simulator inputs; only output paths differ",
        "finished_unix": time.time(),
    })
    record["wall_seconds"] = record["finished_unix"] - record["started_unix"]
    status_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record
  if dry_run:
    record.update({
        "returncode": 0,
        "dry_run": True,
        "finished_unix": time.time(),
    })
    record["wall_seconds"] = record["finished_unix"] - record["started_unix"]
    return record
  try:
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
  except Exception as exc:  # keep independent sweep points running
    record["returncode"] = -1
    record["error"] = repr(exc)
  record["finished_unix"] = time.time()
  record["wall_seconds"] = record["finished_unix"] - record["started_unix"]
  status_path.write_text(
      json.dumps(record, indent=2) + "\n", encoding="utf-8"
  )
  return record


MANIFEST_FIELDS = [
    "case_id", "context_id", "chip", "model", "layers", "hidden",
    "seq_len", "batch_size", "method", "point_kind", "point_label",
    "requested_ratio", "history_recompute_len", "actual_ratio",
    "source_medium", "embedding_source_medium",
    "history_recompute_source_medium", "kv_reuse_ratio",
    "ar_reduce_attention_compute", "returncode", "wall_seconds",
    "measurement_reused", "measurement_source_case_id",
]


def write_manifest(root: Path, records: list[dict]) -> None:
  path = root / "manifest.csv"
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(sorted(records, key=lambda row: row["case_id"]))


def existing_records(root: Path) -> dict[str, dict]:
  records: dict[str, dict] = {}
  for path in (root / "logs").glob("*.status.json") if (root / "logs").exists() else []:
    record = json.loads(path.read_text(encoding="utf-8"))
    records[record["case_id"]] = record
  return records


def main() -> None:
  args = parse_args()
  if not 1 <= args.max_concurrent <= 48:
    raise SystemExit("--max-concurrent must be in [1, 48]")
  repo = Path(__file__).resolve().parents[1]
  root = args.result_root
  root.mkdir(parents=True, exist_ok=True)
  cases = build_cases(args, repo)
  definition = {
      "contexts": [asdict(context) for context in selected_contexts(args.contexts)],
      "methods": args.methods,
      "ratios": None if args.estimate_only else sorted(set(args.ratios)),
      "plan_csv": str(args.plan_csv) if args.plan_csv is not None else None,
      "include_estimate": args.include_estimate or args.estimate_only,
      "estimate_label": args.estimate_label,
      "objective": args.objective,
      "calibration": str(args.calibration),
      "case_count": len(cases),
      "max_concurrent": args.max_concurrent,
      "ssd_only": True,
      "kv_reuse_ratio": KV_REUSE_RATIO,
      "ar_reduce_attention_compute": False,
  }
  definition_path = root / f"matrix_definition_{args.estimate_label}.json"
  definition_path.write_text(
      json.dumps(definition, indent=2) + "\n", encoding="utf-8"
  )

  all_records = existing_records(root)
  with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
    futures = {
        executor.submit(
            run_case, repo, root, args.log_level, case, args.dry_run
        ): case
        for case in cases
    }
    for completed, future in enumerate(as_completed(futures), start=1):
      record = future.result()
      all_records[record["case_id"]] = record
      write_manifest(root, list(all_records.values()))
      print(
          f"[{completed}/{len(cases)}] {record['case_id']} "
          f"k={record['history_recompute_len']} "
          f"rc={record['returncode']} "
          f"wall={record.get('wall_seconds', 0):.1f}s",
          flush=True,
      )

  selected_ids = {case.case_id for case in cases}
  selected_records = [all_records[case_id] for case_id in selected_ids]
  failures = [row for row in selected_records if row.get("returncode") != 0]
  complete_path = root / f"run_complete_{args.estimate_label}.json"
  complete_path.write_text(
      json.dumps({
          "case_count": len(selected_records),
          "failures": len(failures),
          "dry_run": args.dry_run,
      }, indent=2) + "\n",
      encoding="utf-8",
  )
  raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
  main()
