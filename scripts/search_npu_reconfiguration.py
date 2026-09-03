#!/usr/bin/env python3
"""Exhaustive GRACE Cube/Vector reconfiguration search for w_both.

Every feasible (N_c, N_v, W_v) design is checked against the baseline NPU's
compute-unit area budget and, optionally, its power budget.  For every request shape, the paper
item-KV equations select an integer recompute count k.  The optimization
objective is the geometric mean per-request latency over the full HSTU
small/middle/large, seq=4K/6K/8K, batch=1/2/4, hot/cold workload matrix.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import npu_config
import recompute_ratio_cost_model_new as estimator


MODELS = {"small": (4, 256), "middle": (8, 512), "large": (12, 1024)}
SEQUENCES = (4096, 6144, 8192)
BATCHES = (1, 2, 4)
USERS = ("hot", "cold")
KV_REUSE_RATIO = 0.4802


@dataclass(frozen=True)
class Workload:
    model: str
    layers: int
    hidden: int
    sequence: int
    batch: int
    user: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines", nargs="+", type=Path,
        default=[
            Path("configs/910A.json"), Path("configs/910B.json"),
            Path("configs/910C.json"), Path("configs/MTIA2.json"),
        ],
    )
    parser.add_argument("--calibration", type=Path, default=Path("configs/item_kv_calib.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nc-min", type=int, default=1)
    parser.add_argument("--nc-max", type=int, default=64)
    parser.add_argument("--nc-step", type=int, default=1)
    parser.add_argument("--nv-min", type=int, default=1)
    parser.add_argument("--nv-max", type=int, default=128)
    parser.add_argument("--nv-step", type=int, default=1)
    parser.add_argument("--wv-min", type=int, default=1024)
    parser.add_argument("--wv-max", type=int, default=8192)
    parser.add_argument("--wv-step", type=int, default=1024)
    parser.add_argument("--cube-compression", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--min-area-utilization", type=float, default=0.0,
        help="Discard feasible points using less than this fraction of baseline area.",
    )
    parser.add_argument(
        "--ignore-power", action="store_true",
        help="Do not filter by power; estimated power is still reported.",
    )
    parser.add_argument(
        "--require-vector-multiple", action="store_true",
        help="Keep only designs whose Vector-core count is an integer multiple of Cube cores.",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seq-lens", nargs="+", type=int, choices=SEQUENCES, default=list(SEQUENCES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=BATCHES, default=list(BATCHES))
    parser.add_argument("--users", nargs="+", choices=USERS, default=list(USERS))
    parser.add_argument(
        "--min-predicted-speedup", type=float, default=0.0,
        help=(
            "Require every workload to meet this predicted speedup versus "
            "the baseline; 1.0 is a baseline-safe search."
        ),
    )
    return parser.parse_args()


def workload_matrix(
    models: list[str] | tuple[str, ...] = tuple(MODELS),
    sequences: list[int] | tuple[int, ...] = SEQUENCES,
    batches: list[int] | tuple[int, ...] = BATCHES,
    users: list[str] | tuple[str, ...] = USERS,
) -> list[Workload]:
    return [
        Workload(model, layers, hidden, sequence, batch, user)
        for model in models
        for layers, hidden in (MODELS[model],)
        for sequence in sequences
        for batch in batches
        for user in users
    ]


def hardware_profile(calibration: dict, chip: str, medium: str, batch: int) -> dict:
    try:
        return (
            calibration["paper_cost_model"]["hardware_profiles"]["chips"]
            [chip]["media"][medium]["batches"][str(batch)]
        )
    except KeyError as exc:
        raise KeyError(
            f"missing hardware calibration for {chip}/{medium}/bs{batch}"
        ) from exc


def utilization(work: np.ndarray, saturation: float) -> np.ndarray:
    if saturation == 0.0:
        return np.ones_like(work)
    value = -np.expm1(-work / saturation)
    return np.maximum(value, 1e-15)


def fixed_path_time(
    work: np.ndarray,
    raw_rate: float,
    eta: float,
    saturation: float,
    startup_s: float,
    request_count: int,
) -> np.ndarray:
    body = np.divide(
        work,
        raw_rate * eta * utilization(work / request_count, saturation),
        out=np.zeros_like(work),
        where=work > 0.0,
    )
    return body + np.where(work > 0.0, request_count * startup_s, 0.0)


def variable_path_numerator(
    work: np.ndarray,
    eta: float,
    saturation: float,
    request_count: int,
) -> np.ndarray:
    return np.divide(
        work,
        eta * utilization(work / request_count, saturation),
        out=np.zeros_like(work),
        where=work > 0.0,
    )


class WorkloadCost:
    """Vectorized form of the paper equations and exact integer-k decision."""

    def __init__(self, workload: Workload, hw: dict, profile: dict):
        self.workload = workload
        S = workload.sequence
        Si = (S + 1) // 2
        original_actions = S // 2
        action_reuse = estimator.action_reuse_ratio_from_total(
            S, original_actions, KV_REUSE_RATIO
        )
        Sa = estimator.compressed_rows_for_ratio(original_actions, action_reuse)
        # AR reuses repeated Action rows for both KV movement and QK/AV work.
        # Therefore the attention length is the post-AR history S_att=S_h.
        Sh = Si + Sa
        Satt = Sh
        C = 128
        H = workload.hidden
        L = workload.layers
        N = workload.batch
        s = hw["s"]
        requests = L * N
        k = np.arange(Si + 1, dtype=np.float64)
        self.k = k

        kv_work = 2.0 * L * N * (Sa + Si - k) * H * s
        emb_work = N * k * H * s
        cube_work = L * N * (
            8.0 * C * H * H + 4.0 * C * Satt * H
            + 8.0 * k * H * H + 4.0 * k * k * H
        )
        vec_work = L * N * (
            2.0 * (C * Satt + k * k) + 2.0 * (C + k) * H
        )
        core_work = L * N * s * (
            (C + k) * H + 2.0 * Sh * H + 4.0 * k * H
        )

        medium = "ddr" if workload.user == "hot" else "ssd"
        self.memory = (
            fixed_path_time(
                kv_work, hw[f"B_{medium}"], profile["eta_kv"],
                profile["kv_saturation_bytes"], profile["kv_startup_s"], requests,
            )
            + fixed_path_time(
                emb_work, hw[f"B_{medium}"], profile["eta_emb"],
                profile["emb_saturation_bytes"], profile["emb_startup_s"], N,
            )
        )
        self.cube_numerator = variable_path_numerator(
            cube_work, profile["eta_cube"],
            profile["cube_saturation_flops"], requests,
        )
        self.vec_numerator = variable_path_numerator(
            vec_work, profile["eta_vec"],
            profile["vec_saturation_ops"], requests,
        )
        self.fixed_npu = (
            fixed_path_time(
                core_work, hw["B_core"], profile["eta_core"],
                profile["core_saturation_bytes"], profile["core_startup_s"], requests,
            )
            + requests * (profile["cube_startup_s"] + profile["vec_startup_s"])
        )
        self.memory_min_k = int(np.argmin(self.memory))
        # k=0 has no embedding request, while k=1 pays its startup once.  That
        # discrete boundary may create a one-step jump.  The positive-k branch
        # is the continuous cost-model path used for the crossing search; k=0
        # is evaluated explicitly below.
        positive_start = 1 if self.memory_min_k > 0 else 0
        if np.any(
            np.diff(self.memory[positive_start : self.memory_min_k + 1]) > 1e-15
        ):
            raise ValueError(f"non-unimodal memory path for {workload}")
        for name, values in (
            ("cube", self.cube_numerator),
            ("vector", self.vec_numerator),
            ("fixed_npu", self.fixed_npu),
        ):
            if np.any(np.diff(values) < -1e-15):
                raise ValueError(f"non-monotone {name} path for {workload}")

    def select(self, cube_rates: np.ndarray, vector_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute argmin_k max(T_mem,T_npu) without a ratio grid.

        T_npu is monotone increasing.  T_mem is decreasing up to its global
        minimum, so the optimum is the memory minimum or adjacent to the first
        T_npu/T_mem crossing.  Binary search therefore gives the exact integer
        decision while evaluating only O(log S_i) points per design.
        """

        cube_rates = np.asarray(cube_rates, dtype=np.float64)
        vector_rates = np.asarray(vector_rates, dtype=np.float64)

        def npu_at(indices: np.ndarray) -> np.ndarray:
            return (
                self.cube_numerator[indices] / cube_rates
                + self.vec_numerator[indices] / vector_rates
                + self.fixed_npu[indices]
            )

        if self.memory_min_k == 0:
            zero = np.zeros(cube_rates.shape, dtype=np.int32)
            latency = np.maximum(self.memory[0], npu_at(zero))
            return zero, latency

        lo = np.ones(cube_rates.shape, dtype=np.int32)
        hi = np.full(cube_rates.shape, self.memory_min_k, dtype=np.int32)

        while np.any(lo < hi):
            mid = (lo + hi) // 2
            crossed = npu_at(mid) >= self.memory[mid]
            hi = np.where(crossed, mid, hi)
            lo = np.where(crossed, lo, mid + 1)

        crossing = lo
        candidates = np.stack(
            (
                crossing,
                np.maximum(crossing - 1, 0),
                np.full(crossing.shape, self.memory_min_k, dtype=np.int32),
                np.zeros(crossing.shape, dtype=np.int32),
            ),
            axis=0,
        )
        best_time = np.full(cube_rates.shape, np.inf)
        best_k = np.zeros(cube_rates.shape, dtype=np.int32)
        for indices in candidates:
            latency = np.maximum(self.memory[indices], npu_at(indices))
            better = (latency < best_time) | (
                (latency == best_time) & (indices < best_k)
            )
            best_time = np.where(better, latency, best_time)
            best_k = np.where(better, indices, best_k)
        return best_k, best_time

    def latency_at(
        self, k: int, cube_rate: float, vector_rate: float
    ) -> float:
        """Evaluate the visible paper formula at one already-selected k."""

        return float(max(
            self.memory[k],
            self.cube_numerator[k] / cube_rate
            + self.vec_numerator[k] / vector_rate
            + self.fixed_npu[k],
        ))


def candidate_space(args: argparse.Namespace, budget: npu_config.ResourceUsage):
    nc_values = np.arange(args.nc_min, args.nc_max + 1, args.nc_step, dtype=np.int16)
    nv_values = np.arange(args.nv_min, args.nv_max + 1, args.nv_step, dtype=np.int16)
    wv_values = np.arange(args.wv_min, args.wv_max + 1, args.wv_step, dtype=np.int16)
    nc, nv, wv = np.meshgrid(nc_values, nv_values, wv_values, indexing="ij")
    nc, nv, wv = nc.ravel(), nv.ravel(), wv.ravel()
    area = (
        nc * npu_config.CUBE_AREA_MM2
        + nv * npu_config.VECTOR_AREA_MM2
        * (wv / npu_config.REFERENCE_VECTOR_WIDTH_BITS)
    )
    power = (
        nc * npu_config.CUBE_POWER_W
        + nv * npu_config.VECTOR_POWER_W
        * (wv / npu_config.REFERENCE_VECTOR_WIDTH_BITS)
    )
    feasible = (
        (area <= budget.area_mm2 + 1e-12)
        & (area >= args.min_area_utilization * budget.area_mm2 - 1e-12)
    )
    if not args.ignore_power:
        feasible &= power <= budget.power_w + 1e-12
    if args.require_vector_multiple:
        feasible &= (nv % nc) == 0
    return nc[feasible], nv[feasible], wv[feasible], area[feasible], power[feasible], len(nc)


def write_top_csv(path: Path, data: dict[str, np.ndarray], objective: np.ndarray, top_k: int) -> None:
    order = np.argsort(objective, kind="stable")[:top_k]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "cube_cores", "vector_cores", "vector_width_bits", "area_mm2", "power_w", "geomean_request_latency_us"])
        for rank, index in enumerate(order, start=1):
            writer.writerow([
                rank, int(data["nc"][index]), int(data["nv"][index]), int(data["wv"][index]),
                float(data["area"][index]), float(data["power"][index]), float(objective[index] * 1e6),
            ])


def search_one(
    baseline_path: Path,
    calibration: dict,
    args: argparse.Namespace,
) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    chip = baseline.get("metadata", {}).get("name", baseline_path.stem)
    base_point = npu_config.baseline_design(baseline, args.cube_compression)
    budget = npu_config.resource_usage(base_point)
    nc, nv, wv, area, power, enumerated = candidate_space(args, budget)
    hw = estimator.derive_hardware(baseline_path)
    frequency_mhz = float(baseline["core_freq"])
    frequency_scale = frequency_mhz / 1000.0
    cube_rates = (
        nc.astype(np.float64)
        * npu_config.SIMULATOR_CUBE_TFLOPS_AT_1GHZ
        * 1e12
        * frequency_scale
    )
    vector_flops = (
        nv.astype(np.float64) * npu_config.REFERENCE_VECTOR_GFLOPS * 1e9
        * (wv.astype(np.float64) / npu_config.REFERENCE_VECTOR_WIDTH_BITS)
        * frequency_scale
    )
    vector_rates = vector_flops / 2.0
    cube_rates, vector_rates, _ = estimator.reconfiguration_compute_rates(
        cube_rates, vector_rates, calibration, chip, enabled=True,
    )

    workloads = workload_matrix(
        args.models, args.seq_lens, args.batch_sizes, args.users
    )
    log_latency_sum = np.zeros(len(nc), dtype=np.float64)
    safe = np.ones(len(nc), dtype=bool)
    costs = []
    for workload in workloads:
        medium = "ddr" if workload.user == "hot" else "ssd"
        profile = hardware_profile(calibration, chip, medium, workload.batch)
        cost = WorkloadCost(workload, hw, profile)
        _, latency = cost.select(cube_rates, vector_rates)
        log_latency_sum += np.log(latency / workload.batch)
        if args.min_predicted_speedup > 0.0:
            _, baseline_latency = cost.select(
                np.asarray([hw["F_cube"]]),
                np.asarray([hw["F_vec"]]),
            )
            safe &= latency <= (
                float(baseline_latency[0]) / args.min_predicted_speedup
            )
        costs.append(cost)
    objective = np.exp(log_latency_sum / len(workloads))
    if not np.any(safe):
        raise RuntimeError(
            "no feasible design satisfies --min-predicted-speedup="
            f"{args.min_predicted_speedup} for every workload"
        )
    objective = np.where(safe, objective, np.inf)
    best_index = int(np.argmin(objective))
    best_point = npu_config.DesignPoint(
        int(nc[best_index]), int(nv[best_index]), int(wv[best_index])
    )

    baseline_cube_rate = np.asarray([hw["F_cube"]], dtype=np.float64)
    baseline_vector_rate = np.asarray([hw["F_vec"]], dtype=np.float64)
    baseline_logs = 0.0
    selections = []
    for workload, cost in zip(workloads, costs):
        selected_k, latency = cost.select(
            np.asarray([cube_rates[best_index]]),
            np.asarray([vector_rates[best_index]]),
        )
        _, baseline_latency = cost.select(baseline_cube_rate, baseline_vector_rate)
        baseline_logs += math.log(float(baseline_latency[0]) / workload.batch)
        selections.append({
            **asdict(workload),
            "history_recompute_len": int(selected_k[0]),
            "recompute_ratio": float(selected_k[0] / (workload.sequence + 1) * 2.0),
            "predicted_request_latency_us": float(latency[0] / workload.batch * 1e6),
        })
    baseline_objective = math.exp(baseline_logs / len(workloads))

    chip_root = args.output_root / chip
    chip_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chip_root / "candidate_scores.npz",
        cube_cores=nc,
        vector_cores=nv,
        vector_width_bits=wv,
        area_mm2=area,
        power_w=power,
        geomean_request_latency_s=objective,
    )
    write_top_csv(
        chip_root / "top_candidates.csv",
        {"nc": nc, "nv": nv, "wv": wv, "area": area, "power": power},
        objective, args.top_k,
    )
    (chip_root / "optimal_recompute_choices.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    generated, mapping = npu_config.materialize_config(
        baseline, best_point, cube_compression=args.cube_compression
    )
    config_path = args.output_root / "configs" / f"{chip}_optimal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(generated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "chip": chip,
        "baseline_config": str(baseline_path),
        "baseline_design": asdict(base_point),
        "baseline_budget": asdict(budget),
        "enumerated_count": enumerated,
        "feasible_count": int(len(nc)),
        "workload_count": len(workloads),
        "objective": "geometric_mean_per_request_latency",
        "min_predicted_speedup_per_workload": args.min_predicted_speedup,
        "safe_candidate_count": int(np.sum(safe)),
        "baseline_predicted_geomean_us": baseline_objective * 1e6,
        "optimal_predicted_geomean_us": float(objective[best_index] * 1e6),
        "predicted_speedup": baseline_objective / float(objective[best_index]),
        "optimal_design": asdict(best_point),
        "optimal_area_mm2": float(area[best_index]),
        "optimal_power_w": float(power[best_index]),
        "target_cube_tflops": float(cube_rates[best_index] / 1e12),
        "target_vector_tflops": float(vector_flops[best_index] / 1e12),
        "generated_config": str(config_path),
        "simulation_mapping": mapping,
    }


def write_summary(root: Path, results: list[dict], args: argparse.Namespace) -> None:
    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cost_model": "explicit_item_kv_paper_model",
        "method": "w_both",
        "kv_reuse_ratio": KV_REUSE_RATIO,
        "search_ranges": {
            "cube_cores": [args.nc_min, args.nc_max, args.nc_step],
            "vector_cores": [args.nv_min, args.nv_max, args.nv_step],
            "vector_width_bits": [args.wv_min, args.wv_max, args.wv_step],
        },
        "unit_costs": {
            "cube_8tflops": {"area_mm2": npu_config.CUBE_AREA_MM2, "power_w": npu_config.CUBE_POWER_W},
            "vector_256gflops_2048bit": {"area_mm2": npu_config.VECTOR_AREA_MM2, "power_w": npu_config.VECTOR_POWER_W},
        },
        "constraints": {
            "min_area_utilization": args.min_area_utilization,
            "enforce_power_budget": not args.ignore_power,
            "require_vector_multiple": args.require_vector_multiple,
            "min_predicted_speedup_per_workload": args.min_predicted_speedup,
        },
        "results": results,
    }
    (root / "search_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (root / "search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "chip", "enumerated_count", "feasible_count",
            "baseline_predicted_geomean_us", "optimal_predicted_geomean_us", "predicted_speedup",
            "cube_cores", "vector_cores", "vector_width_bits",
            "optimal_area_mm2", "optimal_power_w", "target_cube_tflops", "target_vector_tflops",
            "generated_config",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                **{key: result[key] for key in fields if key in result},
                **result["optimal_design"],
            })
    lines = [
        "# NPU Reconfiguration Search", "",
        "All feasible designs satisfy the baseline compute-unit area and power budgets.", "",
        "| Baseline | Feasible | Optimal (Nc, Nv, Wv) | Cube TFLOPS | Vector TFLOPS | Predicted speedup |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        point = result["optimal_design"]
        lines.append(
            f"| {result['chip']} | {result['feasible_count']} | "
            f"({point['cube_cores']}, {point['vector_cores']}, {point['vector_width_bits']}) | "
            f"{result['target_cube_tflops']:.3f} | {result['target_vector_tflops']:.3f} | "
            f"{result['predicted_speedup']:.4f}x |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for lower, upper, step, name in (
        (args.nc_min, args.nc_max, args.nc_step, "Nc"),
        (args.nv_min, args.nv_max, args.nv_step, "Nv"),
        (args.wv_min, args.wv_max, args.wv_step, "Wv"),
    ):
        if lower <= 0 or upper < lower or step <= 0:
            raise SystemExit(f"invalid {name} range")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for baseline in args.baselines:
        result = search_one(baseline, calibration, args)
        results.append(result)
        print(
            f"[{result['chip']}] feasible={result['feasible_count']} "
            f"optimal={result['optimal_design']} speedup={result['predicted_speedup']:.4f}x",
            flush=True,
        )
    write_summary(args.output_root, results, args)


if __name__ == "__main__":
    main()
