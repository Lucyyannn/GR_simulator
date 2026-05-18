#!/usr/bin/env python3
"""Formula-based item recompute ratio estimator from SOW2 Section 6.

The model scans candidate recompute count k and chooses the value that balances
the repeated non-layer0 critical path:

    T_kv_layer(k) ~= T_compute_layer(k)

where:
    T_mem(k) = T_emb(k) + T_kv(k) + T_cand + T_w
    T_npu(k) = T_cube(k) + T_vec(k) + T_core(k)

Tmem/Tnpu are still reported for diagnostics, but the default chosen ratio
follows the repeated non-layer0 balance rule used for planning:

    layer remaining KV preload duration ~= layer busy compute duration

Busy compute includes both candidate-side work and recomputed history-side
work. The early/late split is reported only as a diagnostic view of the layer
schedule. Embedding/candidate reads and weights are not part of the repeated
per-layer KV preload duration; weights are resident in HBM by default.

Default hardware parameters are derived from configs/910C.json, then
optionally adjusted by scripts/recompute_ratio_calibration.json. CLI
overrides still take highest precedence.

Calibration is used as a parametric correction over hidden size, sequence
length, batch size, reuse mode, user hot/cold class, and recompute ratio. The
legacy per-context table in the calibration JSON is a diagnostic/fallback, not
the primary prediction rule.

脚本共享/迁移说明：
    必需文件：
        - 本脚本。
        - 通过 --config 指定的仿真器硬件配置文件，默认是
          configs/baseline.json。该配置提供 NPU core 拓扑、存储通道与带宽
          参数、SSD 结构参数以及数值精度。
    可选但推荐：
        - scripts/recompute_ratio_calibration.json，默认由 --calibration
          读取。该文件包含从已有仿真结果拟合得到的 per-layer DDR/SSD 利用率
          和参数化 compute-scale 模型。如果缺少该文件，脚本会退化为纯公式估算，
          结果通常不如带校准时可靠。
    换到不同硬件配置时：
        - 优先提供新的 --config JSON，描述目标硬件。
        - 也可以直接修改下方 HW_* 宏定义：HW_NUM_CORES、
          HW_CORE_FREQ_MHZ、HW_CORE_WIDTH/HEIGHT、HW_VECTOR_PROCESS_BITS、
          HW_PRECISION_BYTES、HW_DDR_* / HW_HBM_* 带宽参数，以及 HW_SSD_*
          结构参数或 HW_SSD_PEAK_BPS。
        - 如果已知多 core 扩展效率不理想，可将 HW_NPU_PARALLEL_EFFICIENCY
          设置为小于 1.0；更可靠的做法是用目标硬件上的代表性实验重新运行
          calibrate_recompute_cost_model.py 生成新的 calibration JSON。
    每组实验通常通过 CLI 修改的输入：
        --user hot|cold、--layers、--hidden、--kv-len、--batch-size、
        --candidates、--enable-kv-reuse、--kv-reuse-ratio、
        --embedding-source、--objective。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Hardware macro overrides. Leave a value as None to use the JSON config.
# These are intentionally near the top of the script so a new hardware target
# can be evaluated without changing the modeling code below.
HW_NUM_CORES: int | None = None
HW_CORE_FREQ_MHZ: float | None = None
HW_CORE_WIDTH: int | None = None
HW_CORE_HEIGHT: int | None = None
HW_VECTOR_PROCESS_BITS: int | None = None
HW_PRECISION_BYTES: int | None = None

HW_DDR_CHANNELS: int | None = None
HW_DDR_REQ_SIZE_BYTES: int | None = None
HW_DDR_FREQ_MHZ: float | None = None
HW_DDR_PEAK_BPS: float | None = None

HW_HBM_CHANNELS: int | None = None
HW_HBM_REQ_SIZE_BYTES: int | None = None
HW_HBM_FREQ_MHZ: float | None = None
HW_HBM_PEAK_BPS: float | None = None

HW_SSD_CHANNELS: int | None = None
HW_SSD_LUNS_PER_CH: int | None = None
HW_SSD_PLANES_PER_LUN: int | None = None
HW_SSD_SECTOR_BYTES: int | None = None
HW_SSD_SECTORS_PER_PAGE: int | None = None
HW_SSD_PAGE_READ_LAT_NS: float | None = None
HW_SSD_PEAK_BPS: float | None = None

# If a core-count sweep shows imperfect scaling, set this below 1.0. The
# learned compute-scale model also captures baseline imbalance/overhead, so
# keep this at 1.0 unless calibrating for a new NPU topology.
HW_NPU_PARALLEL_EFFICIENCY: float | None = None

REFERENCE_NUM_CORES = 8
REFERENCE_DDR_PEAK_BPS = 204.8e9
REFERENCE_SSD_PEAK_BPS = 6.5536e9


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bytes_per_second_from_mem(cfg: dict, name: str) -> float:
    mem = cfg[name]
    return float(mem["channels"]) * float(mem["req_size"]) * float(mem["freq"]) * 1e6


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def derive_hardware(config_path: Path) -> dict[str, float]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    precision_bytes = float(first_not_none(HW_PRECISION_BYTES, cfg.get("precision", 2)))
    core_cfg = cfg["core_config"]
    core_count = int(first_not_none(HW_NUM_CORES, cfg.get("num_cores", len(core_cfg))))
    first_core = next(iter(core_cfg.values()))
    core_freq = float(first_not_none(HW_CORE_FREQ_MHZ, cfg["core_freq"])) * 1e6
    core_width = float(first_not_none(HW_CORE_WIDTH, first_core["core_width"]))
    core_height = float(first_not_none(HW_CORE_HEIGHT, first_core["core_height"]))
    vector_bits = float(first_not_none(HW_VECTOR_PROCESS_BITS, first_core["vector_process_bit"]))

    hbm = cfg["hbm"]
    hbm_bps = float(
        first_not_none(
            HW_HBM_PEAK_BPS,
            float(first_not_none(HW_HBM_CHANNELS, hbm["channels"]))
            * float(first_not_none(HW_HBM_REQ_SIZE_BYTES, hbm["req_size"]))
            * float(first_not_none(HW_HBM_FREQ_MHZ, hbm["freq"]))
            * 1e6,
        )
    )
    ddr = cfg["ddr"]
    ddr_bps = float(
        first_not_none(
            HW_DDR_PEAK_BPS,
            float(first_not_none(HW_DDR_CHANNELS, ddr["channels"]))
            * float(first_not_none(HW_DDR_REQ_SIZE_BYTES, ddr["req_size"]))
            * float(first_not_none(HW_DDR_FREQ_MHZ, ddr["freq"]))
            * 1e6,
        )
    )

    ssd = cfg["ssd"]
    page_bytes = float(first_not_none(HW_SSD_SECTOR_BYTES, ssd["secsz"])) * float(
        first_not_none(HW_SSD_SECTORS_PER_PAGE, ssd["secs_per_pg"])
    )
    parallel_pages = (
        float(first_not_none(HW_SSD_CHANNELS, ssd["nchs"]))
        * float(first_not_none(HW_SSD_LUNS_PER_CH, ssd["luns_per_ch"]))
        * float(first_not_none(HW_SSD_PLANES_PER_LUN, ssd["pls_per_lun"]))
    )
    ssd_bps = float(
        first_not_none(
            HW_SSD_PEAK_BPS,
            parallel_pages * page_bytes / (float(first_not_none(HW_SSD_PAGE_READ_LAT_NS, ssd["pg_rd_lat"])) * 1e-9),
        )
    )

    # Systolic array: one multiply-add is counted as two FLOPs.
    parallel_eff = float(first_not_none(HW_NPU_PARALLEL_EFFICIENCY, 1.0))
    f_cube = core_count * parallel_eff * core_width * core_height * 2.0 * core_freq
    # Vector model uses element throughput. precision is bytes, hence *8 bits.
    f_vec = core_count * parallel_eff * (vector_bits / (precision_bytes * 8.0)) * core_freq

    return {
        "B_hbm": hbm_bps,
        "B_ddr": ddr_bps,
        "B_ssd": ssd_bps,
        "B_hbm_peak": hbm_bps,
        "B_ddr_peak": ddr_bps,
        "B_ssd_peak": ssd_bps,
        "B_core": hbm_bps,
        "F_cube": f_cube,
        "F_vec": f_vec,
        "s": precision_bytes,
        "num_cores": core_count,
    }


def action_rows_after_reuse(kv_len: int, kv_reuse_ratio: float) -> int:
    action_count = kv_len // 2
    if kv_reuse_ratio <= 0:
        return action_count
    action_ratio = clamp(kv_reuse_ratio * kv_len / action_count, 0.0, 1.0)
    return max(1, round(action_count * (1.0 - action_ratio)))


def load_calibration(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_scale_features(hidden: int, kv_len: int, batch: int, recompute_ratio: float, enable_kv_reuse: bool, user: str) -> dict[str, float]:
    log_hidden = math.log2(hidden / 256.0)
    log_seq = math.log2(kv_len / 4096.0)
    log_batch = math.log2(max(batch, 1))
    kv_reuse = 1.0 if enable_kv_reuse else 0.0
    hot = 1.0 if user == "hot" else 0.0
    return {
        "intercept": 1.0,
        "log_hidden": log_hidden,
        "log_seq": log_seq,
        "log_batch": log_batch,
        "recompute_ratio": recompute_ratio,
        "kv_reuse": kv_reuse,
        "hot": hot,
        "kv_reuse_hot": kv_reuse * hot,
        "log_seq_kv_reuse": log_seq * kv_reuse,
        "log_hidden_kv_reuse": log_hidden * kv_reuse,
        "log_batch_kv_reuse": log_batch * kv_reuse,
        "log_hidden_log_seq": log_hidden * log_seq,
    }


def compute_scale(
    calibration: dict,
    hidden: int,
    scheme: str,
    user: str,
    kv_len: int,
    batch: int,
    recompute_ratio: float,
    enable_kv_reuse: bool,
) -> float:
    model = calibration.get("compute_scale_model")
    if model:
        features = compute_scale_features(hidden, kv_len, batch, recompute_ratio, enable_kv_reuse, user)
        value = float(model.get("intercept", 0.0))
        for name, coef in model.get("coefficients", {}).items():
            value += float(coef) * features.get(name, 0.0)
        if model.get("target", "log_scale") == "log_scale":
            return math.exp(value)
        return max(0.01, value)

    by_context = calibration.get("compute_scale_by_context", {})
    context_value = (
        by_context.get(scheme, {})
        .get(user, {})
        .get(str(hidden), {})
        .get(str(kv_len))
    )
    if context_value is not None:
        return float(context_value)
    by_hidden = calibration.get("compute_scale_by_hidden", {})
    if str(hidden) in by_hidden:
        return float(by_hidden[str(hidden)])
    return float(calibration.get("compute_scale", 1.0))


def batch_effective(batch: int, calibration: dict) -> float:
    gamma = float(calibration.get("batch_gamma", 1.0))
    scale = float(calibration.get("batch_scale", 1.0))
    return scale * math.pow(batch, gamma)


def apply_calibrated_bandwidth(hw: dict[str, float], calibration: dict) -> None:
    preload_model = calibration.get("preload_bandwidth_model")
    if preload_model:
        if "ddr" in preload_model:
            hw["B_ddr"] = hw["B_ddr_peak"] * float(preload_model["ddr"].get("peak_utilization", 1.0))
        if "ssd" in preload_model:
            hw["B_ssd"] = hw["B_ssd_peak"] * float(preload_model["ssd"].get("peak_utilization", 1.0))
        return

    if "effective_bandwidth_bps" in calibration:
        effective = calibration["effective_bandwidth_bps"]
        if "ddr" in effective:
            hw["B_ddr"] = float(effective["ddr"])
        if "ssd" in effective:
            hw["B_ssd"] = float(effective["ssd"])


def estimate(args: argparse.Namespace) -> dict[str, float | int]:
    hw = derive_hardware(args.config)
    calibration = load_calibration(args.calibration)
    apply_calibrated_bandwidth(hw, calibration)
    for key in ["B_ddr", "B_ssd", "B_hbm", "B_core", "F_cube", "F_vec"]:
        override = getattr(args, key.lower())
        if override is not None:
            hw[key] = override

    L = args.layers
    H = args.hidden
    batch = args.batch_size
    batch_eff = batch_effective(batch, calibration)
    scheme = "W_both" if args.enable_kv_reuse else "W_IR"
    C = args.candidates
    s = hw["s"]
    item_count = (args.kv_len + 1) // 2
    if args.enable_kv_reuse:
        action_count = action_rows_after_reuse(args.kv_len, args.kv_reuse_ratio)
    else:
        action_count = args.kv_len // 2
    Si = item_count
    Sa = action_count
    Sh = Si + Sa
    Sr = args.sr if args.sr is not None else Sh
    Mw = args.weight_matrices

    Bkv = hw["B_ddr"] if args.user == "hot" else hw["B_ssd"]
    Bemb = hw["B_ssd"] if args.embedding_source == "ssd" else hw["B_ddr"]
    Bcand = Bemb
    Bw = hw["B_hbm"] if args.weights_resident else (hw["B_ddr"] if args.weight_source == "ddr" else hw["B_ssd"])
    Bcore = hw["B_core"]
    Fcube = hw["F_cube"]
    Fvec = hw["F_vec"]

    best: dict[str, float | int] | None = None
    for k in range(item_count + 1):
        recompute_ratio = k / item_count if item_count else 0.0
        op_scale = compute_scale(calibration, H, scheme, args.user, args.kv_len, batch, recompute_ratio, args.enable_kv_reuse)
        Temb = batch * k * H * s / Bemb
        remaining_kv_rows = batch * (Sa + max(0, Si - k))
        Tkv_layer = remaining_kv_rows * 2 * H * s / Bkv
        Tkv = L * Tkv_layer
        Tcand = batch * C * H * s / Bcand
        Tw = 0.0 if args.weights_resident else L * Mw * H * H * s / Bw
        Tmem = Temb + Tkv + Tcand + Tw

        # Compute is modeled per user, then scaled by the observed effective
        # batch factor. This avoids turning attention terms into batch^2.
        Tcube_rank_user = (8 * C * H * H + 4 * C * Sh * H) / Fcube
        Tcube_rec_user = (8 * k * H * H + 4 * k * Sr * H) / Fcube
        Tvec_rank_user = (C * Sh + 2 * C * H) / Fvec
        Tvec_rec_user = (k * Sr + 2 * k * H) / Fvec
        Tcore_rank_user = s * (C * H + 2 * Sh * H) / Bcore
        Tcore_rec_user = s * (k * H + 4 * k * H) / Bcore

        Tcube_rank_layer = op_scale * batch_eff * Tcube_rank_user
        Tcube_rec_layer = op_scale * batch_eff * Tcube_rec_user
        Tvec_rank_layer = op_scale * batch_eff * Tvec_rank_user
        Tvec_rec_layer = op_scale * batch_eff * Tvec_rec_user
        Tcore_rank_layer = op_scale * batch_eff * Tcore_rank_user
        Tcore_rec_layer = op_scale * batch_eff * Tcore_rec_user

        Tcube_layer = Tcube_rank_layer + Tcube_rec_layer
        Tvec_layer = Tvec_rank_layer + Tvec_rec_layer
        Tcore_layer = Tcore_rank_layer + Tcore_rec_layer
        Tcube = L * Tcube_layer
        Tvec = L * Tvec_layer
        Tcore = L * Tcore_layer
        Tnpu = Tcube + Tvec + Tcore
        Tlat = max(Tmem, Tnpu)
        Tearly_layer = Tcube_rec_layer + Tvec_rec_layer + Tcore_rec_layer
        Tlate_layer = Tcube_rank_layer + Tvec_rank_layer + Tcore_rank_layer
        Tcompute_layer = Tearly_layer + Tlate_layer
        pipeline_layer = max(Tkv_layer, Tearly_layer) + Tlate_layer
        steady_layer = max(Tkv_layer, Tcompute_layer)
        layer_balance = abs(Tkv_layer - Tcompute_layer)
        if args.objective == "balance":
            objective_value = layer_balance
        elif args.objective == "steady":
            objective_value = steady_layer
        else:
            objective_value = pipeline_layer

        candidate = {
            "history_recompute_len": k,
            "recompute_ratio": recompute_ratio,
            "Tmem_us": Tmem * 1e6,
            "Tnpu_us": Tnpu * 1e6,
            "Tlatency_us": Tlat * 1e6,
            "balance_error_us": abs(Tmem - Tnpu) * 1e6,
            "layer_kv_preload_us": Tkv_layer * 1e6,
            "layer_early_compute_us": Tearly_layer * 1e6,
            "layer_late_compute_us": Tlate_layer * 1e6,
            "layer_compute_us": Tcompute_layer * 1e6,
            "pipeline_layer_us": pipeline_layer * 1e6,
            "steady_layer_us": steady_layer * 1e6,
            "objective_value_us": objective_value * 1e6,
            "objective_layer_us": objective_value * 1e6,
            "objective": args.objective,
            "layer_balance_error_us": layer_balance * 1e6,
            "item_rows": item_count * batch,
            "action_rows": action_count * batch,
            "Sh": Sh,
            "Sr": Sr,
            "num_cores": int(hw["num_cores"]),
            "weights_resident": args.weights_resident,
            "effective_batch": batch_eff,
            "compute_scale": op_scale,
            "B_ddr": hw["B_ddr"],
            "B_ssd": hw["B_ssd"],
        }
        if best is None or candidate["objective_value_us"] < best["objective_value_us"] or (
            candidate["objective_value_us"] == best["objective_value_us"]
            and candidate["steady_layer_us"] < best["steady_layer_us"]
        ):
            best = candidate
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/910C.json"))
    parser.add_argument("--calibration", type=Path, default=Path("scripts/recompute_ratio_calibration.json"))
    parser.add_argument("--user", choices=["hot", "cold"], required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Accepted for interface compatibility; model is per request.")
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--enable-kv-reuse", action="store_true")
    parser.add_argument("--kv-reuse-ratio", type=float, default=0.0)
    parser.add_argument("--embedding-source", choices=["ssd", "ddr"], default="ssd")
    parser.set_defaults(weights_resident=True)
    parser.add_argument("--weights-resident", dest="weights_resident", action="store_true")
    parser.add_argument("--no-weights-resident", dest="weights_resident", action="store_false")
    parser.add_argument("--weight-source", choices=["ssd", "ddr"], default="ddr")
    parser.add_argument("--weight-matrices", type=int, default=4)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--b_ddr", type=float, default=None, help="Override DDR bandwidth in bytes/s.")
    parser.add_argument("--b_ssd", type=float, default=None, help="Override SSD bandwidth in bytes/s.")
    parser.add_argument("--b_hbm", type=float, default=None, help="Override HBM bandwidth in bytes/s.")
    parser.add_argument("--b_core", type=float, default=None, help="Override HBM-to-core bandwidth in bytes/s.")
    parser.add_argument("--f_cube", type=float, default=None, help="Override Cube throughput in FLOP/s.")
    parser.add_argument("--f_vec", type=float, default=None, help="Override vector throughput in element/s.")
    parser.add_argument(
        "--objective",
        choices=["balance", "steady", "pipeline"],
        default="balance",
        help=(
            "balance minimizes abs(kv_preload-busy_compute); steady minimizes "
            "max(kv_preload,busy_compute); pipeline minimizes "
            "max(kv_preload,early_compute)+late_compute."
        ),
    )
    parser.add_argument("--field", choices=["json", "len", "ratio"], default="json")
    args = parser.parse_args()
    result = estimate(args)
    if args.field == "len":
        print(result["history_recompute_len"])
    elif args.field == "ratio":
        print(result["recompute_ratio"])
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
