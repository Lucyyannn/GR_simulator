#!/usr/bin/env python3
"""Formula-based item recompute ratio estimator from SOW2 Section 6.

The model scans candidate recompute count k and chooses the value that balances
the repeated non-layer0 layer cost:

    T_kv_layer(k) ~= T_compute_layer(k)

where:
    T_kv_layer(k) is the repeated-layer cached KV preload time after item
    recompute and optional action reuse.
    T_compute_layer(k) = T_cube(k) + T_vec(k) + T_core(k)

Layer0-only reordered item embedding reads, candidate embedding reads, and
weight reads are still reported for diagnostics, but they are not mixed into
the default IR selection objective. The chosen ratio follows the repeated-layer
balance rule used for item recompute planning:

    next layer cached KV preload duration ~= current layer full compute duration

Pre-attention history embedding, candidate embedding, and weights are not part
of the repeated cached KV preload duration; weights are resident in HBM by
default.

Default hardware parameters are derived from the simulator config. Bandwidth
metadata (`derived_*` / `target_*`) is preferred over raw `channels * req_size *
freq` so the estimator stays aligned with nbl-adjusted simulator accounting.
Calibration is then applied as effective peak-utilization multipliers. CLI
overrides still take highest precedence.

The legacy fitted compute-scale model was trained against an older objective
and is not applied to IR candidate selection by default. A new calibration can
opt in through `ir_cost_model.compute_scale` and
`ir_cost_model.kv_preload_utilization`. The built-in IR compute correction
matches the simulator's measured steady-layer op timing and can be overridden
by calibration.

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
DEFAULT_IR_COMPUTE_SCALE = 2.3
DEFAULT_AR_SPLIT_MERGE_SCALE_DDR = 1.4
DEFAULT_AR_SPLIT_MERGE_SCALE_SSD = 0.0
DEFAULT_IR_KV_PRELOAD_UTILIZATION = 0.88
DEFAULT_IR_COMPUTE_CORRECTION = {
    "910A": {"base_at_hidden_256": 4.13, "hidden_exponent": 0.48},
    "910B": {"base_at_hidden_256": 5.74, "hidden_exponent": 0.51},
    "910C": {"base_at_hidden_256": 11.27, "hidden_exponent": 0.70},
}
DEFAULT_W_BOTH_COMPUTE_CORRECTION = {
    "reference_hidden": 256.0,
    "base": 1.0,
    "slope": 0.2,
    "min_factor": 1.0,
    "max_factor": 1.4,
}


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


def metadata_bandwidth_bps(metadata: dict, *names: str) -> float | None:
    for name in names:
        for prefix in ("derived", "target"):
            value = metadata.get(f"{prefix}_{name}_bandwidth_GBps")
            if value is not None:
                return float(value) * 1e9
    return None


def derive_hardware(config_path: Path) -> dict[str, float]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = cfg.get("metadata", {})
    precision_bytes = float(first_not_none(HW_PRECISION_BYTES, cfg.get("precision", 2)))
    core_cfg = cfg["core_config"]
    core_count = int(first_not_none(HW_NUM_CORES, cfg.get("num_cores", len(core_cfg))))
    first_core = next(iter(core_cfg.values()))
    core_freq = float(first_not_none(HW_CORE_FREQ_MHZ, cfg["core_freq"])) * 1e6
    core_width = float(first_not_none(HW_CORE_WIDTH, first_core["core_width"]))
    core_height = float(first_not_none(HW_CORE_HEIGHT, first_core["core_height"]))
    vector_bits = float(first_not_none(HW_VECTOR_PROCESS_BITS, first_core["vector_process_bit"]))

    hbm = cfg["hbm"]
    metadata_hbm_bps = metadata_bandwidth_bps(metadata, "hbm")
    hbm_bps = float(
        first_not_none(
            HW_HBM_PEAK_BPS,
            metadata_hbm_bps,
            float(first_not_none(HW_HBM_CHANNELS, hbm["channels"]))
            * float(first_not_none(HW_HBM_REQ_SIZE_BYTES, hbm["req_size"]))
            * float(first_not_none(HW_HBM_FREQ_MHZ, hbm["freq"]))
            * 1e6,
        )
    )
    ddr = cfg["ddr"]
    metadata_ddr_bps = metadata_bandwidth_bps(metadata, "ddr", "dram")
    ddr_bps = float(
        first_not_none(
            HW_DDR_PEAK_BPS,
            metadata_ddr_bps,
            float(first_not_none(HW_DDR_CHANNELS, ddr["channels"]))
            * float(first_not_none(HW_DDR_REQ_SIZE_BYTES, ddr["req_size"]))
            * float(first_not_none(HW_DDR_FREQ_MHZ, ddr["freq"]))
            * 1e6,
        )
    )

    ssd = cfg["ssd"]
    metadata_ssd_read_bps = metadata_bandwidth_bps(metadata, "ssd_read", "ssd")
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
            metadata_ssd_read_bps,
            float(ssd["read_bandwidth_GBps"]) * 1e9 if ssd.get("read_bandwidth_GBps") is not None else None,
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
        "chip": str(metadata.get("name", config_path.stem)),
    }


def compressed_rows_for_ratio(logical_rows: int, reuse_ratio: float) -> int:
    if logical_rows <= 0:
        return 0
    ratio = clamp(reuse_ratio, 0.0, 1.0)
    if ratio <= 0:
        return logical_rows
    return max(1, int(round(logical_rows * (1.0 - ratio))))


def item_action_cached_rows_after_recompute(kv_len: int, history_recompute_len: int) -> tuple[int, int]:
    if history_recompute_len >= kv_len:
        return 0, 0
    item_count = (kv_len + 1) // 2
    action_count = kv_len // 2
    recomputed_items = min(history_recompute_len, item_count)
    recomputed_actions = max(0, history_recompute_len - item_count)
    remaining_items = max(0, item_count - recomputed_items)
    remaining_actions = max(0, action_count - recomputed_actions)
    return remaining_items, remaining_actions


def action_reuse_ratio_from_total(kv_len: int, action_count: int, kv_reuse_ratio: float) -> float:
    if action_count <= 0:
        return 0.0
    return clamp(clamp(kv_reuse_ratio, 0.0, 1.0) * kv_len / action_count, 0.0, 1.0)


def effective_cached_rows_after_item_recompute_action_reuse(
    kv_len: int,
    history_recompute_len: int,
    kv_reuse_ratio: float,
) -> int:
    remaining_items, remaining_actions = item_action_cached_rows_after_recompute(kv_len, history_recompute_len)
    if kv_reuse_ratio <= 0:
        return remaining_items + remaining_actions
    total_action_count = kv_len // 2
    action_ratio = action_reuse_ratio_from_total(kv_len, total_action_count, kv_reuse_ratio)
    action_physical_rows = compressed_rows_for_ratio(remaining_actions, action_ratio)
    return remaining_items + action_physical_rows


def unreused_action_compute_ratio(kv_reuse_ratio: float) -> float:
    if kv_reuse_ratio <= 0:
        return 0.0
    return max(0.0, 0.5 - clamp(kv_reuse_ratio, 0.0, 1.0))


def history_action_compute_rows_after_recompute(
    history_recompute_len: int,
    kv_reuse_ratio: float,
) -> int:
    if history_recompute_len <= 0:
        return 0
    return int(
        math.ceil(
            history_recompute_len * unreused_action_compute_ratio(kv_reuse_ratio)
        )
    )


def split_recompute_attention_score_elements(
    cached_kv_len: int,
    history_recompute_len: int,
    candidate_tokens: int,
    kv_reuse_ratio: float,
) -> tuple[int, int]:
    effective_cached_kv_len = effective_cached_rows_after_item_recompute_action_reuse(
        cached_kv_len + history_recompute_len,
        history_recompute_len,
        kv_reuse_ratio,
    )
    action_compute_rows = history_action_compute_rows_after_recompute(
        history_recompute_len, kv_reuse_ratio
    )
    if history_recompute_len <= 0:
        return (
            candidate_tokens * candidate_tokens,
            candidate_tokens * effective_cached_kv_len,
        )

    early_scores = (
        history_recompute_len * (history_recompute_len + 1) // 2
        + history_recompute_len * action_compute_rows
        + candidate_tokens * (history_recompute_len + candidate_tokens)
    )
    cached_scores = candidate_tokens * effective_cached_kv_len
    return early_scores, cached_scores


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


def ir_compute_scale(calibration: dict) -> float:
    model = calibration.get("ir_cost_model", {})
    if not isinstance(model, dict):
        return DEFAULT_IR_COMPUTE_SCALE
    value = model.get("compute_scale")
    return float(value) if value is not None else DEFAULT_IR_COMPUTE_SCALE


def lookup_chip_value(mapping: dict, chip: str):
    if not isinstance(mapping, dict):
        return None
    for key in (chip, chip.lower(), chip.upper(), "default", "*"):
        if key in mapping:
            return mapping[key]
    lower_lookup = {str(key).lower(): key for key in mapping}
    actual_key = lower_lookup.get(chip.lower())
    if actual_key is not None:
        return mapping[actual_key]
    return None


def ir_compute_correction_factor(calibration: dict, chip: str, hidden: int) -> tuple[float, dict]:
    model = calibration.get("ir_cost_model", {})
    entry = None
    source = "default_2layer_feedback"
    if isinstance(model, dict):
        configured = model.get("compute_correction")
        if isinstance(configured, (int, float)):
            return max(1.0, float(configured)), {
                "source": "ir_cost_model.compute_correction",
                "mode": "constant",
                "factor": max(1.0, float(configured)),
            }
        entry = lookup_chip_value(configured, chip) if isinstance(configured, dict) else None
        if entry is not None:
            source = "ir_cost_model.compute_correction"

    if entry is None:
        entry = lookup_chip_value(DEFAULT_IR_COMPUTE_CORRECTION, chip)
    if isinstance(entry, (int, float)):
        factor = max(1.0, float(entry))
        return factor, {"source": source, "mode": "constant", "factor": factor}
    if not isinstance(entry, dict):
        factor = 1.0
        return factor, {"source": "none", "mode": "disabled", "factor": factor}

    base = float(entry.get("base_at_hidden_256", entry.get("base", 1.0)))
    exponent = float(entry.get("hidden_exponent", entry.get("exponent", 0.0)))
    reference_hidden = float(entry.get("reference_hidden", 256.0))
    min_factor = float(entry.get("min_factor", 1.0))
    hidden = max(1, int(hidden))
    factor = max(min_factor, base * math.pow(reference_hidden / hidden, exponent))
    return factor, {
        "source": source,
        "mode": "chip_hidden_power",
        "factor": factor,
        "base_at_hidden_256": base,
        "hidden_exponent": exponent,
        "reference_hidden": reference_hidden,
        "min_factor": min_factor,
    }


def w_both_compute_correction_factor(
    calibration: dict,
    hidden: int,
    enable_kv_reuse: bool,
    history_recompute_len: int,
) -> tuple[float, dict]:
    if not enable_kv_reuse or history_recompute_len <= 0:
        return 1.0, {
            "source": "disabled",
            "mode": "not_w_both_or_k0",
            "factor": 1.0,
        }

    model = calibration.get("ir_cost_model", {})
    configured = None
    source = "default_2layer_feedback"
    if isinstance(model, dict):
        configured = model.get("w_both_compute_correction")
        if configured is not None:
            source = "ir_cost_model.w_both_compute_correction"

    if isinstance(configured, (int, float)):
        factor = max(1.0, float(configured))
        return factor, {"source": source, "mode": "constant", "factor": factor}

    if isinstance(configured, dict):
        by_hidden = configured.get(str(hidden))
        if by_hidden is None:
            by_hidden = configured.get(int(hidden)) if int(hidden) in configured else None
        if by_hidden is None:
            by_hidden = configured.get("default")
        if isinstance(by_hidden, (int, float)):
            factor = max(1.0, float(by_hidden))
            return factor, {
                "source": source,
                "mode": "by_hidden",
                "factor": factor,
                "hidden": int(hidden),
            }
        params = {**DEFAULT_W_BOTH_COMPUTE_CORRECTION, **configured}
    else:
        params = DEFAULT_W_BOTH_COMPUTE_CORRECTION

    hidden = max(1, int(hidden))
    reference_hidden = max(1.0, float(params.get("reference_hidden", 256.0)))
    base = float(params.get("base", 1.0))
    slope = float(params.get("slope", 0.2))
    min_factor = float(params.get("min_factor", 1.0))
    max_factor = float(params.get("max_factor", 1.4))
    factor = base + slope * math.log2(hidden / reference_hidden)
    factor = clamp(factor, min_factor, max_factor)
    return factor, {
        "source": source,
        "mode": "hidden_log2",
        "factor": factor,
        "hidden": hidden,
        "reference_hidden": reference_hidden,
        "base": base,
        "slope": slope,
        "min_factor": min_factor,
        "max_factor": max_factor,
    }


def ir_medium_value(mapping_or_value, medium: str, default: float | None = None) -> float | None:
    if mapping_or_value is None:
        return default
    if isinstance(mapping_or_value, dict):
        for key in (medium, medium.lower(), medium.upper(), "default", "*"):
            if key in mapping_or_value:
                return float(mapping_or_value[key])
        return default
    return float(mapping_or_value)


def ir_ar_split_merge_scale(calibration: dict, medium: str) -> float:
    model = calibration.get("ir_cost_model", {})
    if isinstance(model, dict):
        value = ir_medium_value(model.get("ar_split_merge_scale"), medium)
        if value is not None:
            return value
    if medium == "ddr":
        return DEFAULT_AR_SPLIT_MERGE_SCALE_DDR
    if medium == "ssd":
        return DEFAULT_AR_SPLIT_MERGE_SCALE_SSD
    return 0.0


def ir_kv_preload_bandwidth_bps(
    hw: dict[str, float],
    calibration: dict,
    medium: str,
) -> tuple[float, dict[str, float | str | bool | None]]:
    model = calibration.get("ir_cost_model", {})
    peak_bps = float(hw[f"B_{medium}_peak"])
    detail: dict[str, float | str | bool | None] = {
        "calibrated": False,
        "medium": medium,
        "source": "config_peak",
        "peak_Bps": peak_bps,
        "utilization": 1.0,
        "bandwidth_Bps": peak_bps,
    }
    if not isinstance(model, dict):
        return peak_bps, detail

    bandwidth_gbps = ir_medium_value(model.get("kv_preload_bandwidth_GBps"), medium)
    if bandwidth_gbps is not None:
        bandwidth_bps = bandwidth_gbps * 1e9
        detail.update(
            {
                "calibrated": True,
                "source": "ir_cost_model.kv_preload_bandwidth_GBps",
                "bandwidth_Bps": bandwidth_bps,
                "bandwidth_GBps": bandwidth_gbps,
                "utilization": bandwidth_bps / peak_bps if peak_bps > 0 else None,
            }
        )
        return bandwidth_bps, detail

    utilization = ir_medium_value(
        model.get("kv_preload_utilization"),
        medium,
        DEFAULT_IR_KV_PRELOAD_UTILIZATION,
    )
    if utilization is not None:
        bandwidth_bps = peak_bps * utilization
        detail.update(
            {
                "calibrated": model.get("kv_preload_utilization") is not None,
                "source": (
                    "ir_cost_model.kv_preload_utilization"
                    if model.get("kv_preload_utilization") is not None
                    else "default_2layer_feedback"
                ),
                "bandwidth_Bps": bandwidth_bps,
                "bandwidth_GBps": bandwidth_bps / 1e9,
                "utilization": utilization,
            }
        )
        return bandwidth_bps, detail
    return peak_bps, detail


def batch_effective(batch: int, calibration: dict) -> float:
    gamma = float(calibration.get("batch_gamma", 1.0))
    scale = float(calibration.get("batch_scale", 1.0))
    return scale * math.pow(batch, gamma)


def scheme_aliases(scheme: str) -> tuple[str, ...]:
    aliases = [scheme]
    if scheme.startswith("W_"):
        aliases.append("w_" + scheme[2:])
    if scheme.startswith("w_"):
        aliases.append("W_" + scheme[2:])
    aliases.extend([scheme.lower(), scheme.upper()])
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            result.append(alias)
    return tuple(result)


def nested_get(mapping: dict, keys: tuple[str, ...]) -> dict | None:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, dict) else None


def lookup_ir_context(
    calibration: dict,
    chip: str,
    source_medium: str,
    batch: int,
    scheme: str,
) -> dict:
    root = calibration.get("ir_calibration", {}).get("contexts", {})
    if not root:
        root = calibration.get("ir_contexts", {})
    if not isinstance(root, dict):
        return {}

    chip_keys = (chip, chip.lower(), "default", "*")
    source_keys = (source_medium, source_medium.upper(), "default", "*")
    batch_keys = (str(batch), f"batch{batch}", "default", "*")
    for chip_key in chip_keys:
        for source_key in source_keys:
            for batch_key in batch_keys:
                for scheme_key in scheme_aliases(scheme):
                    value = nested_get(root, (chip_key, source_key, batch_key, scheme_key))
                    if value is not None:
                        return value
    return {}


def apply_calibrated_bandwidth(hw: dict[str, float], calibration: dict) -> None:
    if calibration.get("memory_bandwidth_calibration"):
        return

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


def lookup_mapping_case_insensitive(mapping: dict, *keys: str) -> dict | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and isinstance(mapping[key], dict):
            return mapping[key]
    lower_lookup = {str(key).lower(): key for key in mapping}
    for key in keys:
        actual_key = lower_lookup.get(str(key).lower())
        if actual_key is not None and isinstance(mapping[actual_key], dict):
            return mapping[actual_key]
    return None


def lookup_memory_bandwidth_entry(
    calibration: dict,
    chip: str,
    medium: str,
    pattern: str,
    rw: str,
) -> dict | None:
    root = calibration.get("memory_bandwidth_calibration", {}).get("contexts", {})
    if not isinstance(root, dict):
        return None

    chip_map = lookup_mapping_case_insensitive(root, chip, chip.lower(), "default", "*")
    medium_map = lookup_mapping_case_insensitive(chip_map or {}, medium, medium.lower(), "default", "*")
    pattern_map = lookup_mapping_case_insensitive(medium_map or {}, pattern, pattern.lower(), "default", "*")
    if pattern_map is None and pattern != "contiguous":
        pattern_map = lookup_mapping_case_insensitive(medium_map or {}, "contiguous", "default", "*")
    if pattern_map is None:
        return None
    return lookup_mapping_case_insensitive(pattern_map, rw, rw.lower(), "default", "*")


def pick_memory_bandwidth_by_size(entry: dict, access_size_bytes: int) -> tuple[float | None, int | None]:
    by_size = entry.get("by_size")
    if isinstance(by_size, dict) and by_size:
        parsed_sizes: list[int] = []
        for key in by_size:
            try:
                parsed_sizes.append(int(key))
            except (TypeError, ValueError):
                continue
        if parsed_sizes:
            target = max(1, int(access_size_bytes))
            larger_or_equal = [size for size in parsed_sizes if size >= target]
            selected_size = min(larger_or_equal) if larger_or_equal else max(parsed_sizes)
            selected = by_size.get(str(selected_size), {})
            bandwidth = selected.get("effective_GBps")
            if bandwidth is not None:
                return float(bandwidth) * 1e9, selected_size

    bandwidth = entry.get("effective_GBps")
    if bandwidth is not None:
        return float(bandwidth) * 1e9, entry.get("best_size_bytes")
    return None, None


def memory_bandwidth_bps(
    calibration: dict,
    chip: str,
    medium: str,
    pattern: str,
    rw: str,
    access_size_bytes: int,
    fallback_bps: float,
    skip_calibration: bool = False,
) -> tuple[float, dict[str, float | int | str | bool | None]]:
    detail: dict[str, float | int | str | bool | None] = {
        "calibrated": False,
        "chip": chip,
        "medium": medium,
        "pattern": pattern,
        "rw": rw,
        "requested_size_bytes": int(access_size_bytes),
        "selected_size_bytes": None,
        "bandwidth_Bps": fallback_bps,
    }
    if skip_calibration:
        detail["source"] = "cli_override"
        return fallback_bps, detail

    entry = lookup_memory_bandwidth_entry(calibration, chip, medium, pattern, rw)
    if not entry:
        detail["source"] = "fallback"
        return fallback_bps, detail

    bandwidth, selected_size = pick_memory_bandwidth_by_size(entry, access_size_bytes)
    if bandwidth is None or bandwidth <= 0:
        detail["source"] = "fallback"
        return fallback_bps, detail

    detail.update(
        {
            "calibrated": True,
            "source": "memory_bandwidth_calibration",
            "selected_size_bytes": selected_size,
            "bandwidth_Bps": bandwidth,
            "bandwidth_GBps": bandwidth / 1e9,
        }
    )
    peak_gbps = entry.get("peak_GBps")
    if peak_gbps is not None:
        detail["peak_GBps"] = float(peak_gbps)
        detail["utilization"] = (bandwidth / 1e9) / float(peak_gbps) if float(peak_gbps) > 0 else None
    return bandwidth, detail


def apply_ir_context_calibration(
    hw: dict[str, float],
    context: dict,
    source_medium: str,
) -> dict[str, float]:
    applied = {
        "preload_peak_util": 1.0,
        "hbm_peak_util": 1.0,
        "compute_scale_mult": 1.0,
        "recompute_compute_scale_mult": 1.0,
        "pre_cached_compute_scale_mult": 1.0,
        "ar_split_merge_scale": None,
    }
    if not context:
        return applied

    preload_util = context.get("preload_peak_util", context.get("peak_utilization"))
    if preload_util is not None:
        preload_util = float(preload_util)
        applied["preload_peak_util"] = preload_util
        if source_medium == "ddr":
            hw["B_ddr"] = hw["B_ddr_peak"] * preload_util
        elif source_medium == "ssd":
            hw["B_ssd"] = hw["B_ssd_peak"] * preload_util

    hbm_util = context.get("hbm_peak_util", context.get("core_peak_util"))
    if hbm_util is not None:
        hbm_util = float(hbm_util)
        applied["hbm_peak_util"] = hbm_util
        hw["B_hbm"] = hw["B_hbm_peak"] * hbm_util
        hw["B_core"] = hw["B_hbm_peak"] * hbm_util

    compute_mult = context.get("compute_scale_mult", context.get("compute_multiplier"))
    if compute_mult is not None:
        applied["compute_scale_mult"] = float(compute_mult)
    recompute_compute_mult = context.get(
        "recompute_compute_scale_mult",
        context.get("recompute_compute_multiplier", context.get("early_recompute_compute_scale_mult")),
    )
    if recompute_compute_mult is not None:
        applied["recompute_compute_scale_mult"] = float(recompute_compute_mult)
    pre_cached_mult = context.get("pre_cached_compute_scale_mult", context.get("early_compute_scale_mult"))
    if pre_cached_mult is not None:
        applied["pre_cached_compute_scale_mult"] = float(pre_cached_mult)
    ar_split_merge_scale = context.get("ar_split_merge_scale")
    if ar_split_merge_scale is not None:
        applied["ar_split_merge_scale"] = float(ar_split_merge_scale)
    return applied


def exact_context_recompute_len(context: dict, args: argparse.Namespace) -> int | None:
    value = context.get("history_recompute_len")
    if value is None:
        return None
    expected = {
        "calibrated_hidden": args.hidden,
        "calibrated_kv_len": args.kv_len,
        "calibrated_candidates": args.candidates,
        "calibrated_batch_size": args.batch_size,
    }
    for key, current in expected.items():
        if key in context and int(context[key]) != int(current):
            return None
    return int(value)


def calibrated_recompute_bound(
    context: dict,
    args: argparse.Namespace,
    max_recompute_items: int,
) -> tuple[int, str] | None:
    if not context:
        return None

    value = context.get("safe_max_history_recompute_len")
    source = "safe_max_history_recompute_len"
    if value is None:
        value = context.get("history_recompute_len")
        source = "history_recompute_len"
    if value is None:
        return None

    for key, current in (
        ("calibrated_candidates", args.candidates),
        ("calibrated_batch_size", args.batch_size),
    ):
        if key in context and int(context[key]) != int(current):
            return None

    calibrated_kv_len = int(context.get("calibrated_kv_len", args.kv_len))
    calibrated_item_count = max(1, (calibrated_kv_len + 1) // 2)
    current_item_count = max(1, (args.kv_len + 1) // 2)
    scaled = int(round(float(value) * current_item_count / calibrated_item_count))
    return max(0, min(max_recompute_items, scaled)), source


def compute_preload_tolerance_us(args: argparse.Namespace, layer_kv_preload_us: float) -> float:
    abs_tol = float(getattr(args, "compute_preload_tolerance_us", 5.0))
    ratio_tol = float(getattr(args, "compute_preload_tolerance_ratio", 0.05))
    return max(abs_tol, max(0.0, layer_kv_preload_us) * ratio_tol)


def estimate(args: argparse.Namespace) -> dict[str, float | int]:
    hw = derive_hardware(args.config)
    calibration = load_calibration(args.calibration)
    apply_calibrated_bandwidth(hw, calibration)

    L = args.layers
    H = args.hidden
    batch = args.batch_size
    batch_eff = batch_effective(batch, calibration)
    scheme = "W_both" if args.enable_kv_reuse else "W_IR"
    source_medium = args.embedding_source
    ir_context = lookup_ir_context(calibration, str(hw["chip"]), source_medium, batch, scheme)
    applied_context = apply_ir_context_calibration(hw, ir_context, source_medium)
    overridden: set[str] = set()
    for key in ["B_ddr", "B_ssd", "B_hbm", "B_core", "F_cube", "F_vec"]:
        override = getattr(args, key.lower())
        if override is not None:
            hw[key] = override
            overridden.add(key)

    C = args.candidates
    s = hw["s"]
    item_count = (args.kv_len + 1) // 2
    action_count = args.kv_len // 2
    Si = item_count
    Sa = action_count
    Sh = Si + Sa
    Sr = args.sr if args.sr is not None else Sh
    Mw = args.weight_matrices

    embedding_row_bytes = max(1, int(H * s))
    kv_row_bytes = max(1, int(2 * H * s))
    chip = str(hw["chip"])
    kv_medium = "ddr" if args.user == "hot" else "ssd"
    emb_medium = args.embedding_source
    weight_medium = "hbm" if args.weights_resident else args.weight_source

    if f"B_{kv_medium}" in overridden:
        Bkv = hw[f"B_{kv_medium}"]
        Bkv_cal = {
            "calibrated": False,
            "medium": kv_medium,
            "source": "cli_override",
            "bandwidth_Bps": Bkv,
            "bandwidth_GBps": Bkv / 1e9,
        }
    else:
        Bkv, Bkv_cal = ir_kv_preload_bandwidth_bps(hw, calibration, kv_medium)
    Bemb, Bemb_cal = memory_bandwidth_bps(
        calibration,
        chip,
        emb_medium,
        "random_512b_index",
        "read",
        embedding_row_bytes,
        hw[f"B_{emb_medium}"],
        f"B_{emb_medium}" in overridden,
    )
    Bcand, Bcand_cal = memory_bandwidth_bps(
        calibration,
        chip,
        emb_medium,
        "random_512b_index",
        "read",
        embedding_row_bytes,
        hw[f"B_{emb_medium}"],
        f"B_{emb_medium}" in overridden,
    )
    if args.weights_resident:
        Bw, Bw_cal = memory_bandwidth_bps(
            calibration,
            chip,
            "hbm",
            "contiguous",
            "read",
            max(1, int(H * H * s)),
            hw["B_hbm"],
            "B_hbm" in overridden,
        )
    else:
        Bw, Bw_cal = memory_bandwidth_bps(
            calibration,
            chip,
            weight_medium,
            "contiguous",
            "read",
            max(1, int(H * H * s)),
            hw[f"B_{weight_medium}"],
            f"B_{weight_medium}" in overridden,
        )
    Bcore, Bcore_cal = memory_bandwidth_bps(
        calibration,
        chip,
        "hbm",
        "contiguous",
        "read",
        kv_row_bytes,
        hw["B_core"],
        "B_core" in overridden or "B_hbm" in overridden,
    )
    Fcube = hw["F_cube"]
    Fvec = hw["F_vec"]
    compute_correction, compute_correction_detail = ir_compute_correction_factor(
        calibration, chip, H
    )

    # The trace path currently panics when every item row is recomputed. Keep
    # the formula search just below that simulator boundary.
    max_recompute_items = max(0, item_count - 1)
    fixed_recompute_len = getattr(args, "fixed_recompute_len", None)
    use_calibrated_len = bool(getattr(args, "use_calibrated_len", False))
    if fixed_recompute_len is None and use_calibrated_len:
        fixed_recompute_len = exact_context_recompute_len(ir_context, args)
    calibrated_bound = None
    calibrated_bound_source = ""
    if fixed_recompute_len is None and not bool(getattr(args, "ignore_calibrated_bound", False)):
        bound = calibrated_recompute_bound(ir_context, args, max_recompute_items)
        if bound is not None:
            calibrated_bound, calibrated_bound_source = bound

    search_max_recompute_items = max_recompute_items
    if calibrated_bound is not None:
        search_max_recompute_items = min(search_max_recompute_items, calibrated_bound)

    if fixed_recompute_len is None:
        recompute_lengths = range(search_max_recompute_items + 1)
    else:
        fixed_recompute_len = int(fixed_recompute_len)
        if fixed_recompute_len < 0 or fixed_recompute_len > max_recompute_items:
            raise ValueError(f"fixed_recompute_len={fixed_recompute_len} is outside [0, {max_recompute_items}]")
        recompute_lengths = (fixed_recompute_len,)

    candidates: list[dict] = []
    zero_candidate: dict[str, float | int] | None = None
    for k in recompute_lengths:
        recompute_ratio = k / item_count if item_count else 0.0
        op_scale = ir_compute_scale(calibration)
        full_compute_scale = op_scale * applied_context["compute_scale_mult"]
        recompute_compute_scale = full_compute_scale * float(applied_context["recompute_compute_scale_mult"])
        pre_cached_compute_scale = recompute_compute_scale * float(applied_context["pre_cached_compute_scale_mult"])
        reuse_ratio = args.kv_reuse_ratio if args.enable_kv_reuse else 0.0
        remaining_items, remaining_actions = item_action_cached_rows_after_recompute(args.kv_len, k)
        action_reuse_ratio = action_reuse_ratio_from_total(args.kv_len, action_count, reuse_ratio)
        effective_action_rows = compressed_rows_for_ratio(remaining_actions, action_reuse_ratio)
        effective_cached_rows = effective_cached_rows_after_item_recompute_action_reuse(args.kv_len, k, reuse_ratio)
        cached_kv_len = max(0, args.kv_len - k)
        action_compute_ratio = (
            unreused_action_compute_ratio(reuse_ratio) if args.enable_kv_reuse else 0.0
        )
        history_action_compute_rows = history_action_compute_rows_after_recompute(
            k, reuse_ratio
        )
        active_tokens = C + k
        early_scores, cached_scores = split_recompute_attention_score_elements(
            cached_kv_len,
            k,
            C,
            reuse_ratio,
        )
        Temb = batch * k * H * s / Bemb
        remaining_kv_rows = batch * effective_cached_rows
        Tkv_layer = remaining_kv_rows * 2 * H * s / Bkv
        Tkv = L * Tkv_layer
        Tcand = batch * C * H * s / Bcand
        Tw = 0.0 if args.weights_resident else L * Mw * H * H * s / Bw
        Tmem = Temb + Tkv + Tcand + Tw

        # Compute is modeled per user, then scaled by the observed effective
        # batch factor. This mirrors the trace generator's split recompute
        # attention model and lets AR reduce cached KV preload, cached-late
        # attention work, and HBM-side cached KV traffic together.
        Tcube_late_user = (8 * C * H * H + 4 * cached_scores * H) / Fcube
        # The early side contains a fixed candidate/self-attention term even
        # at k=0 plus the true history-recompute increment. Context calibration
        # should only stretch the increment; otherwise a 910C recompute
        # correction would incorrectly inflate Full-Cache / pure-AR baselines.
        base_early_scores = C * C
        recompute_early_scores = max(0, early_scores - base_early_scores)
        Tcube_early_base_user = 4 * base_early_scores * H / Fcube
        Tcube_early_recompute_user = (8 * k * H * H + 4 * recompute_early_scores * H) / Fcube
        Tvec_late_user = (cached_scores + 2 * C * H) / Fvec
        Tvec_early_base_user = base_early_scores / Fvec
        Tvec_early_recompute_user = (recompute_early_scores + 2 * k * H) / Fvec
        Tcore_late_user = s * (C * H + 2 * effective_cached_rows * H) / Bcore
        Tcore_early_base_user = s * (2 * C * H) / Bcore
        Tcore_early_recompute_user = s * (7 * k * H) / Bcore
        Tar_split_merge_layer = 0.0
        if args.enable_kv_reuse and k > 0 and effective_cached_rows > 0:
            # Split IR+AR attention produces early/cached AV tensors and a
            # merge elementwise op. Once AR shrinks cached attention, this
            # fixed active-token cost is visible on fast KV media such as DDR.
            ar_split_merge_scale = applied_context["ar_split_merge_scale"]
            if ar_split_merge_scale is None:
                ar_split_merge_scale = ir_ar_split_merge_scale(calibration, kv_medium)
            Tar_split_merge_layer = (
                ar_split_merge_scale
                * recompute_compute_scale
                * batch_eff
                * (3 * active_tokens * H * s / Bcore)
            )

        Tcube_rank_layer = full_compute_scale * batch_eff * Tcube_late_user
        Tcube_rec_layer = batch_eff * (
            full_compute_scale * Tcube_early_base_user
            + recompute_compute_scale * Tcube_early_recompute_user
        )
        Tvec_rank_layer = full_compute_scale * batch_eff * Tvec_late_user
        Tvec_rec_layer = batch_eff * (
            full_compute_scale * Tvec_early_base_user
            + recompute_compute_scale * Tvec_early_recompute_user
        )
        Tcore_rank_layer = full_compute_scale * batch_eff * Tcore_late_user
        Tcore_rec_layer = batch_eff * (
            full_compute_scale * Tcore_early_base_user
            + recompute_compute_scale * Tcore_early_recompute_user
        )

        Tpre_cached_layer = batch_eff * (
            full_compute_scale
            * (Tcube_early_base_user + Tvec_early_base_user + Tcore_early_base_user)
            + pre_cached_compute_scale
            * (Tcube_early_recompute_user + Tvec_early_recompute_user + Tcore_early_recompute_user)
        )

        w_both_compute_correction, w_both_compute_correction_detail = (
            w_both_compute_correction_factor(
                calibration,
                H,
                bool(args.enable_kv_reuse),
                k,
            )
        )
        total_compute_correction = compute_correction * w_both_compute_correction

        Tcube_rank_layer *= total_compute_correction
        Tcube_rec_layer *= total_compute_correction
        Tvec_rank_layer *= total_compute_correction
        Tvec_rec_layer *= total_compute_correction
        Tcore_rank_layer *= total_compute_correction
        Tcore_rec_layer *= total_compute_correction
        Tpre_cached_layer *= total_compute_correction
        Tar_split_merge_layer *= total_compute_correction

        Tcube_layer = Tcube_rank_layer + Tcube_rec_layer
        Tvec_layer = Tvec_rank_layer + Tvec_rec_layer
        Tcore_layer = Tcore_rank_layer + Tcore_rec_layer
        Tcube = L * Tcube_layer
        Tvec = L * Tvec_layer
        Tcore = L * Tcore_layer
        Tnpu = Tcube + Tvec + Tcore
        Tlat = max(Tmem, Tnpu)
        Tearly_layer = Tcube_rec_layer + Tvec_rec_layer + Tcore_rec_layer
        Tlate_layer = Tcube_rank_layer + Tvec_rank_layer + Tcore_rank_layer + Tar_split_merge_layer
        Tcompute_layer = Tearly_layer + Tlate_layer
        pipeline_layer = max(Tkv_layer, Tpre_cached_layer) + Tlate_layer
        steady_layer = max(Tkv_layer, Tcompute_layer)
        whole_layer_balance = abs(Tkv_layer - Tcompute_layer)
        overlap_balance = abs(Tkv_layer - Tpre_cached_layer)
        layer_kv_preload_us = Tkv_layer * 1e6
        layer_compute_us = Tcompute_layer * 1e6
        compute_minus_preload_us = layer_compute_us - layer_kv_preload_us
        overrun_tolerance_us = compute_preload_tolerance_us(args, layer_kv_preload_us)
        compute_preload_overrun_us = max(0.0, compute_minus_preload_us - overrun_tolerance_us)
        if args.objective == "balance":
            objective_value = whole_layer_balance
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
            "layer_kv_preload_us": layer_kv_preload_us,
            "layer_early_compute_us": Tearly_layer * 1e6,
            "layer_late_compute_us": Tlate_layer * 1e6,
            "ar_split_merge_us": Tar_split_merge_layer * 1e6,
            "layer_compute_us": layer_compute_us,
            "layer_compute_minus_preload_us": compute_minus_preload_us,
            "compute_preload_tolerance_us": overrun_tolerance_us,
            "compute_preload_overrun_us": compute_preload_overrun_us,
            "compute_preload_safe": 1 if compute_preload_overrun_us <= 0.0 else 0,
            "pre_cached_compute_us": Tpre_cached_layer * 1e6,
            "pipeline_layer_us": pipeline_layer * 1e6,
            "latency_proxy_us": pipeline_layer * 1e6,
            "steady_layer_us": steady_layer * 1e6,
            "objective_value_us": objective_value * 1e6,
            "objective_layer_us": objective_value * 1e6,
            "objective": args.objective,
            "layer_balance_error_us": whole_layer_balance * 1e6,
            "preload_overlap_error_us": overlap_balance * 1e6,
            "preload_overlap_gap_us": (Tkv_layer - Tpre_cached_layer) * 1e6,
            "item_rows": item_count * batch,
            "action_rows": action_count * batch,
            "remaining_item_rows": remaining_items * batch,
            "remaining_action_rows": remaining_actions * batch,
            "effective_action_rows": effective_action_rows * batch,
            "effective_cached_rows": effective_cached_rows * batch,
            "action_reuse_ratio": action_reuse_ratio,
            "history_action_compute_rows": history_action_compute_rows * batch,
            "unreused_action_compute_ratio": action_compute_ratio,
            "active_tokens": active_tokens,
            "early_attention_score_elements": early_scores * batch,
            "cached_attention_score_elements": cached_scores * batch,
            "Sh": Sh,
            "Sr": Sr,
            "num_cores": int(hw["num_cores"]),
            "chip": hw["chip"],
            "source_medium": source_medium,
            "precision_bytes": s,
            "weights_resident": args.weights_resident,
            "effective_batch": batch_eff,
            "compute_scale": full_compute_scale,
            "base_compute_scale": op_scale,
            "recompute_compute_scale": recompute_compute_scale,
            "pre_cached_compute_scale": pre_cached_compute_scale,
            "B_ddr": hw["B_ddr"],
            "B_ssd": hw["B_ssd"],
            "B_hbm": hw["B_hbm"],
            "B_core": hw["B_core"],
            "B_kv": Bkv,
            "B_emb": Bemb,
            "B_cand": Bcand,
            "B_weight": Bw,
            "B_core_used": Bcore,
            "B_ddr_peak": hw["B_ddr_peak"],
            "B_ssd_peak": hw["B_ssd_peak"],
            "B_hbm_peak": hw["B_hbm_peak"],
            "B_kv_calibration": Bkv_cal,
            "B_emb_calibration": Bemb_cal,
            "B_cand_calibration": Bcand_cal,
            "B_weight_calibration": Bw_cal,
            "B_core_calibration": Bcore_cal,
            "preload_peak_util": applied_context["preload_peak_util"],
            "hbm_peak_util": applied_context["hbm_peak_util"],
            "ir_compute_correction": compute_correction,
            "ir_compute_correction_detail": compute_correction_detail,
            "w_both_compute_correction": w_both_compute_correction,
            "w_both_compute_correction_detail": w_both_compute_correction_detail,
            "total_compute_correction": total_compute_correction,
            "calibrated_max_history_recompute_len": search_max_recompute_items,
            "calibration_bound_applied": 1 if calibrated_bound is not None else 0,
            "calibration_bound_source": calibrated_bound_source,
            "compute_scale_mult": applied_context["compute_scale_mult"],
            "recompute_compute_scale_mult": applied_context["recompute_compute_scale_mult"],
            "pre_cached_compute_scale_mult": applied_context["pre_cached_compute_scale_mult"],
            "ar_split_merge_scale": ar_split_merge_scale if args.enable_kv_reuse and k > 0 and effective_cached_rows > 0 else 0.0,
        }
        candidates.append(candidate)
        if k == 0:
            zero_candidate = candidate

    assert candidates

    def selection_key(candidate: dict) -> tuple[float, float, int]:
        return (
            float(candidate["objective_value_us"]),
            float(candidate["steady_layer_us"]),
            int(candidate["history_recompute_len"]),
        )

    raw_best = min(candidates, key=selection_key)
    if fixed_recompute_len is not None:
        best = raw_best
        best["raw_best_history_recompute_len"] = raw_best["history_recompute_len"]
        best["selection_candidate_count"] = len(candidates)
        best["viable_candidate_count"] = len(candidates)
        best["rejected_compute_overrun_count"] = 0
        best["rejected_baseline_guard_count"] = 0
        best["negative_ir_clamped"] = 0
        return best

    assert zero_candidate is not None
    zero_compute_us = float(zero_candidate["layer_compute_us"])
    zero_preload_us = float(zero_candidate["layer_kv_preload_us"])
    zero_tolerance_us = compute_preload_tolerance_us(args, zero_preload_us)
    ar_baseline_guard = (
        bool(args.enable_kv_reuse)
        and not bool(getattr(args, "allow_negative_ir", False))
        and zero_preload_us <= zero_compute_us + zero_tolerance_us
    )
    if ar_baseline_guard:
        best = zero_candidate
        best["raw_best_history_recompute_len"] = raw_best["history_recompute_len"]
        best["raw_best_objective_value_us"] = raw_best["objective_value_us"]
        best["selection_candidate_count"] = len(candidates)
        best["viable_candidate_count"] = 1
        best["rejected_compute_overrun_count"] = sum(
            1
            for candidate in candidates
            if int(candidate["history_recompute_len"]) != 0
        )
        best["rejected_baseline_guard_count"] = 0
        best["baseline_latency_proxy_us"] = best["latency_proxy_us"]
        best["baseline_latency_guard_limit_us"] = best["latency_proxy_us"]
        best["selected_latency_proxy_us"] = best["latency_proxy_us"]
        best["ar_baseline_compute_guard"] = 1
        best["negative_ir_clamped"] = 1 if int(raw_best["history_recompute_len"]) != 0 else 0
        if best["negative_ir_clamped"]:
            best["clamped_from_history_recompute_len"] = raw_best["history_recompute_len"]
            best["clamped_from_steady_layer_us"] = raw_best["steady_layer_us"]
            best["clamped_from_latency_proxy_us"] = raw_best["latency_proxy_us"]
        return best

    allow_overrun = bool(getattr(args, "allow_compute_overrun", False)) or bool(getattr(args, "allow_negative_ir", False))
    allow_baseline_regression = bool(getattr(args, "allow_baseline_regression", False)) or bool(
        getattr(args, "allow_negative_ir", False)
    )
    min_gain = float(getattr(args, "min_ir_gain", 0.05))
    baseline_proxy_us = float(zero_candidate["latency_proxy_us"])
    baseline_limit_us = baseline_proxy_us * (1.0 - min_gain)

    viable: list[dict] = []
    rejected_compute_overrun = 0
    rejected_baseline_guard = 0
    for candidate in candidates:
        k = int(candidate["history_recompute_len"])
        if k == 0:
            viable.append(candidate)
            continue
        if not allow_overrun and float(candidate["compute_preload_overrun_us"]) > 0.0:
            rejected_compute_overrun += 1
            continue
        if not allow_baseline_regression and float(candidate["latency_proxy_us"]) >= baseline_limit_us:
            rejected_baseline_guard += 1
            continue
        viable.append(candidate)

    best = min(viable, key=selection_key) if viable else zero_candidate
    best["raw_best_history_recompute_len"] = raw_best["history_recompute_len"]
    best["raw_best_objective_value_us"] = raw_best["objective_value_us"]
    best["selection_candidate_count"] = len(candidates)
    best["viable_candidate_count"] = len(viable)
    best["rejected_compute_overrun_count"] = rejected_compute_overrun
    best["rejected_baseline_guard_count"] = rejected_baseline_guard
    best["baseline_latency_proxy_us"] = baseline_proxy_us
    best["baseline_latency_guard_limit_us"] = baseline_limit_us
    best["selected_latency_proxy_us"] = best["latency_proxy_us"]
    if int(best["history_recompute_len"]) == 0 and int(raw_best["history_recompute_len"]) != 0:
        best["negative_ir_clamped"] = 1
        best["clamped_from_history_recompute_len"] = raw_best["history_recompute_len"]
        best["clamped_from_steady_layer_us"] = raw_best["steady_layer_us"]
        best["clamped_from_latency_proxy_us"] = raw_best["latency_proxy_us"]
    else:
        best["negative_ir_clamped"] = 0
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
    parser.add_argument("--fixed-recompute-len", type=int, default=None)
    parser.add_argument(
        "--use-calibrated-len",
        action="store_true",
        help="Use an exact context history_recompute_len from calibration when it matches the current model shape.",
    )
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
    parser.add_argument(
        "--allow-negative-ir",
        action="store_true",
        help="Allow a nonzero IR choice even when safety guards would prefer k=0.",
    )
    parser.add_argument("--ignore-calibrated-bound", action="store_true", help="Do not cap k by calibration context bounds.")
    parser.add_argument("--allow-compute-overrun", action="store_true", help="Allow candidates whose compute exceeds KV preload.")
    parser.add_argument(
        "--allow-baseline-regression",
        action="store_true",
        help="Allow candidates whose latency proxy is not better than k=0 baseline.",
    )
    parser.add_argument("--min-ir-gain", type=float, default=0.05, help="Required latency-proxy gain over the k=0 baseline.")
    parser.add_argument(
        "--compute-preload-tolerance-us",
        type=float,
        default=5.0,
        help="Absolute tolerance for layer compute exceeding KV preload.",
    )
    parser.add_argument(
        "--compute-preload-tolerance-ratio",
        type=float,
        default=0.05,
        help="Relative tolerance for layer compute exceeding KV preload.",
    )
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
