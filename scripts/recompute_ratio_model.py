#!/usr/bin/env python3
# Example usage:
#
#   python3 -B docs/scripts/recompute_ratio_model.py \
#     --user cold \
#     --layers 4 \
#     --hidden 256 \
#     --kv-len 4096 \
#     --batch-size 4 \
#     --field json
#
# Example output:
#
#   {
#     "history_recompute_len": 1854,
#     "recompute_ratio": 0.9052734375,
#     "predicted_end_time_us": 8106.6964800000005,
#     "effective_cached_rows": 2242,
#     "target_kv_preload_us": 1412.46,
#     "target_layer_compute_us": 1412.3132799999998,
#     "balance_error_us": 0.1467200000001867
#   }
#
# W_both / KV reuse example:
#
#   python3 -B docs/scripts/recompute_ratio_model.py \
#     --user cold \
#     --layers 4 \
#     --hidden 256 \
#     --kv-len 4096 \
#     --batch-size 4 \
#     --enable-kv-reuse \
#     --kv-reuse-ratio 0.4791 \
#     --field len
#

import argparse
import json
import math


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def effective_cached_rows(kv_len, recompute_len, kv_reuse_ratio):
    item_count = (kv_len + 1) // 2
    action_count = kv_len // 2
    remaining_items = max(0, item_count - recompute_len)
    if kv_reuse_ratio <= 0 or action_count == 0:
        return remaining_items + action_count
    action_reuse_ratio = clamp(kv_reuse_ratio * kv_len / action_count, 0.0, 1.0)
    remaining_actions = max(1, round(action_count * (1.0 - action_reuse_ratio)))
    return remaining_items + remaining_actions


def predict_end_time_us(args, recompute_len):
    hidden_scale = args.hidden / 256.0
    batch_scale = args.batch_size
    item_count = (args.kv_len + 1) // 2
    recompute_len = clamp(recompute_len, 0, item_count)
    cached_rows = effective_cached_rows(
        args.kv_len, recompute_len, args.kv_reuse_ratio if args.enable_kv_reuse else 0.0
    )

    # Calibrated from the current simulator/hardware setup at HSTU-small,
    # candidates=128. These are model constants, not per-experiment sweeps.
    candidate_embedding_us = 80.0 * hidden_scale * batch_scale
    history_embedding_us = 0.0977 * recompute_len * hidden_scale * batch_scale
    ssd_kv_us_per_row = 0.1575 * hidden_scale * batch_scale
    ddr_kv_us_per_row = 0.04545 * hidden_scale * batch_scale

    kv_preload_us = cached_rows * (
        ddr_kv_us_per_row if args.user == "hot" else ssd_kv_us_per_row
    )
    if args.user == "hot":
        layer0_preload_us = max(candidate_embedding_us + history_embedding_us, kv_preload_us)
    else:
        layer0_preload_us = candidate_embedding_us + history_embedding_us + kv_preload_us

    per_layer_compute_us = layer_compute_us(args, recompute_len, cached_rows)

    later_layer_us = max(kv_preload_us, per_layer_compute_us)
    return layer0_preload_us + args.layers * per_layer_compute_us + max(
        0, args.layers - 1
    ) * max(0.0, later_layer_us - per_layer_compute_us)


def kv_preload_us(args, cached_rows):
    hidden_scale = args.hidden / 256.0
    batch_scale = args.batch_size
    ssd_kv_us_per_row = 0.1575 * hidden_scale * batch_scale
    ddr_kv_us_per_row = 0.04545 * hidden_scale * batch_scale
    return cached_rows * (
        ddr_kv_us_per_row if args.user == "hot" else ssd_kv_us_per_row
    )


def layer_compute_us(args, recompute_len, cached_rows):
    hidden_scale = args.hidden / 256.0
    batch_scale = args.batch_size
    # Wall-clock per-layer compute model calibrated from non-layer0 W_IR
    # timelines. This includes compute-side movin effects and is intended to
    # match the layer compute span, not the summed work across cores.
    base_compute_us = 83.22
    recompute_compute_us_per_row = 0.12305
    cached_compute_us_per_row = 0.01861
    return (
        base_compute_us
        + recompute_compute_us_per_row * recompute_len
        + cached_compute_us_per_row * cached_rows
    ) * hidden_scale * batch_scale


def choose_ratio(args):
    item_count = (args.kv_len + 1) // 2
    best = None
    for recompute_len in range(item_count + 1):
        cached_rows = effective_cached_rows(
            args.kv_len,
            recompute_len,
            args.kv_reuse_ratio if args.enable_kv_reuse else 0.0,
        )
        preload_us = kv_preload_us(args, cached_rows)
        compute_us = layer_compute_us(args, recompute_len, cached_rows)
        balance_error_us = abs(preload_us - compute_us)
        predicted_us = predict_end_time_us(args, recompute_len)
        if (
            best is None
            or balance_error_us < best["balance_error_us"]
            or (
                math.isclose(balance_error_us, best["balance_error_us"])
                and predicted_us < best["predicted_end_time_us"]
            )
        ):
            best = {
                "history_recompute_len": recompute_len,
                "recompute_ratio": recompute_len / item_count if item_count else 0.0,
                "predicted_end_time_us": predicted_us,
                "effective_cached_rows": cached_rows,
                "target_kv_preload_us": preload_us,
                "target_layer_compute_us": compute_us,
                "balance_error_us": balance_error_us,
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", choices=["hot", "cold"], required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--enable-kv-reuse", action="store_true")
    parser.add_argument("--kv-reuse-ratio", type=float, default=0.0)
    parser.add_argument("--field", choices=["json", "len", "ratio"], default="json")
    args = parser.parse_args()
    result = choose_ratio(args)
    if args.field == "len":
        print(result["history_recompute_len"])
    elif args.field == "ratio":
        print(result["recompute_ratio"])
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
