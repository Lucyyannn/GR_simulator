#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def tensor(name, shape, dtype="float16", is_weight=False, **meta):
    data = {
        "name": name,
        "shape": list(shape),
        "dtype": dtype,
    }
    if is_weight:
        data["is_weight"] = True
    data.update({k: v for k, v in meta.items() if v is not None})
    return data


def hbm_tensor(name, shape, dtype="float16", **meta):
    return tensor(name, shape, dtype=dtype, runtime_medium="hbm", **meta)


def ddr_tensor(name, shape, dtype="float16", is_weight=False, **meta):
    return tensor(
        name,
        shape,
        dtype=dtype,
        is_weight=is_weight,
        initial_medium="ddr",
        runtime_medium="ddr",
        **meta,
    )


def ddr_to_hbm_tensor(name, shape, dtype="float16", is_weight=False, **meta):
    return tensor(
        name,
        shape,
        dtype=dtype,
        is_weight=is_weight,
        initial_medium="ddr",
        runtime_medium="hbm",
        **meta,
    )


def source_tensor(source_medium, name, shape, dtype="float16", is_weight=False, **meta):
    return tensor(
        name,
        shape,
        dtype=dtype,
        is_weight=is_weight,
        initial_medium=source_medium,
        runtime_medium=source_medium,
        **meta,
    )


def source_to_hbm_tensor(
    source_medium, name, shape, dtype="float16", is_weight=False, **meta
):
    return tensor(
        name,
        shape,
        dtype=dtype,
        is_weight=is_weight,
        initial_medium=source_medium,
        runtime_medium="hbm",
        **meta,
    )


def add_op(ops, name, inputs, outputs, attrs=None):
    ops.append(
        {
            "id": len(ops),
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "attrs": attrs or {},
        }
    )


def common_meta(user_id, batch_id, macro_batch_id, role=None, layer_id=None):
    meta = {
        "user_id": user_id,
        "batch_id": batch_id,
        "macro_batch_id": macro_batch_id,
    }
    if role is not None:
        meta["role"] = role
    if layer_id is not None:
        meta["layer_id"] = layer_id
    return meta


def op_modeling_attrs(op_modeling, op_name):
    mode = op_modeling.get(op_name)
    if mode is None:
        return {}
    return {"modeling_mode": mode}


def build_action_reuse_mapping(
    length,
    action_count,
    rng,
    action_offset=1,
    action_stride=2,
):
    action_positions = list(range(action_offset, length, action_stride))
    if (
        action_count is None
        or action_count <= 0
        or action_count >= len(action_positions)
    ):
        return None

    action_count = min(action_count, len(action_positions))
    action_ids = list(range(action_count))
    action_ids.extend(
        rng.randrange(action_count) for _ in range(len(action_positions) - action_count)
    )

    action_id_by_position = dict(zip(action_positions, action_ids))
    row_to_physical = {}
    logical_to_physical = []
    for logical_row in range(length):
        if logical_row in action_id_by_position:
            row_key = ("action", action_id_by_position[logical_row])
        else:
            row_key = ("item", logical_row)

        if row_key not in row_to_physical:
            row_to_physical[row_key] = len(row_to_physical)
        logical_to_physical.append(row_to_physical[row_key])

    return {
        "reuse_mode": "row_reuse",
        "reuse_axis": 0,
        "reuse_physical_rows": len(row_to_physical),
        "reuse_logical_to_physical": logical_to_physical,
        "reuse_action_positions": action_positions,
        "reuse_action_ids": action_ids,
        "reuse_action_offset": action_offset,
        "reuse_action_stride": action_stride,
    }


def _window_topk_hot_quotas(action_count, topk, hot_total):
    if topk <= 0:
        return []

    topk = min(topk, action_count, hot_total)
    if topk <= 0:
        return []

    weights = list(range(topk, 0, -1))
    quotas = [1] * topk
    remaining = hot_total - topk
    if remaining > 0:
        weight_sum = sum(weights)
        extras = [remaining * w // weight_sum for w in weights]
        quotas = [quota + extra for quota, extra in zip(quotas, extras)]
        leftover = hot_total - sum(quotas)
        for i in range(leftover):
            quotas[i % topk] += 1
    return quotas


def build_window_topk_reuse_mapping(
    length,
    rng,
    window_size=1024,
    topk=4,
    action_offset=1,
    action_stride=2,
    hot_share=0.75,
):
    if window_size is None or window_size <= 0:
        return None
    if topk is None or topk <= 0:
        return None

    action_positions = list(range(action_offset, length, action_stride))
    if not action_positions:
        return None
    action_position_set = set(action_positions)

    row_to_physical = {}
    logical_to_physical = []
    reuse_action_positions = []
    reuse_action_ids = []
    window_topk_actions = []
    for window_id, window_start in enumerate(range(0, length, window_size)):
        window_end = min(window_start + window_size, length)
        window_action_positions = [
            pos for pos in action_positions if window_start <= pos < window_end
        ]
        window_action_count = len(window_action_positions)

        if window_action_count == 0:
            for logical_row in range(window_start, window_end):
                row_key = ("item", logical_row)
                if row_key not in row_to_physical:
                    row_to_physical[row_key] = len(row_to_physical)
                logical_to_physical.append(row_to_physical[row_key])
            continue

        local_topk = min(topk, window_action_count)
        if window_action_count <= local_topk:
            hot_total = window_action_count
        else:
            hot_total = min(
                window_action_count,
                max(local_topk + 1, int(round(window_action_count * hot_share))),
            )

        quotas = _window_topk_hot_quotas(window_action_count, local_topk, hot_total)
        assigned_action_ids = []
        for action_id, quota in enumerate(quotas):
            assigned_action_ids.extend([action_id] * quota)
        cold_count = window_action_count - len(assigned_action_ids)
        for cold_idx in range(cold_count):
            assigned_action_ids.append(local_topk + cold_idx)

        window_rng = random.Random(rng.randrange(1 << 63))
        window_rng.shuffle(assigned_action_ids)
        action_id_by_position = dict(zip(window_action_positions, assigned_action_ids))

        counts = Counter(assigned_action_ids)
        top_action_ids = [
            action_id
            for action_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                :local_topk
            ]
        ]
        window_topk_actions.append(
            {
                "window_id": window_id,
                "window_start": window_start,
                "window_end": window_end,
                "action_count": window_action_count,
                "topk": local_topk,
                "selected_action_ids": top_action_ids,
            }
        )

        for logical_row in range(window_start, window_end):
            if logical_row in action_position_set:
                action_id = action_id_by_position[logical_row]
                reuse_action_positions.append(logical_row)
                reuse_action_ids.append(action_id)
                if action_id in top_action_ids:
                    row_key = ("action", window_id, action_id)
                else:
                    row_key = ("cold_action", window_id, logical_row)
            else:
                row_key = ("item", logical_row)

            if row_key not in row_to_physical:
                row_to_physical[row_key] = len(row_to_physical)
            logical_to_physical.append(row_to_physical[row_key])

    return {
        "reuse_mode": "row_reuse",
        "reuse_variant": "window_topk",
        "reuse_axis": 0,
        "reuse_physical_rows": len(row_to_physical),
        "reuse_logical_to_physical": logical_to_physical,
        "reuse_action_positions": reuse_action_positions,
        "reuse_action_ids": reuse_action_ids,
        "reuse_window_size": window_size,
        "reuse_topk": topk,
        "reuse_hot_share": hot_share,
        "reuse_window_topk_actions": window_topk_actions,
    }


def build_kv_reuse_mapping(
    length,
    rng,
    variant="window_topk",
    action_count=4,
    window_size=1024,
    topk=4,
    hot_share=0.75,
    action_offset=1,
    action_stride=2,
):
    if variant == "global":
        mapping = build_action_reuse_mapping(
            length,
            action_count,
            rng,
            action_offset=action_offset,
            action_stride=action_stride,
        )
        if mapping is not None:
            mapping["reuse_variant"] = "global"
            mapping["reuse_action_count"] = action_count
        return mapping
    if variant == "window_topk":
        return build_window_topk_reuse_mapping(
            length,
            rng,
            window_size=window_size,
            topk=topk,
            hot_share=hot_share,
            action_offset=action_offset,
            action_stride=action_stride,
        )
    raise ValueError(f"Unsupported KV reuse variant: {variant}")


def recompute_attention_score_elements(
    batch_size, cached_kv_len, history_recompute_len, candidate_tokens
):
    if history_recompute_len <= 0:
        return batch_size * candidate_tokens * (cached_kv_len + candidate_tokens)

    history_scores = (
        history_recompute_len * cached_kv_len
        + history_recompute_len * (history_recompute_len + 1) // 2
    )
    candidate_scores = candidate_tokens * (
        cached_kv_len + history_recompute_len + candidate_tokens
    )
    return batch_size * (history_scores + candidate_scores)


def attention_mask_mode(history_recompute_len):
    return (
        "suffix_history_causal_candidate_full"
        if history_recompute_len
        else "baseline_full_current"
    )


def build_trace(
    layers,
    tokens,
    hidden,
    kv_len,
    vocab,
    user_id=0,
    batch_id=0,
    macro_batch_id=0,
    indices_values=None,
    model_name="hstu_8layer_baseline_small",
    op_modeling=None,
    attention_modeling="decomposed",
    pipeline_enabled=False,
    kv_reuse_enabled=False,
    kv_reuse_variant="window_topk",
    kv_reuse_action_count=4,
    kv_reuse_window_size=1024,
    kv_reuse_topk=4,
    kv_reuse_hot_share=0.75,
    kv_reuse_action_offset=1,
    kv_reuse_action_stride=2,
    source_medium="ddr",
    seed=0,
    history_recompute_len=0,
):
    op_modeling = op_modeling or {}
    if source_medium not in {"ddr", "ssd"}:
        raise ValueError(f"Unsupported source medium: {source_medium}")
    if history_recompute_len < 0 or history_recompute_len > kv_len:
        raise ValueError("history_recompute_len must be in [0, kv_len]")
    candidate_tokens = tokens
    cached_kv_len = kv_len - history_recompute_len
    active_tokens = history_recompute_len + candidate_tokens
    score_elements = recompute_attention_score_elements(
        1, cached_kv_len, history_recompute_len, candidate_tokens
    )
    reuse_rng = random.Random(seed + user_id * 1000003 + batch_id * 9176 + macro_batch_id)
    ops = []
    indices_values = indices_values or [i % vocab for i in range(active_tokens)]
    if len(indices_values) != active_tokens:
        raise ValueError("indices_values length must match active token count")
    kv_reuse_meta = (
        build_kv_reuse_mapping(
            cached_kv_len,
            reuse_rng,
            variant=kv_reuse_variant,
            action_count=kv_reuse_action_count,
            window_size=kv_reuse_window_size,
            topk=kv_reuse_topk,
            hot_share=kv_reuse_hot_share,
            action_offset=kv_reuse_action_offset,
            action_stride=kv_reuse_action_stride,
        )
        if kv_reuse_enabled
        else None
    )

    base = common_meta(user_id, batch_id, macro_batch_id)
    add_op(
        ops,
        "aten::embedding",
        [
            source_tensor(
                source_medium,
                "embedding_table",
                [vocab, hidden],
                is_weight=True,
                logical_id="embedding.table",
                role="embedding_table",
                **base,
            ),
            hbm_tensor(
                f"u{user_id}.b{batch_id}.m{macro_batch_id}.candidate_ids",
                [active_tokens],
                dtype="int64",
                role="indices",
                **base,
            ),
        ],
        [
            source_to_hbm_tensor(
                source_medium,
                f"u{user_id}.b{batch_id}.m{macro_batch_id}.x0",
                [active_tokens, hidden],
                logical_id=f"u{user_id}.b{batch_id}.m{macro_batch_id}.embedding_rows",
                role="embedding_rows",
                preload_group="pre_attention",
                source_logical_id="embedding.table",
                source_shape=[vocab, hidden],
                indices_values=indices_values,
                **base,
            )
        ],
        {
            "indices_values": ",".join(str(v) for v in indices_values),
            "modeling_mode": "preloaded_rows",
            "history_recompute_len": history_recompute_len,
            "candidate_tokens": candidate_tokens,
        },
    )

    current = f"u{user_id}.b{batch_id}.m{macro_batch_id}.x0"
    for layer in range(layers):
        prefix = f"u{user_id}.b{batch_id}.m{macro_batch_id}.layer{layer}"
        shared_layer = f"layer{layer}"
        z = f"{prefix}.z"
        zact = f"{prefix}.zact"
        u = f"{prefix}.u"
        v = f"{prefix}.v"
        q = f"{prefix}.q"
        k = f"{prefix}.k"
        k_cache = f"user{user_id}.{shared_layer}.kc"
        v_cache = f"user{user_id}.{shared_layer}.vc"
        k_all = f"{prefix}.k_all"
        k_all_t = f"{prefix}.k_all_t"
        score = f"{prefix}.score"
        attn = f"{prefix}.attn"
        v_all = f"{prefix}.v_all"
        av = f"{prefix}.av"
        av_norm = f"{prefix}.av_norm"
        gated = f"{prefix}.gated"
        out = f"u{user_id}.b{batch_id}.m{macro_batch_id}.x{layer + 1}"
        layer_meta = common_meta(user_id, batch_id, macro_batch_id, layer_id=layer)

        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(current, [active_tokens, hidden], role="activation", **layer_meta),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.w1",
                    [hidden, hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w1",
                    role="weight",
                    preload_group="pre_attention",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.b1",
                    [hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b1",
                    role="weight",
                    preload_group="pre_attention",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(z, [active_tokens, hidden * 4], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(z, [active_tokens, hidden * 4], role="activation", **layer_meta)],
            [hbm_tensor(zact, [active_tokens, hidden * 4], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::split",
            [hbm_tensor(zact, [active_tokens, hidden * 4], role="activation", **layer_meta)],
            [
                hbm_tensor(u, [active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(v, [active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(q, [active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(k, [active_tokens, hidden], role="activation", **layer_meta),
            ],
            {"axis": 1, **op_modeling_attrs(op_modeling, "split")},
        )
        if attention_modeling == "fused":
            add_op(
                ops,
                "hstu::attention",
                [
                    hbm_tensor(q, [active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(k, [active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(v, [active_tokens, hidden], role="activation", **layer_meta),
                    source_to_hbm_tensor(
                        source_medium,
                        k_cache,
                        [cached_kv_len, hidden],
                        logical_id=f"user{user_id}.{shared_layer}.kc",
                        role="kv_cache_k",
                        preload_group="kvcache",
                        **(kv_reuse_meta or {}),
                        **layer_meta,
                    ),
                    source_to_hbm_tensor(
                        source_medium,
                        v_cache,
                        [cached_kv_len, hidden],
                        logical_id=f"user{user_id}.{shared_layer}.vc",
                        role="kv_cache_v",
                        preload_group="kvcache",
                        **(kv_reuse_meta or {}),
                        **layer_meta,
                    ),
                ],
                [hbm_tensor(av, [active_tokens, hidden], role="activation", **layer_meta)],
                {
                    "kv_axis": 0,
                    "logical_kv_len": cached_kv_len + active_tokens,
                    "current_tokens": active_tokens,
                    "candidate_tokens": candidate_tokens,
                    "history_recompute_len": history_recompute_len,
                    "cached_kv_len": cached_kv_len,
                    "mask_mode": attention_mask_mode(history_recompute_len),
                    "attention_score_elements": score_elements,
                    "hidden": hidden,
                    "batch_size": 1,
                },
            )
        else:
            add_op(
                ops,
                "aten::cat",
                [
                    source_to_hbm_tensor(
                        source_medium,
                        k_cache,
                        [cached_kv_len, hidden],
                        logical_id=f"user{user_id}.{shared_layer}.kc",
                        role="kv_cache_k",
                        preload_group="kvcache",
                        **(kv_reuse_meta or {}),
                        **layer_meta,
                    ),
                    hbm_tensor(k, [active_tokens, hidden], role="activation", **layer_meta),
                ],
                [hbm_tensor(k_all, [cached_kv_len + active_tokens, hidden], role="activation", **layer_meta)],
                {"axis": 0, **op_modeling_attrs(op_modeling, "concat")},
            )
            add_op(
                ops,
                "aten::transpose",
                [hbm_tensor(k_all, [cached_kv_len + active_tokens, hidden], role="activation", **layer_meta)],
                [hbm_tensor(k_all_t, [hidden, cached_kv_len + active_tokens], role="activation", **layer_meta)],
                {"dims": "1,0", **op_modeling_attrs(op_modeling, "view")},
            )
            add_op(
                ops,
                "aten::matmul",
                [
                    hbm_tensor(q, [active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(k_all_t, [hidden, cached_kv_len + active_tokens], role="activation", **layer_meta),
                ],
                [hbm_tensor(score, [active_tokens, cached_kv_len + active_tokens], role="activation", **layer_meta)],
            )
            add_op(
                ops,
                "aten::silu",
                [hbm_tensor(score, [active_tokens, cached_kv_len + active_tokens], role="activation", **layer_meta)],
                [hbm_tensor(attn, [active_tokens, cached_kv_len + active_tokens], role="activation", **layer_meta)],
            )
            add_op(
                ops,
                "aten::cat",
                [
                    source_to_hbm_tensor(
                        source_medium,
                        v_cache,
                        [cached_kv_len, hidden],
                        logical_id=f"user{user_id}.{shared_layer}.vc",
                        role="kv_cache_v",
                        preload_group="kvcache",
                        **(kv_reuse_meta or {}),
                        **layer_meta,
                    ),
                    hbm_tensor(v, [active_tokens, hidden], role="activation", **layer_meta),
                ],
                [hbm_tensor(v_all, [cached_kv_len + active_tokens, hidden], role="activation", **layer_meta)],
                {"axis": 0, **op_modeling_attrs(op_modeling, "concat")},
            )
            add_op(
                ops,
                "aten::matmul",
                [
                    hbm_tensor(attn, [active_tokens, cached_kv_len + active_tokens], role="activation", **layer_meta),
                    hbm_tensor(v_all, [cached_kv_len + active_tokens, hidden], role="activation", **layer_meta),
                ],
                [hbm_tensor(av, [active_tokens, hidden], role="activation", **layer_meta)],
            )
        add_op(
            ops,
            "aten::layer_norm",
            [
                hbm_tensor(av, [active_tokens, hidden], role="activation", **layer_meta),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.ln_w",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_w",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.ln_b",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_b",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(av_norm, [active_tokens, hidden], role="activation", **layer_meta)],
            {**op_modeling_attrs(op_modeling, "layer_norm")},
        )
        add_op(
            ops,
            "aten::mul",
            [
                hbm_tensor(av_norm, [active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(u, [active_tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(gated, [active_tokens, hidden], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(gated, [active_tokens, hidden], role="activation", **layer_meta),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.w2",
                    [hidden, hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w2",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.b2",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b2",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(out, [active_tokens, hidden], role="activation", **layer_meta)],
        )
        current = out

    return {
        "metadata": {
            "format_version": "1.0",
            "model_name": model_name,
            "model_type": "hstu_ranking",
            "workload_type": "hstu_ranking",
            "layout": "NHWC",
            "num_layers": layers,
            "user_id": user_id,
            "batch_id": batch_id,
            "macro_batch_id": macro_batch_id,
            "pipeline_enabled": pipeline_enabled,
            "baseline_preload": True,
            "fail_on_unknown_op": True,
            "kv_reuse_enabled": kv_reuse_enabled,
            "random_seed": seed,
            "op_modeling": op_modeling,
            "attention_modeling": attention_modeling,
            "source_medium": source_medium,
            "kv_reuse_variant": kv_reuse_variant,
            "kv_reuse_action_count": kv_reuse_action_count,
            "kv_reuse_window_size": kv_reuse_window_size,
            "kv_reuse_topk": kv_reuse_topk,
            "kv_reuse_hot_share": kv_reuse_hot_share,
            "kv_reuse_action_offset": kv_reuse_action_offset,
            "kv_reuse_action_stride": kv_reuse_action_stride,
            "candidate_tokens": candidate_tokens,
            "history_recompute_len": history_recompute_len,
            "cached_kv_len": cached_kv_len,
            "active_tokens": active_tokens,
            "attention_mask_mode": attention_mask_mode(history_recompute_len),
        },
        "operators": ops,
    }


def build_batched_trace(
    layers,
    tokens,
    hidden,
    kv_len,
    vocab,
    user_ids,
    batch_id=0,
    macro_batch_id=0,
    indices_values_per_user=None,
    model_name="hstu_batched_baseline_small",
    op_modeling=None,
    attention_modeling="decomposed",
    pipeline_enabled=False,
    kv_reuse_enabled=False,
    kv_reuse_variant="window_topk",
    kv_reuse_action_count=4,
    kv_reuse_window_size=1024,
    kv_reuse_topk=4,
    kv_reuse_hot_share=0.75,
    kv_reuse_action_offset=1,
    kv_reuse_action_stride=2,
    source_medium="ddr",
    seed=0,
    history_recompute_len=0,
):
    op_modeling = op_modeling or {}
    if source_medium not in {"ddr", "ssd"}:
        raise ValueError(f"Unsupported source medium: {source_medium}")
    if history_recompute_len < 0 or history_recompute_len > kv_len:
        raise ValueError("history_recompute_len must be in [0, kv_len]")
    candidate_tokens = tokens
    cached_kv_len = kv_len - history_recompute_len
    active_tokens = history_recompute_len + candidate_tokens

    batch_size = len(user_ids)
    if batch_size <= 0:
        raise ValueError("user_ids must be non-empty for build_batched_trace")

    if indices_values_per_user is None:
        indices_values_per_user = [
            [i % vocab for i in range(active_tokens)] for _ in range(batch_size)
        ]
    if len(indices_values_per_user) != batch_size:
        raise ValueError(
            "indices_values_per_user size must match batch_size in build_batched_trace"
        )
    for row in indices_values_per_user:
        if len(row) != active_tokens:
            raise ValueError("Each batched indices row must match active token count")

    flat_indices_values = [
        idx for per_user_indices in indices_values_per_user for idx in per_user_indices
    ]

    def batch_kv_reuse_meta(layer):
        if not kv_reuse_enabled:
            return {}
        mappings = []
        physical_rows = []
        window_topk_actions = []
        for user_id in user_ids:
            reuse_rng = random.Random(seed + user_id * 1000003 + layer * 9176)
            mapping = build_kv_reuse_mapping(
                cached_kv_len,
                reuse_rng,
                variant=kv_reuse_variant,
                action_count=kv_reuse_action_count,
                window_size=kv_reuse_window_size,
                topk=kv_reuse_topk,
                hot_share=kv_reuse_hot_share,
                action_offset=kv_reuse_action_offset,
                action_stride=kv_reuse_action_stride,
            )
            if mapping is None:
                logical_to_physical = list(range(cached_kv_len))
                rows = cached_kv_len
            else:
                logical_to_physical = mapping["reuse_logical_to_physical"]
                rows = mapping["reuse_physical_rows"]
                if "reuse_window_topk_actions" in mapping:
                    window_topk_actions.append(mapping["reuse_window_topk_actions"])
            mappings.append(logical_to_physical)
            physical_rows.append(rows)
        meta = {
            "reuse_mode": "row_reuse",
            "reuse_variant": kv_reuse_variant,
            "reuse_axis": 1,
            "reuse_physical_rows_per_user": physical_rows,
            "reuse_logical_to_physical_per_user": mappings,
            "reuse_action_count": kv_reuse_action_count,
            "reuse_action_offset": kv_reuse_action_offset,
            "reuse_action_stride": kv_reuse_action_stride,
            "reuse_window_size": kv_reuse_window_size,
            "reuse_topk": kv_reuse_topk,
            "reuse_hot_share": kv_reuse_hot_share,
        }
        if window_topk_actions:
            meta["reuse_window_topk_actions_per_user"] = window_topk_actions
        return meta

    ops = []
    base = {
        "batch_id": batch_id,
        "macro_batch_id": macro_batch_id,
        "user_id": user_ids[0],
    }
    add_op(
        ops,
        "aten::embedding",
        [
            source_tensor(
                source_medium,
                "embedding_table",
                [vocab, hidden],
                is_weight=True,
                logical_id="embedding.table",
                role="embedding_table",
                **base,
            ),
            hbm_tensor(
                f"b{batch_id}.m{macro_batch_id}.candidate_ids",
                [batch_size, active_tokens],
                dtype="int64",
                role="indices",
                **base,
            ),
        ],
        [
            source_to_hbm_tensor(
                source_medium,
                f"b{batch_id}.m{macro_batch_id}.x0",
                [batch_size, active_tokens, hidden],
                logical_id=f"batch{batch_id}.macro{macro_batch_id}.embedding_rows",
                role="embedding_rows",
                preload_group="pre_attention",
                source_logical_id="embedding.table",
                source_shape=[vocab, hidden],
                indices_values=flat_indices_values,
                user_ids=user_ids,
                indices_values_per_user=indices_values_per_user,
                **base,
            )
        ],
        {
            "indices_values": ",".join(str(v) for v in flat_indices_values),
            "modeling_mode": "preloaded_rows",
            "history_recompute_len": history_recompute_len,
            "candidate_tokens": candidate_tokens,
        },
    )

    current = f"b{batch_id}.m{macro_batch_id}.x0"
    for layer in range(layers):
        prefix = f"b{batch_id}.m{macro_batch_id}.layer{layer}"
        shared_layer = f"layer{layer}"
        kv_reuse_meta = batch_kv_reuse_meta(layer)
        z = f"{prefix}.z"
        zact = f"{prefix}.zact"
        u = f"{prefix}.u"
        v = f"{prefix}.v"
        q = f"{prefix}.q"
        k = f"{prefix}.k"
        k_cache = f"batch{batch_id}.{shared_layer}.kc"
        v_cache = f"batch{batch_id}.{shared_layer}.vc"
        k_all = f"{prefix}.k_all"
        k_all_t = f"{prefix}.k_all_t"
        score = f"{prefix}.score"
        attn = f"{prefix}.attn"
        v_all = f"{prefix}.v_all"
        av = f"{prefix}.av"
        av_norm = f"{prefix}.av_norm"
        gated = f"{prefix}.gated"
        out = f"b{batch_id}.m{macro_batch_id}.x{layer + 1}"
        layer_meta = {
            "batch_id": batch_id,
            "macro_batch_id": macro_batch_id,
            "user_id": user_ids[0],
            "layer_id": layer,
        }

        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(
                    current, [batch_size, active_tokens, hidden], role="activation", **layer_meta
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.w1",
                    [hidden, hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w1",
                    role="weight",
                    preload_group="pre_attention",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.b1",
                    [hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b1",
                    role="weight",
                    preload_group="pre_attention",
                    **layer_meta,
                ),
            ],
            [
                hbm_tensor(
                    z, [batch_size, active_tokens, hidden * 4], role="activation", **layer_meta
                )
            ],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(z, [batch_size, active_tokens, hidden * 4], role="activation", **layer_meta)],
            [
                hbm_tensor(
                    zact, [batch_size, active_tokens, hidden * 4], role="activation", **layer_meta
                )
            ],
        )
        add_op(
            ops,
            "aten::split",
            [
                hbm_tensor(
                    zact, [batch_size, active_tokens, hidden * 4], role="activation", **layer_meta
                )
            ],
            [
                hbm_tensor(u, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(v, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(q, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(k, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
            ],
            {"axis": 2, **op_modeling_attrs(op_modeling, "split")},
        )
        if attention_modeling == "fused":
            add_op(
                ops,
                "hstu::attention",
                [
                    hbm_tensor(q, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(k, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(v, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                    source_to_hbm_tensor(
                        source_medium,
                        k_cache,
                        [batch_size, cached_kv_len, hidden],
                        logical_id=f"batch{batch_id}.{shared_layer}.kc",
                        role="kv_cache_k_batch",
                        preload_group="kvcache",
                        user_ids=user_ids,
                        **kv_reuse_meta,
                        **layer_meta,
                    ),
                    source_to_hbm_tensor(
                        source_medium,
                        v_cache,
                        [batch_size, cached_kv_len, hidden],
                        logical_id=f"batch{batch_id}.{shared_layer}.vc",
                        role="kv_cache_v_batch",
                        preload_group="kvcache",
                        user_ids=user_ids,
                        **kv_reuse_meta,
                        **layer_meta,
                    ),
                ],
                [hbm_tensor(av, [batch_size, active_tokens, hidden], role="activation", **layer_meta)],
                {
                    "kv_axis": 1,
                    "logical_kv_len": cached_kv_len + active_tokens,
                    "current_tokens": active_tokens,
                    "candidate_tokens": candidate_tokens,
                    "history_recompute_len": history_recompute_len,
                    "cached_kv_len": cached_kv_len,
                    "mask_mode": attention_mask_mode(history_recompute_len),
                    "attention_score_elements": recompute_attention_score_elements(
                        batch_size,
                        cached_kv_len,
                        history_recompute_len,
                        candidate_tokens,
                    ),
                    "hidden": hidden,
                    "batch_size": batch_size,
                },
            )
        else:
            add_op(
                ops,
                "aten::cat",
                [
                    source_to_hbm_tensor(
                        source_medium,
                        k_cache,
                        [batch_size, cached_kv_len, hidden],
                        logical_id=f"batch{batch_id}.{shared_layer}.kc",
                        role="kv_cache_k_batch",
                        preload_group="kvcache",
                        user_ids=user_ids,
                        **kv_reuse_meta,
                        **layer_meta,
                    ),
                    hbm_tensor(k, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                ],
                [
                    hbm_tensor(
                        k_all, [batch_size, cached_kv_len + active_tokens, hidden], role="activation", **layer_meta
                    )
                ],
                {"axis": 1, **op_modeling_attrs(op_modeling, "concat")},
            )
            add_op(
                ops,
                "aten::transpose",
                [
                    hbm_tensor(
                        k_all, [batch_size, cached_kv_len + active_tokens, hidden], role="activation", **layer_meta
                    )
                ],
                [
                    hbm_tensor(
                        k_all_t,
                        [batch_size, hidden, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    )
                ],
                {"dims": "0,2,1", **op_modeling_attrs(op_modeling, "view")},
            )
            add_op(
                ops,
                "aten::matmul",
                [
                    hbm_tensor(q, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                    hbm_tensor(
                        k_all_t,
                        [batch_size, hidden, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    ),
                ],
                [
                    hbm_tensor(
                        score,
                        [batch_size, active_tokens, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    )
                ],
            )
            add_op(
                ops,
                "aten::silu",
                [
                    hbm_tensor(
                        score,
                        [batch_size, active_tokens, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    )
                ],
                [
                    hbm_tensor(
                        attn,
                        [batch_size, active_tokens, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    )
                ],
            )
            add_op(
                ops,
                "aten::cat",
                [
                    source_to_hbm_tensor(
                        source_medium,
                        v_cache,
                        [batch_size, cached_kv_len, hidden],
                        logical_id=f"batch{batch_id}.{shared_layer}.vc",
                        role="kv_cache_v_batch",
                        preload_group="kvcache",
                        user_ids=user_ids,
                        **kv_reuse_meta,
                        **layer_meta,
                    ),
                    hbm_tensor(v, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                ],
                [
                    hbm_tensor(
                        v_all, [batch_size, cached_kv_len + active_tokens, hidden], role="activation", **layer_meta
                    )
                ],
                {"axis": 1, **op_modeling_attrs(op_modeling, "concat")},
            )
            add_op(
                ops,
                "aten::matmul",
                [
                    hbm_tensor(
                        attn,
                        [batch_size, active_tokens, cached_kv_len + active_tokens],
                        role="activation",
                        **layer_meta,
                    ),
                    hbm_tensor(
                        v_all, [batch_size, cached_kv_len + active_tokens, hidden], role="activation", **layer_meta
                    ),
                ],
                [hbm_tensor(av, [batch_size, active_tokens, hidden], role="activation", **layer_meta)],
            )
        add_op(
            ops,
            "aten::layer_norm",
            [
                hbm_tensor(av, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.ln_w",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_w",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.ln_b",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_b",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(av_norm, [batch_size, active_tokens, hidden], role="activation", **layer_meta)],
            {**op_modeling_attrs(op_modeling, "layer_norm")},
        )
        add_op(
            ops,
            "aten::mul",
            [
                hbm_tensor(
                    av_norm, [batch_size, active_tokens, hidden], role="activation", **layer_meta
                ),
                hbm_tensor(u, [batch_size, active_tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(gated, [batch_size, active_tokens, hidden], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(
                    gated, [batch_size, active_tokens, hidden], role="activation", **layer_meta
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.w2",
                    [hidden, hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w2",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
                source_to_hbm_tensor(
                    source_medium,
                    f"{shared_layer}.b2",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b2",
                    role="weight",
                    preload_group="post_attention_weights",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(out, [batch_size, active_tokens, hidden], role="activation", **layer_meta)],
        )
        current = out

    return {
        "metadata": {
            "format_version": "1.0",
            "model_name": model_name,
            "model_type": "hstu_ranking_batched",
            "workload_type": "hstu_ranking_pipeline_batched",
            "layout": "NHWC",
            "num_layers": layers,
            "batch_id": batch_id,
            "macro_batch_id": macro_batch_id,
            "batch_size": batch_size,
            "user_ids": list(user_ids),
            "pipeline_enabled": pipeline_enabled,
            "baseline_preload": True,
            "fail_on_unknown_op": True,
            "kv_reuse_enabled": kv_reuse_enabled,
            "random_seed": seed,
            "op_modeling": op_modeling,
            "attention_modeling": attention_modeling,
            "source_medium": source_medium,
            "kv_reuse_variant": kv_reuse_variant,
            "kv_reuse_action_count": kv_reuse_action_count,
            "kv_reuse_window_size": kv_reuse_window_size,
            "kv_reuse_topk": kv_reuse_topk,
            "kv_reuse_hot_share": kv_reuse_hot_share,
            "kv_reuse_action_offset": kv_reuse_action_offset,
            "kv_reuse_action_stride": kv_reuse_action_stride,
            "candidate_tokens": candidate_tokens,
            "history_recompute_len": history_recompute_len,
            "cached_kv_len": cached_kv_len,
            "active_tokens": active_tokens,
            "attention_mask_mode": attention_mask_mode(history_recompute_len),
        },
        "operators": ops,
    }


def contiguous_batches(num_users, users_per_batch):
    return [
        list(range(start, min(start + users_per_batch, num_users)))
        for start in range(0, num_users, users_per_batch)
    ]


def parse_op_modeling(value):
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        if not item:
            continue
        key, _, mode = item.partition("=")
        if not key or not mode:
            raise ValueError(f"Invalid --op-modeling item: {item}")
        if mode not in {"skip", "materialize"}:
            raise ValueError(f"Invalid modeling mode for {key}: {mode}")
        result[key.strip()] = mode.strip()
    return result


def write_json(path, data, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(
            json.dumps(data, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_single_trace(args, op_modeling):
    rng = random.Random(args.seed)
    active_tokens = args.tokens + args.history_recompute_len
    indices = [rng.randrange(args.vocab) for _ in range(active_tokens)]
    trace = build_trace(
        args.layers,
        args.tokens,
        args.hidden,
        args.kv_len,
        args.vocab,
        indices_values=indices,
        model_name="hstu_8layer_baseline_small",
        op_modeling=op_modeling,
        attention_modeling=args.attention_modeling,
        pipeline_enabled=args.pipeline,
        kv_reuse_enabled=args.enable_kv_reuse,
        kv_reuse_variant=args.kv_reuse_variant,
        kv_reuse_action_count=args.kv_reuse_action_count,
        kv_reuse_window_size=args.kv_reuse_window_size,
        kv_reuse_topk=args.kv_reuse_topk,
        kv_reuse_hot_share=args.kv_reuse_hot_share,
        kv_reuse_action_offset=args.kv_reuse_action_offset,
        kv_reuse_action_stride=args.kv_reuse_action_stride,
        source_medium=args.source_medium,
        seed=args.seed,
        history_recompute_len=args.history_recompute_len,
    )
    output = Path(args.output)
    write_json(output, trace, compact=args.compact_json)
    models_list = {"models": [{"name": trace["metadata"]["model_name"], "trace_path": str(output)}]}
    models_path = Path(args.models_list)
    write_json(models_path, models_list, compact=args.compact_json)


def write_pipeline_traces(args, op_modeling):
    rng = random.Random(args.seed)
    output = Path(args.output)
    if output.suffix == ".json":
        output_dir = output.with_suffix("")
    else:
        output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_batch_size = args.macro_batch_size or args.tokens
    candidates_per_user = args.candidates_per_user or args.tokens
    num_macros = (candidates_per_user + macro_batch_size - 1) // macro_batch_size
    batches = contiguous_batches(args.num_users, args.users_per_batch)

    models = []
    for batch_id, users in enumerate(batches):
        for macro_id in range(num_macros):
            start = macro_id * macro_batch_size
            end = min(start + macro_batch_size, candidates_per_user)
            tokens = end - start
            batch_size = len(users)
            model_name = f"hstu_b{batch_id}_m{macro_id}"
            indices_values_per_user = []
            for user in users:
                history_rng = random.Random(args.seed + user * 1000003)
                history_indices = [
                    history_rng.randrange(args.vocab)
                    for _ in range(args.history_recompute_len)
                ]
                candidate_indices = [rng.randrange(args.vocab) for _ in range(tokens)]
                indices_values_per_user.append(history_indices + candidate_indices)
            trace = build_batched_trace(
                args.layers,
                tokens,
                args.hidden,
                args.kv_len,
                args.vocab,
                user_ids=users,
                batch_id=batch_id,
                macro_batch_id=macro_id,
                indices_values_per_user=indices_values_per_user,
                model_name=model_name,
                op_modeling=op_modeling,
                attention_modeling=args.attention_modeling,
                pipeline_enabled=True,
                kv_reuse_enabled=args.enable_kv_reuse,
                kv_reuse_variant=args.kv_reuse_variant,
                kv_reuse_action_count=args.kv_reuse_action_count,
                kv_reuse_window_size=args.kv_reuse_window_size,
                kv_reuse_topk=args.kv_reuse_topk,
                kv_reuse_hot_share=args.kv_reuse_hot_share,
                kv_reuse_action_offset=args.kv_reuse_action_offset,
                kv_reuse_action_stride=args.kv_reuse_action_stride,
                source_medium=args.source_medium,
                seed=args.seed,
                history_recompute_len=args.history_recompute_len,
            )
            trace_path = output_dir / f"{model_name}.json"
            write_json(trace_path, trace, compact=args.compact_json)
            weight_key = (
                f"hstu_shared_b{batch_size}_t{tokens}_hr{args.history_recompute_len}_h{args.hidden}_kv{args.kv_len}_l{args.layers}"
            )
            model_index = len(models)
            models.append(
                {
                    "name": model_name,
                    "trace_path": str(trace_path),
                    "request_time": model_index * 1e-9,
                    "weight_key": weight_key,
                    "batch_id": batch_id,
                    "macro_batch_id": macro_id,
                    "batch_size": batch_size,
                    "user_ids": users,
                }
            )

    models_path = Path(args.models_list)
    write_json(
        models_path,
        {
            "metadata": {
                "workload_type": "hstu_ranking_pipeline",
                "num_users": args.num_users,
                "users_per_batch": args.users_per_batch,
                "candidates_per_user": candidates_per_user,
                "macro_batch_size": macro_batch_size,
                "num_macrobatches": num_macros,
                "batch_policy": args.batch_policy,
                "random_seed": args.seed,
                "op_modeling": op_modeling,
                "attention_modeling": args.attention_modeling,
                "shared_trace": False,
                "kv_reuse_enabled": args.enable_kv_reuse,
                "kv_reuse_variant": args.kv_reuse_variant,
                "kv_reuse_action_count": args.kv_reuse_action_count,
                "kv_reuse_window_size": args.kv_reuse_window_size,
                "kv_reuse_topk": args.kv_reuse_topk,
                "kv_reuse_hot_share": args.kv_reuse_hot_share,
                "kv_reuse_action_offset": args.kv_reuse_action_offset,
                "kv_reuse_action_stride": args.kv_reuse_action_stride,
                "source_medium": args.source_medium,
                "history_recompute_len": args.history_recompute_len,
                "attention_mask_mode": attention_mask_mode(args.history_recompute_len),
            },
            "models": models,
        },
        compact=args.compact_json,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="example/trace_tests/test_hstu_8layer_baseline.json")
    parser.add_argument("--models-list", default="example/hstu_trace_models_list.json")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--kv-len", "--history-len", dest="kv_len", type=int, default=4)
    parser.add_argument(
        "--history-recompute-len",
        type=int,
        default=0,
        help=(
            "Number of tail history rows recomputed from embedding instead of "
            "loaded from KV cache. 0 preserves baseline."
        ),
    )
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-users", type=int, default=1)
    parser.add_argument("--users-per-batch", type=int, default=1)
    parser.add_argument("--candidates-per-user", type=int)
    parser.add_argument("--macro-batch-size", type=int)
    parser.add_argument("--batch-policy", choices=["contiguous"], default="contiguous")
    parser.add_argument(
        "--source-medium",
        choices=["ddr", "ssd"],
        default="ddr",
        help="Initial storage medium for preload source tensors.",
    )
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--enable-kv-reuse", action="store_true")
    parser.add_argument(
        "--kv-reuse-variant",
        choices=["global", "window_topk"],
        default="window_topk",
        help=(
            "KV reuse mapping variant. Use global to reproduce the original "
            "whole-history action reuse experiments."
        ),
    )
    parser.add_argument(
        "--kv-reuse-action-count",
        type=int,
        default=4,
        help=(
            "Number of distinct synthetic actions for global action reuse. "
            "Kept for compatibility with the original KV reuse experiments."
        ),
    )
    parser.add_argument(
        "--kv-reuse-window-size",
        type=int,
        default=1024,
        help=(
            "Logical window size used to build local reuse layout. "
            "Reuse never crosses window boundaries."
        ),
    )
    parser.add_argument(
        "--kv-reuse-topk",
        type=int,
        default=4,
        help=(
            "Number of hot action groups retained per window."
        ),
    )
    parser.add_argument(
        "--kv-reuse-hot-share",
        type=float,
        default=0.75,
        help=(
            "Fraction of action rows in each window assigned to the hot top-k "
            "action groups before cold rows are left unique."
        ),
    )
    parser.add_argument(
        "--kv-reuse-action-offset",
        type=int,
        default=1,
        help=(
            "0-based index of the first action row in the [item, action, ...] "
            "history sequence."
        ),
    )
    parser.add_argument(
        "--kv-reuse-action-stride",
        type=int,
        default=2,
        help="Stride between action rows in the generated KV history sequence.",
    )
    parser.add_argument(
        "--shared-trace",
        action="store_true",
        help="Reuse one trace template per macro shape and keep unique request entries in the models list.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Write compact JSON for large generated workloads.",
    )
    parser.add_argument(
        "--op-modeling",
        default="",
        help="Comma-separated op modes, e.g. split=skip,view=skip,concat=materialize",
    )
    parser.add_argument(
        "--attention-modeling",
        choices=["decomposed", "fused"],
        default="decomposed",
        help="Use decomposed HSTU attention subgraph or one reuse-aware fused attention op.",
    )
    args = parser.parse_args()

    if args.num_users < 1 or args.users_per_batch < 1:
        raise ValueError("--num-users and --users-per-batch must be positive")
    if args.kv_reuse_action_count < 1:
        raise ValueError("--kv-reuse-action-count must be positive")
    if args.kv_reuse_window_size < 1:
        raise ValueError("--kv-reuse-window-size must be positive")
    if args.kv_reuse_topk < 1:
        raise ValueError("--kv-reuse-topk must be positive")
    if not (0.0 < args.kv_reuse_hot_share <= 1.0):
        raise ValueError("--kv-reuse-hot-share must be in (0, 1]")
    if args.kv_reuse_action_stride < 1:
        raise ValueError("--kv-reuse-action-stride must be positive")
    if args.kv_reuse_action_offset < 0:
        raise ValueError("--kv-reuse-action-offset must be non-negative")
    if args.history_recompute_len < 0:
        raise ValueError("--history-recompute-len must be non-negative")
    if args.history_recompute_len > args.kv_len:
        raise ValueError("--history-recompute-len must be <= --kv-len")
    op_modeling = parse_op_modeling(args.op_modeling)
    multi_trace = (
        args.pipeline
        or args.num_users > 1
        or args.candidates_per_user is not None
        or args.macro_batch_size is not None
    )
    if multi_trace:
        write_pipeline_traces(args, op_modeling)
    else:
        write_single_trace(args, op_modeling)


if __name__ == "__main__":
    main()
