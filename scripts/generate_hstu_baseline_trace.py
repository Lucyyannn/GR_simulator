#!/usr/bin/env python3

import argparse
import json
import random
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


def build_reuse_mapping(length, period):
    if period is None or period <= 0 or period >= length:
        return None
    physical_rows = min(period, length)
    return {
        "reuse_mode": "row_reuse",
        "reuse_axis": 0,
        "reuse_physical_rows": physical_rows,
        "reuse_logical_to_physical": [i % physical_rows for i in range(length)],
    }


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
    pipeline_enabled=False,
    kv_reuse_enabled=False,
    kv_reuse_period=None,
    seed=0,
):
    op_modeling = op_modeling or {}
    ops = []
    indices_values = indices_values or [i % vocab for i in range(tokens)]
    kv_reuse_meta = (
        build_reuse_mapping(kv_len, kv_reuse_period) if kv_reuse_enabled else None
    )

    base = common_meta(user_id, batch_id, macro_batch_id)
    add_op(
        ops,
        "aten::embedding",
        [
            ddr_tensor(
                "embedding_table",
                [vocab, hidden],
                is_weight=True,
                logical_id="embedding.table",
                role="embedding_table",
                **base,
            ),
            hbm_tensor(
                f"u{user_id}.b{batch_id}.m{macro_batch_id}.candidate_ids",
                [tokens],
                dtype="int64",
                role="indices",
                **base,
            ),
        ],
        [
            ddr_to_hbm_tensor(
                f"u{user_id}.b{batch_id}.m{macro_batch_id}.x0",
                [tokens, hidden],
                logical_id=f"u{user_id}.b{batch_id}.m{macro_batch_id}.embedding_rows",
                role="embedding_rows",
                source_logical_id="embedding.table",
                source_shape=[vocab, hidden],
                indices_values=indices_values,
                **base,
            )
        ],
        {
            "indices_values": ",".join(str(v) for v in indices_values),
            "modeling_mode": "preloaded_rows",
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
                hbm_tensor(current, [tokens, hidden], role="activation", **layer_meta),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.w1",
                    [hidden, hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w1",
                    role="weight",
                    **layer_meta,
                ),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.b1",
                    [hidden * 4],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b1",
                    role="weight",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(z, [tokens, hidden * 4], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(z, [tokens, hidden * 4], role="activation", **layer_meta)],
            [hbm_tensor(zact, [tokens, hidden * 4], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::split",
            [hbm_tensor(zact, [tokens, hidden * 4], role="activation", **layer_meta)],
            [
                hbm_tensor(u, [tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(v, [tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(q, [tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(k, [tokens, hidden], role="activation", **layer_meta),
            ],
            {"axis": 1, **op_modeling_attrs(op_modeling, "split")},
        )
        add_op(
            ops,
            "aten::cat",
            [
                ddr_to_hbm_tensor(
                    k_cache,
                    [kv_len, hidden],
                    logical_id=f"user{user_id}.{shared_layer}.kc",
                    role="kv_cache_k",
                    **(kv_reuse_meta or {}),
                    **layer_meta,
                ),
                hbm_tensor(k, [tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(k_all, [kv_len + tokens, hidden], role="activation", **layer_meta)],
            {"axis": 0, **op_modeling_attrs(op_modeling, "concat")},
        )
        add_op(
            ops,
            "aten::transpose",
            [hbm_tensor(k_all, [kv_len + tokens, hidden], role="activation", **layer_meta)],
            [hbm_tensor(k_all_t, [hidden, kv_len + tokens], role="activation", **layer_meta)],
            {"dims": "1,0", **op_modeling_attrs(op_modeling, "view")},
        )
        add_op(
            ops,
            "aten::matmul",
            [
                hbm_tensor(q, [tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(k_all_t, [hidden, kv_len + tokens], role="activation", **layer_meta),
            ],
            [hbm_tensor(score, [tokens, kv_len + tokens], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(score, [tokens, kv_len + tokens], role="activation", **layer_meta)],
            [hbm_tensor(attn, [tokens, kv_len + tokens], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::cat",
            [
                ddr_to_hbm_tensor(
                    v_cache,
                    [kv_len, hidden],
                    logical_id=f"user{user_id}.{shared_layer}.vc",
                    role="kv_cache_v",
                    **(kv_reuse_meta or {}),
                    **layer_meta,
                ),
                hbm_tensor(v, [tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(v_all, [kv_len + tokens, hidden], role="activation", **layer_meta)],
            {"axis": 0, **op_modeling_attrs(op_modeling, "concat")},
        )
        add_op(
            ops,
            "aten::matmul",
            [
                hbm_tensor(attn, [tokens, kv_len + tokens], role="activation", **layer_meta),
                hbm_tensor(v_all, [kv_len + tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(av, [tokens, hidden], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::layer_norm",
            [
                hbm_tensor(av, [tokens, hidden], role="activation", **layer_meta),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.ln_w",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_w",
                    role="weight",
                    **layer_meta,
                ),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.ln_b",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.ln_b",
                    role="weight",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(av_norm, [tokens, hidden], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::mul",
            [
                hbm_tensor(av_norm, [tokens, hidden], role="activation", **layer_meta),
                hbm_tensor(u, [tokens, hidden], role="activation", **layer_meta),
            ],
            [hbm_tensor(gated, [tokens, hidden], role="activation", **layer_meta)],
        )
        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(gated, [tokens, hidden], role="activation", **layer_meta),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.w2",
                    [hidden, hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.w2",
                    role="weight",
                    **layer_meta,
                ),
                ddr_to_hbm_tensor(
                    f"{shared_layer}.b2",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{shared_layer}.b2",
                    role="weight",
                    **layer_meta,
                ),
            ],
            [hbm_tensor(out, [tokens, hidden], role="activation", **layer_meta)],
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
    indices = [rng.randrange(args.vocab) for _ in range(args.tokens)]
    trace = build_trace(
        args.layers,
        args.tokens,
        args.hidden,
        args.kv_len,
        args.vocab,
        indices_values=indices,
        model_name="hstu_8layer_baseline_small",
        op_modeling=op_modeling,
        pipeline_enabled=args.pipeline,
        kv_reuse_enabled=args.enable_kv_reuse,
        kv_reuse_period=args.kv_reuse_period,
        seed=args.seed,
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
    shared_trace_paths = {}
    shared_weight_keys = {}
    for batch_id, users in enumerate(batches):
        for macro_id in range(num_macros):
            start = macro_id * macro_batch_size
            end = min(start + macro_batch_size, candidates_per_user)
            tokens = end - start
            shared_trace_path = None
            shared_weight_key = None
            if args.shared_trace:
                shared_weight_key = f"hstu_shared_t{tokens}_h{args.hidden}_kv{args.kv_len}_l{args.layers}"
                if tokens not in shared_trace_paths:
                    indices = [rng.randrange(args.vocab) for _ in range(tokens)]
                    trace = build_trace(
                        args.layers,
                        tokens,
                        args.hidden,
                        args.kv_len,
                        args.vocab,
                        user_id=0,
                        batch_id=0,
                        macro_batch_id=0,
                        indices_values=indices,
                        model_name=shared_weight_key,
                        op_modeling=op_modeling,
                        pipeline_enabled=True,
                        kv_reuse_enabled=args.enable_kv_reuse,
                        kv_reuse_period=args.kv_reuse_period,
                        seed=args.seed,
                    )
                    trace_path = output_dir / f"{shared_weight_key}.json"
                    write_json(trace_path, trace, compact=args.compact_json)
                    shared_trace_paths[tokens] = trace_path
                    shared_weight_keys[tokens] = shared_weight_key
                shared_trace_path = shared_trace_paths[tokens]
                shared_weight_key = shared_weight_keys[tokens]
            for user_id in users:
                model_name = f"hstu_u{user_id}_b{batch_id}_m{macro_id}"
                if args.shared_trace:
                    trace_path = shared_trace_path
                    weight_key = shared_weight_key
                else:
                    indices = [rng.randrange(args.vocab) for _ in range(tokens)]
                    trace = build_trace(
                        args.layers,
                        tokens,
                        args.hidden,
                        args.kv_len,
                        args.vocab,
                        user_id=user_id,
                        batch_id=batch_id,
                        macro_batch_id=macro_id,
                        indices_values=indices,
                        model_name=model_name,
                        op_modeling=op_modeling,
                        pipeline_enabled=True,
                        kv_reuse_enabled=args.enable_kv_reuse,
                        kv_reuse_period=args.kv_reuse_period,
                        seed=args.seed,
                    )
                    trace_path = output_dir / f"{model_name}.json"
                    write_json(trace_path, trace, compact=args.compact_json)
                    weight_key = model_name
                model_index = len(models)
                models.append(
                    {
                        "name": model_name,
                        "trace_path": str(trace_path),
                        "request_time": model_index * 1e-9,
                        "weight_key": weight_key,
                        "user_id": user_id,
                        "batch_id": batch_id,
                        "macro_batch_id": macro_id,
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
                "shared_trace": args.shared_trace,
                "kv_reuse_enabled": args.enable_kv_reuse,
                "kv_reuse_period": args.kv_reuse_period,
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
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-users", type=int, default=1)
    parser.add_argument("--users-per-batch", type=int, default=1)
    parser.add_argument("--candidates-per-user", type=int)
    parser.add_argument("--macro-batch-size", type=int)
    parser.add_argument("--batch-policy", choices=["contiguous"], default="contiguous")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--enable-kv-reuse", action="store_true")
    parser.add_argument(
        "--kv-reuse-period",
        type=int,
        default=4,
        help="Synthetic action reuse period for KV cache rows when --enable-kv-reuse is set.",
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
    args = parser.parse_args()

    if args.num_users < 1 or args.users_per_batch < 1:
        raise ValueError("--num-users and --users-per-batch must be positive")
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
