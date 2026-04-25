#!/usr/bin/env python3

import argparse
import json
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


def build_trace(layers, tokens, hidden, kv_len, vocab):
    ops = []

    add_op(
        ops,
        "aten::embedding",
        [
            ddr_to_hbm_tensor(
                "embedding_table",
                [vocab, hidden],
                is_weight=True,
                logical_id="embedding.table",
                role="embedding_table",
            ),
            hbm_tensor("candidate_ids", [tokens], dtype="int64", role="indices"),
        ],
        [hbm_tensor("x0", [tokens, hidden], role="activation")],
        {"indices_values": ",".join(str(i % vocab) for i in range(tokens))},
    )

    current = "x0"
    for layer in range(layers):
        prefix = f"layer{layer}"
        z = f"{prefix}.z"
        zact = f"{prefix}.zact"
        u = f"{prefix}.u"
        v = f"{prefix}.v"
        q = f"{prefix}.q"
        k = f"{prefix}.k"
        k_cache = f"{prefix}.kc"
        v_cache = f"{prefix}.vc"
        k_all = f"{prefix}.k_all"
        k_all_t = f"{prefix}.k_all_t"
        score = f"{prefix}.score"
        attn = f"{prefix}.attn"
        v_all = f"{prefix}.v_all"
        av = f"{prefix}.av"
        av_norm = f"{prefix}.av_norm"
        gated = f"{prefix}.gated"
        out = f"x{layer + 1}"

        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(current, [tokens, hidden], role="activation"),
                ddr_to_hbm_tensor(
                    f"{prefix}.w1",
                    [hidden, hidden * 4],
                    is_weight=True,
                    logical_id=f"{prefix}.w1",
                    role="weight",
                    layer_id=layer,
                ),
                ddr_to_hbm_tensor(
                    f"{prefix}.b1",
                    [hidden * 4],
                    is_weight=True,
                    logical_id=f"{prefix}.b1",
                    role="weight",
                    layer_id=layer,
                ),
            ],
            [hbm_tensor(z, [tokens, hidden * 4], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(z, [tokens, hidden * 4], role="activation", layer_id=layer)],
            [hbm_tensor(zact, [tokens, hidden * 4], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::split",
            [hbm_tensor(zact, [tokens, hidden * 4], role="activation", layer_id=layer)],
            [
                hbm_tensor(u, [tokens, hidden], role="activation", layer_id=layer),
                hbm_tensor(v, [tokens, hidden], role="activation", layer_id=layer),
                hbm_tensor(q, [tokens, hidden], role="activation", layer_id=layer),
                hbm_tensor(k, [tokens, hidden], role="activation", layer_id=layer),
            ],
            {"axis": 1},
        )
        add_op(
            ops,
            "aten::cat",
            [
                ddr_to_hbm_tensor(
                    k_cache,
                    [kv_len, hidden],
                    logical_id=f"user0.{prefix}.kc",
                    role="kv_cache_k",
                    layer_id=layer,
                    user_id=0,
                ),
                hbm_tensor(k, [tokens, hidden], role="activation", layer_id=layer),
            ],
            [hbm_tensor(k_all, [kv_len + tokens, hidden], role="activation", layer_id=layer)],
            {"axis": 0},
        )
        add_op(
            ops,
            "aten::transpose",
            [hbm_tensor(k_all, [kv_len + tokens, hidden], role="activation", layer_id=layer)],
            [hbm_tensor(k_all_t, [hidden, kv_len + tokens], role="activation", layer_id=layer)],
            {"dims": "1,0"},
        )
        add_op(
            ops,
            "aten::matmul",
            [
                hbm_tensor(q, [tokens, hidden], role="activation", layer_id=layer),
                hbm_tensor(k_all_t, [hidden, kv_len + tokens], role="activation", layer_id=layer),
            ],
            [hbm_tensor(score, [tokens, kv_len + tokens], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::silu",
            [hbm_tensor(score, [tokens, kv_len + tokens], role="activation", layer_id=layer)],
            [hbm_tensor(attn, [tokens, kv_len + tokens], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::cat",
            [
                ddr_to_hbm_tensor(
                    v_cache,
                    [kv_len, hidden],
                    logical_id=f"user0.{prefix}.vc",
                    role="kv_cache_v",
                    layer_id=layer,
                    user_id=0,
                ),
                hbm_tensor(v, [tokens, hidden], role="activation", layer_id=layer),
            ],
            [hbm_tensor(v_all, [kv_len + tokens, hidden], role="activation", layer_id=layer)],
            {"axis": 0},
        )
        add_op(
            ops,
            "aten::matmul",
            [
                hbm_tensor(attn, [tokens, kv_len + tokens], role="activation", layer_id=layer),
                hbm_tensor(v_all, [kv_len + tokens, hidden], role="activation", layer_id=layer),
            ],
            [hbm_tensor(av, [tokens, hidden], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::layer_norm",
            [
                hbm_tensor(av, [tokens, hidden], role="activation", layer_id=layer),
                ddr_to_hbm_tensor(
                    f"{prefix}.ln_w",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{prefix}.ln_w",
                    role="weight",
                    layer_id=layer,
                ),
                ddr_to_hbm_tensor(
                    f"{prefix}.ln_b",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{prefix}.ln_b",
                    role="weight",
                    layer_id=layer,
                ),
            ],
            [hbm_tensor(av_norm, [tokens, hidden], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::mul",
            [
                hbm_tensor(av_norm, [tokens, hidden], role="activation", layer_id=layer),
                hbm_tensor(u, [tokens, hidden], role="activation", layer_id=layer),
            ],
            [hbm_tensor(gated, [tokens, hidden], role="activation", layer_id=layer)],
        )
        add_op(
            ops,
            "aten::linear",
            [
                hbm_tensor(gated, [tokens, hidden], role="activation", layer_id=layer),
                ddr_to_hbm_tensor(
                    f"{prefix}.w2",
                    [hidden, hidden],
                    is_weight=True,
                    logical_id=f"{prefix}.w2",
                    role="weight",
                    layer_id=layer,
                ),
                ddr_to_hbm_tensor(
                    f"{prefix}.b2",
                    [hidden],
                    is_weight=True,
                    logical_id=f"{prefix}.b2",
                    role="weight",
                    layer_id=layer,
                ),
            ],
            [hbm_tensor(out, [tokens, hidden], role="activation", layer_id=layer)],
        )
        current = out

    return {
        "metadata": {
            "format_version": "1.0",
            "model_name": "hstu_8layer_baseline_small",
            "model_type": "hstu_ranking",
            "layout": "NHWC",
            "num_layers": layers,
            "baseline_preload": True,
            "fail_on_unknown_op": True,
        },
        "operators": ops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="example/trace_tests/test_hstu_8layer_baseline.json")
    parser.add_argument("--models-list", default="example/hstu_trace_models_list.json")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--kv-len", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=128)
    args = parser.parse_args()

    trace = build_trace(args.layers, args.tokens, args.hidden, args.kv_len, args.vocab)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    models_list = {
        "models": [
            {
                "name": "hstu_8layer_baseline_small",
                "trace_path": str(output),
            }
        ]
    }
    models_path = Path(args.models_list)
    models_path.parent.mkdir(parents=True, exist_ok=True)
    models_path.write_text(json.dumps(models_list, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
