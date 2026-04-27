# KV Cache Action Reuse 说明
当前版本仅实现了同一用户历史序列全局的kvreuse，尚未实现window版本。
## 1. 目标

KV cache action reuse 用于建模 GR 历史序列中重复 action 的存储优化。逻辑上，后续 HSTU 计算仍看到完整的 KV cache：

```text
KV logical shape = [history_len, hidden]
```

存储上，重复 action 只保留一份 physical row：

```text
logical row -> physical row
0 -> 0
1 -> 1
2 -> 0
3 -> 1
...
```

这样从 DDR/SSD 加载到 HBM 时只搬运唯一 action 行，执行阶段再按 logical row 映射在 SRAM 中补齐重复位置。

## 2. 生成方式

示例：

```bash
python3 scripts/generate_hstu_baseline_trace.py \
  --pipeline \
  --shared-trace \
  --compact-json \
  --layers 8 \
  --hidden 512 \
  --history-len 1024 \
  --vocab 65536 \
  --num-users 1 \
  --users-per-batch 1 \
  --candidates-per-user 256 \
  --macro-batch-size 64 \
  --tokens 64 \
  --op-modeling split=skip,view=skip,concat=skip \
  --enable-kv-reuse \
  --kv-reuse-action-count 4 \
  --output example/trace_tests/kv_reuse_action_eval_on \
  --models-list example/kv_reuse_action_eval_on_models_list.json
```
可以通过`--enable-kv-reuse`开关来控制是否使用kv reuse。

`--kv-reuse-action-count 4` 表示 trace generator 会生成一条长度为 `history_len` 的模拟 action 序列，且 action id 取自 4 个 distinct synthetic actions。随后 generator 按 action 首次出现顺序构建 `reuse_logical_to_physical`。例如：

```text
action ids:              [0, 1, 2, 3, 1, 0, 0, 2]
logical -> physical row: [0, 1, 2, 3, 1, 0, 0, 2]
```
## 3.效果验证

当前方法基于baseline改造而来，即仍保持HBM+大容量DDR的存储配置；配置文件：

```text
configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ddr_default.json
```

同一 workload，开启：

```text
--enable-kv-reuse --kv-reuse-action-count 4
```

| 版本 | KV reuse | movement 序列 | first request KV logical bytes | first request KV physical bytes | saved | finished requests | simulation time (us) | wall-clock (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不使用kv reuse | off | 128, 64, 64, 64 | - | - | - | 4 | 1934.592 | 98.329 |
| 使用kv reuse | on | 176, 64, 64, 64 | 16777216 | 65536 | 99.61% | 4 | 1535.314 | 86.507 |

说明：

- movement 数从 128 增加到 176，是因为当前实现按 unique row 生成 row-level movement；单个 movement 变小。
- 总 KV preload bytes 明显下降，因此首个 request data ready 时间从约 901 us 降到约 503 us。
- 后续 macrobatch 仍复用 HBM resident 的 weight/KV，因此 movement 序列保持 64。
