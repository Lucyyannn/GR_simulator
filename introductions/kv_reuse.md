# KV Cache Action Reuse 说明
当前版本仅实现了同一用户历史序列全局的kvreuse，尚未实现window版本。
## 1. 目标

KV cache action reuse 用于建模 GR 历史序列中重复 action 的存储优化。逻辑上，后续 HSTU 计算仍看到完整的 KV cache：

```text
KV logical shape = [history_len, hidden]
```

存储上，history 按 `[item, action, item, action, ...]` 的 logical row 顺序描述；重复 action 只保留一份 physical row，item row 始终保持唯一：

```text
logical row 0 item0   -> physical row 0
logical row 1 action0 -> physical row 1
logical row 2 item1   -> physical row 2
logical row 3 action0 -> physical row 1
...
```

这样从 DDR/SSD 加载到 HBM 时，item 行完整搬运，action 行只搬运唯一 action，执行阶段再按 logical row 映射在 SRAM 中补齐重复位置。

## 2.仿真原理
  1. trace generator 继续生成 reuse_logical_to_physical，表示 logical history row 到 compact physical row 的映射。
  2. DDR/SSD 中保留完整历史序列的 logical row 布局，即 [item, action, item, action, ...] 原始顺序。
  3. HBM 中保存 compact layout，只为唯一 item row 和唯一 action row 分配 physical row。
  4. TraceModel 对每个 KV tensor 只提交一个 migration request，但 request 内部包含多段 {src_addr, dst_addr, bytes}。
  5. 每个 segment 表示“从 DDR/SSD 原始 logical row 读取，写到 HBM compact physical row”。
  6. StorageController 内部遍历 segments，再按 hbm.req_size 拆成真实 memory access。
  7. tensor 的 residency 只在整个 segmented migration 完成后标记为 resident。
  8. 后续同一用户同一 KV tensor 命中 resident 后，不再重复提交 migration。
  9. concat/materialize/compute 阶段继续通过 tensor 的 reuse layout 把 logical row 访问映射到 HBM physical row。

  这样，上层调度看到的是“加载一个 KV tensor”，不是“加载 516 个 row”；底层存储仍能看到真实地址和真实访存量。

## 3. 生成方式

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

`--kv-reuse-action-count 4` 表示 trace generator 只在 history 中的 action row 上生成模拟 action id，且 action id 取自 4 个 distinct synthetic actions。默认 history 为 `[item, action, item, action, ...]`，即 0-based row `1,3,5,...` 是 action row。随后 generator 按 item/action 首次出现顺序构建 `reuse_logical_to_physical`。例如：

```text
history row:             [item0, action0, item1, action1, item2, action0, item3, action1]
logical -> physical row: [0,     1,       2,     3,       4,     1,       5,     3]
```
## 4.效果验证

当前方法基于baseline改造而来，即仍保持HBM+大容量DDR的存储配置；配置文件：

```text
configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ddr_default.json
```

以下结果基于当前“仅 action row 复用 + tensor-level segmented migration”的实现重新测试得到。

同一 workload，开启：

```text
--enable-kv-reuse --kv-reuse-action-count 4
```

| 版本 | KV reuse | movement 序列 | first request KV logical bytes | first request KV physical bytes | saved | finished requests | simulation time (us) | wall-clock (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不使用kv reuse | off | 128, 64, 64, 64 | - | - | - | 4 | 1934.592 | 96.603 |
| 使用kv reuse | on | 128, 64, 64, 64 | 16777216 | 8454144 | 49.61% | 4 | 1713.249 | 91.342 |

说明：

- 开启 reuse 后，movement 数不再增加。TraceModel 仍按 tensor 级提交 KV migration，row-level gather/scatter 地址通过 segmented migration 下沉到 StorageController 内部处理。
- 只复用 action row，因此 first request KV physical bytes 从完整的 16 MiB 降到约 8.06 MiB，而不是旧实现中“所有 history row 复用”得到的 64 KiB。
- 后续 macrobatch 仍复用 HBM resident 的 weight/KV，因此 movement 序列保持 64。
