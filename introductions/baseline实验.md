# Baseline 实验说明

本文说明当前 HSTU ranking baseline 实验在仿真器中的执行流程、复现方式和实验结果。

## 1. Pipeline 执行流程

### 1.1 仿真启动

baseline 实验使用 `--mode trace`。`src/main.cc` 读取硬件配置、models list，并为每个 request 创建一个 `TraceModel`。models list 中每条 request 至少包含：

| 字段 | 作用 |
| --- | --- |
| `name` | request 名称，例如 `hstu_u0_b0_m3`。 |
| `trace_path` | Aten 风格 JSON trace 路径。 |
| `request_time` | request 到达时间。 |
| `weight_key` | 共享权重表 key；shared trace 下多个 request 复用同一份 weight table。 |
| `user_id` / `batch_id` / `macro_batch_id` | 用于 pipeline 调度、KV residency key 和结果解释。 |

仿真器初始化时会创建 HBM、DDR、可选 SSD、`StorageController`、`ResidencyManager`、NPU cores 和 scheduler。当前实验使用的配置文件是：

```bash
./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ddr_default.json
```

该配置包含 4 个 NPU core，核心计算和 HBM/DDR 访存统一由事件推进。

### 1.2 Request 进入 preload 阶段

到达的 request 先进入 `_waiting_to_preload_models`。`Simulator::admit_preload_models()` 根据 `max_preloading_models` 控制 pipeline admission。当前 baseline 设置 `k=1`，即同一时刻最多一个 request 在 storage preload 阶段；已经 data ready 的 request 可以进入 compute，同时下一个 request 可以开始 preload。

request 被 admit 后，`TraceModel::initialize_model()` 解析 trace，并根据 tensor 的 `role`、`initial_medium`、`runtime_medium` 规划数据迁移：

| role | baseline 行为 |
| --- | --- |
| `embedding_table` | 保留在 DDR，不整表搬到 HBM。 |
| `embedding_rows` | 按当前 macrobatch 的 `indices_values` 从 DDR embedding table gather 对应行，写入 HBM 中的 embedding 输出 `x0`。 |
| `weight` | 首次使用时 DDR->HBM；完成后按 `logical_id` 在 HBM residency 中复用。 |
| `kv_cache_k` / `kv_cache_v` | 首次使用时 DDR->HBM；完成后按 `user_id + layer_id + kc/vc` 在 HBM residency 中复用。 |
| `activation` / `indices` | 作为 HBM runtime tensor，不进入 residency 管理。 |

embedding lookup 当前由 storage preload 建模。trace 中 embedding op 的输出 `x0` 带 `role=embedding_rows`，`TraceModel` 会把它展开成：

```text
DDR embedding_table[index_i] -> HBM x0[i]
```

每条 movement 的大小为 `hidden_dim * precision`。对应的 `aten::embedding` op 使用 `modeling_mode=preloaded_rows`，只保留 DAG 依赖，不重复读取 embedding table。

### 1.3 StorageController 执行数据搬运

`TraceModel::submit_data_movements()` 将规划出的 movements 提交给 `StorageController`。每个 movement 是一个 `MigrationRequest`，包含源介质、目标介质、源地址、目标地址和字节数。

`StorageController` 负责把 DDR->HBM 迁移拆成实际底层访存请求并推进 Ramulator2 DDR/HBM 时序。所有 movements 完成后：

1. `TraceModel::data_movements_ready()` 返回 true。
2. `Simulator` 调用 `TraceModel::complete_data_movements()`。
3. 本 request 加载成功的 weight/KV 被标记为 HBM resident。
4. request 从 `_preloading_models` 移入 `_ready_to_compute_models`。

### 1.4 Storage-compute overlap 时序

当前 pipeline 的时序是：

```text
t0: request_0 preload
t1: request_0 data ready -> request_0 compute
t1: request_1 preload begins
t2: request_1 data ready -> request_1 compute after scheduler admits it
t2: request_2 preload begins
...
```

因此 overlap 发生在“前一个或多个已 ready request 的 NPU compute”和“后续 request 的 DDR->HBM preload”之间。当前没有实现多 request compute 并发 admission policy；compute 端仍由 scheduler 和 core 可用性决定 tile 分发。

### 1.5 NPU core 调度

request 进入 compute 后，`Scheduler` 从 DAG 中选择 executable op，并把 op tiles 分发给 NPU cores。当前配置有 4 个 core。算子内部会产生 tile，每个 tile 包含：

- HBM->SRAM 的 MOVIN；
- core 上的 GEMM/elementwise/layernorm 等计算；
- SRAM->HBM 的 MOVOUT。

对 baseline trace 来说，完整 8-layer HSTU ranking 单层主要包含：

```text
linear -> silu -> split -> concat(K cache) -> transpose/view
-> matmul attention score -> silu attention
-> concat(V cache) -> matmul AV -> layer_norm -> mul gate -> linear
```

`split/view/concat` 可以通过 `--op-modeling split=skip,view=skip,concat=skip` 切换为 skip tile，以便在大规模实验中弱化非重点 layout op 的仿真开销；也可以改成 `materialize` 建模真实读写延迟。

## 2. 复现与生成 Trace

### 2.1 生成一组 baseline pipeline trace

示例：1 个用户，8 层，hidden=512，history=4096，candidate set=512，macro batch size=64。

```bash
python3 scripts/generate_hstu_baseline_trace.py \
  --pipeline \
  --shared-trace \
  --compact-json \
  --layers 8 \
  --hidden 512 \
  --history-len 4096 \
  --vocab 65536 \
  --num-users 1 \
  --users-per-batch 1 \
  --candidates-per-user 512 \
  --macro-batch-size 64 \
  --tokens 64 \
  --op-modeling split=skip,view=skip,concat=skip \
  --output example/trace_tests/baseline_intro_example \
  --models-list example/baseline_intro_example_models_list.json
```

该命令会产生：

| 文件/目录 | 内容 |
| --- | --- |
| `example/trace_tests/baseline_intro_example/` | trace JSON。`--shared-trace` 时通常只生成每种 macro shape 一份共享 trace。 |
| `example/baseline_intro_example_models_list.json` | request 列表，包含每个 user/macrobatch 的 `name`、`trace_path`、`weight_key`、`user_id`、`batch_id`、`macro_batch_id`。 |

关键参数含义：

| 参数 | 含义 |
| --- | --- |
| `--num-users` | 用户数。 |
| `--users-per-batch` | 用户 batch 划分大小。 |
| `--candidates-per-user` | 每个用户候选集大小。 |
| `--macro-batch-size` | m-Falcon 风格 macrobatch 大小；总 request 数为 `num_users * ceil(candidates_per_user / macro_batch_size)`。 |
| `--history-len` | 用户历史 KV cache 长度。 |
| `--hidden` | hidden dimension。 |
| `--vocab` | embedding table 行数；indices 在 `[0, vocab)` 范围内随机生成。 |
| `--shared-trace` | 对相同 shape 的 request 复用 trace 模板，减小 trace 文件体积。 |
| `--op-modeling` | 控制 split/view/concat 等 layout op 是 skip 还是 materialize。 |

### 2.3 运行仿真

```bash
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ddr_default.json \
  --models_list ./example/baseline_intro_example_models_list.json \
  --mode trace \
  --log_level info
```

常用日志字段：

| 日志 | 含义 |
| --- | --- |
| `submitting N data movements` | 当前 request 需要提交给 StorageController 的 DDR->HBM movement 数。 |
| `Model ... data ready at X us` | 当前 request preload 完成时间。 |
| `Schedule model ... at X us` | 当前 request 被送入 NPU scheduler 的时间。 |
| `Model[...] finish:Y us` | 当前 request 计算完成时间。 |
| `simulation time : X us` | 仿真得到的系统时间。 |
| `wall-clock=X s` | 仿真程序真实运行耗时。 |

## 3. 本轮 Baseline 实验结果

本轮实验固定：

- `users=1`
- `layers=8`
- `hidden=512`
- `macro_batch_size=64`
- `vocab=65536`
- `op_modeling=split=skip,view=skip,concat=skip`

扫描参数：

- `candidates_per_user = 256, 512, 1024, 2048`
- `history_len = 1024, 2048, 4096, 8192`

截至本次暂停，已完成 9/16 组实验；后续 7 组未继续运行。完整 CSV 已落盘到：

```text
results/baseline_intro_sweep.csv
```

已完成结果如下：

| history_len | candidates_per_user | macro_batch_size | request 数 | 首个 request movement | 稳态 request movement | simulation time (us) | wall-time (s) | finished requests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 256 | 64 | 4 | 128 | 64 | 1934.592 | 98.329 | 4 |
| 1024 | 512 | 64 | 8 | 128 | 64 | 2961.190 | 169.612 | 8 |
| 1024 | 1024 | 64 | 16 | 128 | 64 | 5019.874 | 307.141 | 16 |
| 1024 | 2048 | 64 | 32 | 128 | 64 | 9138.331 | 596.796 | 32 |
| 2048 | 256 | 64 | 4 | 128 | 64 | 2564.481 | 136.483 | 4 |
| 2048 | 512 | 64 | 8 | 128 | 64 | 3821.562 | 233.249 | 8 |
| 2048 | 1024 | 64 | 16 | 128 | 64 | 6339.779 | 424.879 | 16 |
| 2048 | 2048 | 64 | 32 | 128 | 64 | 11380.372 | 826.115 | 32 |
| 4096 | 256 | 64 | 4 | 128 | 64 | 4006.518 | 250.751 | 4 |

movement 解释：

- 首个 request 的 128 次 movement = 64 行 embedding gather + 48 个 weight + 16 个 KV cache。
- 同一用户后续 macrobatch 的 64 次 movement = 64 行 embedding gather；weight 和 KV cache 已通过 HBM residency 复用。

未完成/暂停的实验组合：

| history_len | candidates_per_user |
| ---: | ---: |
| 4096 | 512 |
| 4096 | 1024 |
| 4096 | 2048 |
| 8192 | 256 |
| 8192 | 512 |
| 8192 | 1024 |
| 8192 | 2048 |

暂停时，`history_len=4096,candidates_per_user=512` 的仿真正在运行中，未写入完整 CSV，因此不计入有效结果。

TODO：改成32个core继续实验
