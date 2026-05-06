# 使用 `run_hstu.sh` 复现 HSTU 推理实验

## 1. `run_hstu.sh` 的执行流程

`scripts/run_hstu.sh` 是单次 HSTU 负载实验的入口。一次运行会完成以下步骤：

1. 调用 `scripts/generate_hstu_baseline_trace.py` 生成 HSTU 推理 trace 和 `models.json`。
2. 基于 `--base-config` 指定的基础配置生成本次实验的 `runtime_config.json`。
3. 在 `runtime_config.json` 中打开 layer-level preload pipeline，并写入 `pipeline.breakdown_csv`。
4. 调用 `./build/bin/Simulator` 执行 trace 模式仿真。
5. 调用 `scripts/plot_pipeline_timeline.py`，根据 `layer_breakdown.csv` 绘制 `layer_timeline.png`。

注意：如果 `--result-dir` 指定的目录已经存在，`run_hstu.sh` 会先删除该目录再重新生成结果。因此不要把需要保留的手工文件放在结果目录下。

## 3. 最小运行示例

关闭 KV Reuse，初始数据放在 DDR：

```bash
bash scripts/run_hstu.sh \
  --source-medium ddr \
  --base-config configs/910c_mini_ddr.json \
  --result-dir results/hstu_kv_off \
  --layers 4 \
  --hidden 256 \
  --kv-len 2048 \
  --num-users 4 \
  --users-per-batch 4 \
  --candidates-per-user 512 \
  --macro-batch-size 512 \
  --vocab 262144 \
  --attention-modeling fused \
  --log-level warn
```

开启 KV Reuse，只需要增加 `--enable-kv-reuse`，并显式指定复用策略参数：

```bash
bash scripts/run_hstu.sh \
  --source-medium ddr \
  --base-config configs/910c_mini_ddr.json \
  --result-dir results/hstu_kv_on \
  --layers 4 \
  --hidden 256 \
  --kv-len 2048 \
  --num-users 4 \
  --users-per-batch 4 \
  --candidates-per-user 512 \
  --macro-batch-size 512 \
  --vocab 262144 \
  --attention-modeling fused \
  --enable-kv-reuse \
  --kv-reuse-variant window_topk \
  --kv-reuse-window-size 1024 \
  --kv-reuse-topk 4 \
  --kv-reuse-hot-share 0.75 \
  --log-level warn
```

## 4. 指定 KV Reuse 开关与核心参数

KV Reuse 的开关由 `--enable-kv-reuse` 控制：

| 目标 | 参数写法 |
| --- | --- |
| 关闭 KV Reuse | 不传入 `--enable-kv-reuse` |
| 开启 KV Reuse | 传入 `--enable-kv-reuse` |

KV Reuse 相关核心参数如下：

| 参数 | 含义 | 常用取值 |
| --- | --- | --- |
| `--kv-reuse-variant` | KV Reuse 映射策略。`window_topk` 表示按局部窗口选择 Top-k 热门 action；`global` 表示全局 action 复用策略。 | `window_topk` |
| `--kv-reuse-window-size` | 局部窗口大小。复用映射不会跨越窗口边界。 | `1024` |
| `--kv-reuse-topk` | 每个窗口内参与复用的热门 action 类型数。 | `4` |
| `--kv-reuse-hot-share` | 生成合成 action 序列时，分配给 Top-k 热门 action 的比例。 | `0.75` |


## 5. 指定 HSTU 负载规模

常用负载参数如下：

| 参数 | 含义 |
| --- | --- |
| `--layers` | HSTU layer 数量。 |
| `--hidden` | hidden size。 |
| `--kv-len` 或 `--history-len` | 每个用户历史 KV Cache 长度。 |
| `--num-users` | 本次生成负载中的用户总数。 |
| `--users-per-batch` | 每个 batch 中包含的用户数。 |
| `--candidates-per-user` | 每个用户的候选 item 数量。 |
| `--macro-batch-size` | 每个 macrobatch 处理的候选 item 数量。 |
| `--vocab` | embedding 词表规模。 |
| `--seed` | 生成 trace 的随机种子。 |
| `--attention-modeling` | attention 建模方式，支持 `decomposed` 和 `fused`。建议使用 `fused`。 |
| `--op-modeling` | 部分 PyTorch 算子的物化策略，例如 `split=materialize,view=materialize,concat=materialize`。 |


## 6. 指定初始数据存放介质：DDR 或 SSD

初始数据介质由 `--source-medium` 指定。该参数会影响 weights、KV Cache、embedding rows 等 preload 源张量的初始位置：

| 参数 | 含义 | 默认配置 |
| --- | --- | --- |
| `--source-medium ddr` | 初始数据位于 DDR，仿真 DDR 到 HBM 的搬运。 | `configs/910c_mini_ddr.json` |
| `--source-medium ssd` | 初始数据位于 SSD，仿真 SSD 到 HBM 的搬运。 | `configs/910c_mini_ssd.json` |

如果不指定 `--base-config`，脚本会根据 `--source-medium` 自动选择 `configs/910c_mini_<source-medium>.json`。

```bash
bash scripts/run_hstu.sh \
  --source-medium ddr \
  --base-config configs/baseline.json \
  --result-dir results/ddr_baseline
```

使用 SSD 作为初始介质的示例：

```bash
bash scripts/run_hstu.sh \
  --source-medium ssd \
  --base-config configs/910c_mini_ssd.json \
  --result-dir results/ssd_baseline
```

## 7. 指定结果输出目录

结果目录由 `--result-dir` 指定。如果不指定，默认输出到：

```text
results/run_hstu_<source-medium>
```

一次成功运行后，结果目录通常包含：

| 文件或目录 | 含义 |
| --- | --- |
| `traces/` | 生成的 HSTU trace。每个 trace 文件通常对应一个 batch/macrobatch。 |
| `models.json` | 仿真器 trace 模式读取的模型列表。 |
| `runtime_config.json` | 本次实验实际使用的仿真器配置。 |
| `layer_breakdown.csv` | 仿真器输出的 pipeline 分阶段统计，是后续绘图和性能分析的主要数据源。 |
| `layer_timeline.png` | 由 `plot_pipeline_timeline.py` 根据 `layer_breakdown.csv` 绘制的时间线图。 |
| `layer.log` | 仿真器日志，包含端到端 simulation time、各时钟域 cycle 数和 wall-clock 时间等信息。 |

`run_hstu.sh` 运行结束时会打印这些路径：

```text
Result dir: ...
Config: ...
Models: ...
Breakdown: ...
Timeline: ...
Log: ...
```


