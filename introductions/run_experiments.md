# 使用 `run_hstu.sh` 复现 HSTU 推理实验

`scripts/run_hstu.sh` 是运行 HSTU 推理仿真实验的统一入口。脚本中包含从trace生成到数据可视化的完整流程。

## 1. 执行流程

一次 `run_hstu.sh` 调用会在 `--result-dir` 下依次生成如下文件：

| 文件 / 目录 | 作用 |
|---|---|
| `traces/` | HSTU trace 文件目录，描述每个 request / macro batch 的算子图、tensor 来源、preload 信息和算子属性。 |
| `models.json` | simulator trace mode 的模型列表入口，记录需要加载的 trace 文件及 request metadata。 |
| `runtime_config.json` | 基于 `--base-config` 生成的本次运行配置；其中会写入 `layer_breakdown.csv` 和 `hardware_summary.csv` 的输出路径，并启用 layer-level preload pipeline。 |
| `layer.log` | simulator 标准输出和错误输出日志。 |
| `layer_breakdown.csv` | 本次仿真的核心明细结果，记录每个 layer 的 preload、compute、op、movin 等事件的起止时间和耗时。 |
| `hardware_summary.csv` | simulator 汇总出的 NPU、HBM、DDR、SSD 等硬件统计信息，包括利用率、带宽等指标。 |
| `layer_timeline.png` | 根据 `layer_breakdown.csv` 生成的 pipeline timeline 可视化图片。 |


## 2. 完整参数说明

| 参数 | 可选值 / 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `--source-medium` | `ddr` / `ssd` | `ssd` | 历史 KV cache 等主要源数据的初始介质。cold 用户通常用 `ssd`，hot 用户通常用 `ddr`。 |
| `--embedding-source-medium` | `ddr` / `ssd` | `ssd` | candidate embedding rows 的初始介质。 |
| `--history-recompute-source-medium` | `ddr` / `ssd` | 同 `--source-medium` | IR/recompute 的历史 embedding rows 的初始介质。 |
| `--base-config` | path | `configs/910C.json` | simulator 基础配置文件。 |
| `--result-dir` | path | `results/run_hstu_<source-medium>` | 实验输出目录。若目录已存在会被删除重建。 |
| `--layers` | int | `4` | HSTU 层数。 |
| `--hidden` | int | `256` | hidden dimension。 |
| `--kv-len` / `--history-len` | int | `4096` | 历史 KV 序列长度。 |
| `--history-recompute-len` | int | `0` | 从 embedding 重新计算的历史token行数。`0` 表示不启用 IR；必须不超过 `--kv-len`。 |
| `--history-recompute-index-mode` | `continuous` / `random` | `continuous` | recompute 历史 embedding 的索引模式。 |
| `--num-users` | int | `1` | 生成 workload 中的用户数。 |
| `--users-per-batch` | int | `1` | 每个 batch 的用户数；必须能被 `--npu-count` 整除。 |
| `--candidates-per-user` | int | `128` | 每个用户的候选 item 数。 |
| `--macro-batch-size` | int | `128` | candidate macro batch size。 |
| `--npu-count` | int | `1` | trace mode batch sharding 使用的同构 NPU 数量。 |
| `--vocab` | int | `262144` | embedding vocabulary size。 |
| `--seed` | int | `1234` | trace 随机种子。 |
| `--op-modeling` | string | `split=materialize,view=materialize,concat=materialize` | 指定部分算子的建模方式，例如 `split=materialize,view=materialize,concat=materialize`。 |
| `--attention-modeling` | `decomposed` / `fused` | `fused` | attention 建模方式。当前 HSTU 实验通常使用 `fused`。 |
| `--without-ooo-pipeline` | flag | 不启用 | 标准 without out-of-order pipeline 模式：关闭 attention partial start，关闭 AR attention compute reduction，并增加 HBM->HBM history restore。 |
| `--enable-kv-reuse` | flag | 不启用 | 开启 KV row reuse metadata，即 AR。 |
| `--kv-reuse-ratio` | float in `[0,1)` | `0` | KV reuse 压缩比例。开启 `--enable-kv-reuse` 后，该值决定 action KV 的压缩强度。 |
| `--log-level` | string | `warn` | simulator log level。 |
| `-h`, `--help` | flag | - | 打印帮助信息。 |

## 3. 运行示例

### 3.1 最小运行示例

该示例不启用 Action KV Reuse 和 Item Recompute等优化。默认使用 HSTU-small 形状：`layers=4`、`hidden=256`、`kv_len=4096`、`batch_size=1`、`candidates=128`；使用910C配置，通过指定--source-medium ssd 对cold用户进行请求推理。


```bash
bash scripts/run_hstu.sh \
  --source-medium ssd \
  --embedding-source-medium ssd \
  --history-recompute-source-medium ssd \
  --base-config configs/910C.json \
  --result-dir results/examples/hstu_minimal_no_opt \
  --layers 4 \
  --hidden 256 \
  --kv-len 4096 \
  --history-recompute-len 0 \
  --num-users 4 \
  --users-per-batch 4 \
  --candidates-per-user 128 \
  --macro-batch-size 128 \
  --vocab 262144 \
  --attention-modeling fused \
  --log-level warn
```
修改--history-recompute-len 即可设置Recompute的token数量 。在Item Recompute 方法中，可以先使用脚本scripts/recompute_ratio_cost_model_new.py，根据代价模型估算最优重算比例，再将结果作为 --history-recompute-len 参数传入。

### 3.2 Action KV Reuse运行示例

需要通过 `--enable-kv-reuse` 开关开启Action KV Reuse优化，并传入对应的 reuse 参数。下面示例运行 HSTU-small、`kv_len=4096`、`batch_size=4`、cold 用户：

```bash
bash scripts/run_hstu.sh \
  --source-medium ssd \
  --embedding-source-medium ssd \
  --history-recompute-source-medium ssd \
  --base-config configs/910C.json \
  --result-dir results/examples/w_ar_hstu_small_seq4096_bs4_cold \
  --layers 4 \
  --hidden 256 \
  --kv-len 4096 \
  --history-recompute-len 0 \
  --num-users 4 \
  --users-per-batch 4 \
  --candidates-per-user 128 \
  --macro-batch-size 128 \
  --vocab 262144 \
  --attention-modeling fused \
  --enable-kv-reuse \
  --kv-reuse-ratio 0.4360 \
  --log-level warn
```