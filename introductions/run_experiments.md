# 使用 `run_hstu.sh` 复现 HSTU 推理实验

## 1. `run_hstu.sh` 的执行流程

`scripts/run_hstu.sh` 是单次 HSTU 负载实验的入口。一次运行会完成以下步骤：

1. 调用 `scripts/generate_hstu_baseline_trace.py` 生成 HSTU 推理 trace 和 `models.json`。
2. 基于 `--base-config` 指定的基础配置生成本次实验的 `runtime_config.json`。
3. 在 `runtime_config.json` 中打开 layer-level preload pipeline，并写入 `pipeline.breakdown_csv`。
4. 调用 `./build/bin/Simulator` 执行 trace 模式仿真。
5. 调用 `scripts/plot_pipeline_timeline.py`，根据 `layer_breakdown.csv` 绘制 `layer_timeline.png`。

注意：如果 `--result-dir` 指定的目录已经存在，`run_hstu.sh` 会先删除该目录再重新生成结果。因此不要把需要保留的手工文件放在结果目录下。


## 2. 最小运行示例

关闭 KV Reuse，初始数据放在 DDR：

```bash
bash scripts/run_hstu.sh \
  --source-medium ddr \
  --base-config configs/910c_mini_ddr.json \
  --result-dir results/recompute_kv_off \
  --layers 4 \
  --hidden 256 \
  --kv-len 2048 \
  --history-recompute-len 1024 \
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
