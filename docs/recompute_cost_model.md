# 重计算比例预测

使用 item-KV cost model 预测 `w_both`（AR + recompute）的重计算量时，请使用：

- 硬件配置：`configs/910A.json`、`910B.json` 或 `910C.json`
- 校准配置：`configs/item_kv_calib.json`
- 预测脚本：`scripts/recompute_ratio_cost_model_new.py`

## 直接预测

下面的例子预测 910C、HSTU-small、seq=4096、bs=1、cold 用户的重计算量：

```bash
python3 scripts/recompute_ratio_cost_model_new.py \
  --cost-model paper \
  --config configs/910C.json \
  --calibration configs/item_kv_calib.json \
  --user cold \
  --layers 4 \
  --hidden 256 \
  --kv-len 4096 \
  --batch-size 1 \
  --enable-kv-reuse \
  --kv-reuse-ratio 0.4802
```

模型规模对应参数如下：

| 模型 | `--layers` | `--hidden` |
| --- | ---: | ---: |
| HSTU-small | 4 | 256 |
| HSTU-middle | 8 | 512 |
| HSTU-large | 12 | 1024 |

`--user hot` 使用 DDR，`--user cold` 使用 SSD。输出中的 `history_recompute_len` 是重计算的 item 数量 `k`，`recompute_ratio` 是 `k/S_i`；只取数值时可分别添加 `--field len` 或 `--field ratio`。模型枚举所有整数 `k`，所以预测比例可以是 0.13 一类的中间值，不受 0.1 校准网格限制。

上述命令匹配本次校准数据中的 AR + recompute 设置。若只预测 `w_IR`，去掉 `--enable-kv-reuse` 和 `--kv-reuse-ratio 0.4802`；不要把这两个参数用于 Full-Recompute 或 Full-Cache。

## 硬件参数与校准参数

原始 `B_kv`、`B_emb`、`B_core`、`F_cube` 和 `F_vec` 来自 `--config`，也可用同名命令行参数覆盖。`configs/item_kv_calib.json` 只提供按 chip、存储介质和 batch 分组的硬件有效率、饱和尺度和启动开销，不修改公式中的计算量或访存量，也没有 model、hidden、seq 或 ratio 专用系数。

脚本按以下核心决策计算：

```text
R_eff = R_peak * eta * (1 - exp(-(W / n_req) / x_sat))
T_path = W / R_eff + n_req * tau_startup
T(k) = max(T_mem(k), T_cube(k) + T_vec(k) + T_core(k))
k* = argmin T(k),  k ∈ {0, 1, ..., S_i}
```

命令行硬件参数优先级最高，随后是校准配置中的硬件系数，最后是 NPU 配置。完整 JSON 输出的 `hardware_parameter_sources` 可用于核对每个参数的来源。

## 用新增结果重新校准

ratio sweep 全部完成或硬件发生变化后，在项目容器中运行：

```bash
python3 scripts/calibrate_item_kv_hardware.py \
  results/hstu_calibration_matrix_20260901/w_both_ratio \
  --output configs/item_kv_calib.json \
  --workers 18
```

校准器只使用具有 0、0.1、…、1.0 全部 11 个测量点的曲线。配置中的 `source_root`、`complete_curve_count` 和 `in_sample_validation` 记录实际数据范围与拟合质量。
