# NPU 重配置搜索

`scripts/search_npu_reconfiguration.py` 在每个基线 NPU 自己的计算单元面积、功耗预算内，搜索 `w_both` 的 Cube/Vector 配置。搜索不会修改基线 JSON；最优配置会自动生成到结果目录。

## 搜索与目标函数

候选点为 `(N_c, N_v, W_v)`。默认范围是 `N_c=1..64`、`N_v=1..128`、`W_v=1024..8192 bit`，三者步长分别为 `1`、`1`、`1024 bit`。资源约束直接对应论文公式：

```text
A = N_c * 2.57 + N_v * 0.70 * (W_v / 2048)
P = N_c * 3.13 + N_v * 0.46 * (W_v / 2048)
A <= A_base,  P <= P_base
```

对每个可行候选，脚本使用 item-KV cost model 的显式公式计算 `T_mem(k)` 和 `T_npu(k)`，并在整数 `k` 上选择使 `max(T_mem(k), T_npu(k))` 最小的重计算量。最终目标是完整 54 请求矩阵中“每请求 E2E 延迟”的几何平均。`k` 是逐请求重新选择的，不限制在 0.1 比例网格上。

## 运行

在项目容器内执行：

```bash
python3 scripts/search_npu_reconfiguration.py \
  --calibration configs/item_kv_calib.json \
  --output-root results/npu_reconfiguration
```

如需改变范围，使用 `--nc-min/--nc-max/--nc-step`、`--nv-min/--nv-max/--nv-step` 和 `--wv-min/--wv-max/--wv-step`。无需为每次搜索手工编写配置。

主要输出：

- `search_summary.json/csv`：预算、最优设计、预测延迟和预测加速比；
- `<chip>/top_candidates.csv`：排序后的候选点；
- `<chip>/optimal_recompute_choices.json`：54 个请求各自的整数 `k`；
- `configs/<chip>_optimal.json`：自动生成、可直接仿真的配置。

## 仿真最优配置

批量脚本支持重复传入 `--chip-config CHIP=PATH`，因此三种基线的最优配置可以在一次运行中仿真：

```bash
python3 scripts/run_meta_hstu_full_matrix.py \
  --result-root results/npu_reconfiguration/simulation \
  --calibration configs/item_kv_calib.json \
  --methods w_both \
  --max-concurrent 192 \
  --max-simulator-rss-gib 460 \
  --chip-config 910A=results/npu_reconfiguration/configs/910A_optimal.json \
  --chip-config 910B=results/npu_reconfiguration/configs/910B_optimal.json \
  --chip-config 910C=results/npu_reconfiguration/configs/910C_optimal.json
```

生成配置用 `ceil(N_c/4)` 个仿真核压缩物理 Cube 核数，通过调整 Cube 高度和 Vector width 对齐候选点的总算力。目标与实际总算力及相对误差记录在配置的 `metadata.npu_reconfiguration` 中。

## 面积–延迟 Pareto 实验

`scripts/run_npu_area_latency_pareto.py` 提供 `prepare`、`run`、`plot` 三个可独立恢复的阶段。默认实验固定为 HSTU-small、`seq=4096`、`bs={4,8}` 和 `user={hot,cold}`。每颗芯片的随机配置会复用于四种 workload，便于公平比较。

使用 `--model middle` 可运行 HSTU-middle。若使用相同的 `--seed`、点数和搜索范围重新执行 `prepare`，small 与 middle 会得到逐点相同的硬件样本。

以下命令生成每颗芯片 299 个可复现随机点和一个 baseline，并额外加入 GRACE 预测点。随机点只要求计算单元面积位于 baseline 的 90%–110%，功耗会被记录但不参与筛选：

```bash
python3 scripts/run_npu_area_latency_pareto.py prepare \
  --output-root results/npu_area_latency_pareto \
  --points-per-chip 300 \
  --seed 20260830

python3 scripts/run_npu_area_latency_pareto.py run \
  --output-root results/npu_area_latency_pareto \
  --max-concurrent 196 \
  --max-simulator-rss-gib 460

python3 scripts/run_npu_area_latency_pareto.py plot \
  --output-root results/npu_area_latency_pareto
```

最终图 `normalized_area_latency_pareto.png/pdf` 用各自 baseline 归一化面积和 E2E latency，并标注 baseline、GRACE 预测配置、Pareto front，以及面积不超过 baseline 时 latency 最低的实测配置。对应原始点和最优点分别保存在 `pareto_points.csv` 与 `pareto_summary.json`。
