# Challenge 3 / Observation 3 数据来源

本文档记录论文 **Challenge 3 / Observation 3** 中 Cube/Vector 资源失配表格和
NPU 配置搜索图的数据来源、计算口径与复现方法。表格数据来自已有仿真日志，DSE
图数据来自解析枚举；两者不能作为同一类测量值混用。

## 1. Cube/Vector 资源失配表格

### 1.1 统一 workload

四种 NPU 均使用同一个标准 Full-Cache GR ranking workload：

| 参数 | 数值 |
| --- | ---: |
| Model | HSTU-small |
| Layers | 4 |
| Hidden size | 256 |
| Batch size | 4 |
| History length | 8192 |
| Candidates per user | 128 |
| History handling | Full KV Cache |

Full-Cache 只计算当前 candidate 路径，并令 candidates 读取已有的历史 KV；它不重算
8192 个历史 token。Hot 和 cold 的存储路径不同，但计算 trace 相同，因此二者的
Cube/Vector active cycles 相同。

### 1.2 原始日志

每个 NPU 使用以下目录中的 `compute_activity.csv`：

```text
results/hstu_calibration_matrix_20260901/base_methods/cases/
  <NPU>/Full_Cache/HSTU-small_seq8192_bs4_<hot|cold>/compute_activity.csv
```

其中 `<NPU>` 为 `910A`、`910B`、`910C` 或 `MTIA2`。只读取
`scope == core_total` 的行：

```text
C_cube = sum(cube_active_cycles) / number_of_cores
C_vec  = sum(vector_active_cycles) / number_of_cores
```

因此表中的 cycles 是每个计算核的平均 active cycles，不是端到端时间，也不包含
HBM、DDR 或 SSD 等待时间。

### 1.3 Vector latency 权重

Vector cycles 已由仿真器按照 NPU 配置中的 primitive latency 计数。四种当前配置均为：

```text
ADD = 1, MUL = 1, SWISH = 8, DIV = 4, ADD_TREE = 7 cycles
```

例如 HSTU attention score 路径包含

```text
MUL -> SWISH -> DIV -> MUL
```

所以每个 Vector chunk 的 score 路径代价为

```text
1 + 8 + 4 + 1 = 14 cycles.
```

910C Full-Recompute 日志中的明细也直接验证了该比例：score SWISH 和 score DIV
分别是 score MUL cycles 的 8 倍和 4 倍。相关明细位于：

```text
results/hstu_calibration_matrix_20260901/base_methods/cases/910C/
  Full_Recompute/HSTU-small_seq8192_bs4_hot/compute_activity_detail.csv
```

### 1.4 面积与 R

面积采用固定的架构估计，不从仿真日志拟合：

```text
one Cube core:                 2.57 mm^2
one 2048-bit Vector core:      0.70 mm^2
Vector area scaling:           linear with Vector width
```

对配置 `(N_c, N_v, W_v)`：

```text
A_cube = 2.57 * N_c
A_vec  = 0.70 * N_v * (W_v / 2048)
```

论文中的面积归一化 Cube-to-Vector cycle ratio 为：

```text
R = (C_cube / A_cube) / (C_vec / A_vec)
  = (C_cube / C_vec) * (A_vec / A_cube).
```

因此，原始 `C_cube/C_vec` 不能直接作为 R。以 910C 为例，日志中的 cycle ratio
为 0.577，但乘以面积比例 `67.20/123.36` 后，R 为 0.315。

### 1.5 最终表格数值

| NPU | Cube cycles | Vector cycles | Cycles C/V | Cube area | Vector area | Area C/V | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ascend 910A | 6.82e4 | 2.38e5 | 0.29 | 82.24 | 22.40 | 3.67 | 0.08 |
| Ascend 910B | 9.09e4 | 1.57e5 | 0.58 | 61.68 | 33.60 | 1.84 | 0.31 |
| Ascend 910C | 4.55e4 | 7.87e4 | 0.58 | 123.36 | 67.20 | 1.84 | 0.31 |
| MTIA 2i | 1.36e5 | 4.75e5 | 0.29 | 41.12 | 11.20 | 3.67 | 0.08 |

表中显示值经过舍入；计算 R 时使用未舍入的 cycles 和 area。

## 2. 910C Cube/Vector DSE 图

### 2.1 数据生成入口

解析枚举和绘图脚本为：

```text
scripts/plot_cube_vector_dse.py
```

在已有项目容器中复现：

```bash
docker exec -w /workspace/GR_simulator grsim1 \
  bash -lc 'PYTHONDONTWRITEBYTECODE=1 python3 scripts/plot_cube_vector_dse.py'
```

脚本生成：

```text
results/figures/cube_vector_dse/
  cube_vector_dse_910c_hstu_small_bs4_seq8k.csv
  cube_vector_dse_910c_hstu_small_bs4_seq8k_linear.pdf
  cube_vector_dse_910c_hstu_small_bs4_seq8k_linear.png
```

CSV 保存所有可行设计点及其核数、Vector width、面积、cycles、R、latency 和归一化
latency；PDF 是论文优先使用的矢量图。

### 2.2 图中 workload 与解析计算量

当前 DSE 图使用 4-layer HSTU-small、batch size 4、history length 8192、128
candidates per user。与第 1 节的 Full-Cache 表格不同，当前图的解析 workload 会重算
完整历史序列。令：

```text
L = 4, H = 256, N = 4, S = 8192, C = 128, number of heads = 4
T = N(S + C)
E = N[S(S + 1)/2 + CS + C].
```

Cube work 显式包含输入投影、输出投影、QK 和 AV：

```text
F_cube = L [14 T H^2 + 4 E H].
```

Vector work 显式包含输入/输出 LayerNorm、SiLU、gating 和 attention score 操作。
当前脚本用 910C primitive latency 构造 latency-weighted operation count：

```text
w_ln    = 2*l_tree + 2*l_add + 3*l_mul = 19
w_token = (w_ln + l_swish) + (w_ln + l_mul) = 47
w_attn  = 2*l_mul + l_swish + l_div = 14
O_vec   = L [w_token T H + w_attn * heads * E].
```

这些是公开的架构 primitive latency，不是针对 workload 拟合的系数。

### 2.3 候选 NPU 的 cycles、R 与 latency

对于候选 `(N_c, N_v, W_v)`，每个 64x64 Cube core 每周期完成 8192 FLOPs；
一个 `W_v`-bit Vector core 每周期处理 `W_v/16` 个 FP16 elements。因此：

```text
C_cube = F_cube / (8192 * N_c)
C_vec  = O_vec / [N_v * (W_v / 16)]

A_cube = 2.57 * N_c
A_vec  = 0.70 * N_v * (W_v / 2048)

R = (C_cube / A_cube) / (C_vec / A_vec)
T_compute = (C_cube + C_vec) / frequency.
```

910C frequency 固定为 1.8 GHz。图中的 normalized latency 为：

```text
Normalized Latency = T_compute(candidate) / T_compute(original 910C).
```

该图只建模 Cube 和 Vector 的串行解析计算时间，不包含访存、调度、tiling、流水重叠
或仿真器启动开销。

### 2.4 枚举范围与面积约束

```text
N_c: 1 ... 64, step 1
N_v: 1 ... 128, step 1
W_v: 1024 ... 8192 bits, step 1024
```

原始 910C 为 `(48, 48, 4096)`，计算面积为 190.56 mm²。只保留总计算面积位于
原始 910C 的 99%--100% 之间的候选，共 501 个设计点。当前解析结果为：

| Design | `(N_c, N_v, W_v)` | Area (mm²) | R | Latency (ms) | Normalized latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline NPU | `(48, 48, 4096)` | 190.56 | 0.360 | 2.448 | 1.000 |
| Optimal Design | `(39, 43, 6144)` | 190.53 | 0.984 | 2.296 | 0.938 |

对应的解析加速为 1.066x。

### 2.5 绘图筛选

完整的 501 个可行点全部保存在 CSV 中。论文单栏线性图为了避免极端失衡设计压缩
主要区域，只显示：

```text
R <= 3.0 and Normalized Latency <= 1.5.
```

图中的青绿色空心圆表示 `Feasible Designs`，灰色实心菱形表示 `Baseline NPU`，
黄色五角星表示 `Optimal Design`；`R=1` 使用浅灰色竖直虚线。图中没有拟合曲线。

## 3. 使用时的口径约束

1. 表格报告已有 Full-Cache 仿真日志中的 active cycles；DSE 图报告 full-history
   recompute 的解析 cycles 和解析 latency。
2. 表格中的 cycles、图中的解析 cycles 和端到端 latency 是三种不同指标，引用时必须
   明确标注。
3. 若 Challenge 3/Observation 3 要求表格和 DSE 图严格使用同一 workload 语义，应先
   决定统一采用 Full-Cache 还是 full-history recompute，再重新生成其中一组数据。
4. 修改 NPU 核数、Vector width、primitive latency 或面积常数后，必须重新计算 R；
   不能只移动图中的 baseline 标记。
