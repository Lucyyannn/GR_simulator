# HSTU 可复现实验说明（full scalability）

本文档是当前仓库 `scripts/run_scalability.sh` 的完整实验复现流程。  
实验对象是 AGENTS 约定的单芯片、单 NPU、单用户下
`HSTU-small / middle / large` 的 DRAM、SSD 全量对比。

## 1. 需要的脚本与输入文件

- `scripts/calibrate_memory_bandwidth.py`
  - 做内存基准校准，产生有效带宽与峰值带宽对齐信息。
- `scripts/recompute_ratio_cost_model_new.py`
  - 根据模型形状/硬件参数/校准信息估计每个 case 的 `history_recompute_len`。
- `scripts/run_hstu.sh`
  - 生成单条实验的 trace 并调用 `./build/bin/Simulator`。
- `scripts/run_scalability.sh`
  - 负责 DRAM/SSD×910A/B/C×small/middle/large×5 种方案的批量调度。
- `scripts/summarize_scalability_results.py`
  - 实验完成后输出 `summary.md / scalability_summary.csv / time_qps.csv / time_qps.xlsx / reproduce.md`。
- `scripts/recompute_ratio_calibration.json`
  - 默认基准校准文件（可替换）。
- `configs/910A.json`、`configs/910B.json`、`configs/910C.json`
  - 当前实验的硬件配置。

## 2. 推荐顺序（先校准再跑完整实验）

### 2.1 内存带宽校准（推荐一次完成，多次复用）

```bash
python3 scripts/calibrate_memory_bandwidth.py \
  --result-root MISC/hstu_modelsize_calibration_cache/manual_$(date +%Y%m%d_%H%M%S)/memory \
  --calibration scripts/recompute_ratio_calibration.json \
  --merged-calibration-output MISC/hstu_modelsize_calibration_cache/manual_$(date +%Y%m%d_%H%M%S)/memory/recompute_ratio_calibration_memory_merged.json \
  --chips 910A,910B,910C \
  --patterns contiguous,random_512b_index \
  --access-types read \
  --sizes-bytes 512,1024,2048 \
  --burst-counts 1,2,4,8 \
  --max-concurrent 30 \
  --poll-interval 10 \
  --log-level warn
```

`run_scalability.sh` 会自动复用上述结果到：

- `MISC/hstu_modelsize_calibration_cache/<hash>/memory/recompute_ratio_calibration_memory_merged.json`

其中 `<hash>` 由脚本根据硬件配置、pattern、chip 列表自动计算，后续只要这些输入不变就可复用。

### 2.2 全量实验

```bash
bash scripts/run_scalability.sh \
  --result-root results/hstu_modelsize_$(date +%Y%m%d_%H%M%S) \
  --max-concurrent 45 \
  --calibration-cache-root MISC/hstu_modelsize_calibration_cache \
  --docker-container gr-simulator-mini \
  --kv-len 4096 \
  --schemes Full_Cache,Full_Recompute,w_AR,w_IR,w_both
```

说明：

- 不加 `--force-calibration` 时，若对应 hash 缓存已存在则直接复用内存校准；若不存在才会重新跑 `calibrate_memory_bandwidth.py`。
- 若明确知道使用的是已经落盘好的 merged calibration，可用 `--skip-calibration`，前提是 `--calibration` 指向你要复用的文件。
- `run_scalability.sh` 在每个 `w_IR` / `w_both` case 下会生成 `ir_selection.json`，其中包含该 case 的候选 `history_recompute_len` 评估明细。

## 3. 何时需要重新校准

- **需要重跑校准**：
  - 修改 `configs/910A.json / 910B.json / 910C.json` 中与带宽、频率、核数相关字段；
  - 修改校准参数（patterns、sizes、burst、access_types）；
  - 改变模拟器代码导致 mem_bench 口径变化。
- **通常不需要重跑**：
  - 仅改变方案（Full_Cache/Full_Recompute/w_AR/w_IR/w_both）；
  - 仅改变模型规模（small/middle/large）；
  - 固定同一套硬件配置时，IR 的缓存上下文可直接复用同一份 merged calibration。

## 4. 用已有校准跑单独 HSTU case（非完整 scalability）

先算 k，再跑 trace，保证与 scalability 实验一致：

```bash
CALIB=MISC/hstu_modelsize_calibration_cache/<hash>/memory/recompute_ratio_calibration_memory_merged.json

K=$(python3 scripts/recompute_ratio_cost_model_new.py \
  --config configs/910A.json \
  --calibration "${CALIB}" \
  --user hot \
  --layers 8 \
  --hidden 512 \
  --kv-len 4096 \
  --batch-size 4 \
  --candidates 128 \
  --embedding-source ddr \
  --objective balance \
  --field len)

bash scripts/run_hstu.sh \
  --base-config configs/910A.json \
  --result-dir results/one_case \
  --source-medium ddr \
  --embedding-source-medium ddr \
  --layers 8 \
  --hidden 512 \
  --kv-len 4096 \
  --num-users 1 \
  --users-per-batch 4 \
  --candidates-per-user 128 \
  --macro-batch-size 128 \
  --npu-count 1 \
  --history-recompute-len "${K}" \
  --log-level info
```

若为 `w_both`，加入：

```bash
--enable-kv-reuse --kv-reuse-ratio 0.4360
```

`--enable-kv-reuse` 组合 `w_AR` 只改变 KV 访问行为，不改 HBM 内核参数。

## 5. 实验后如何汇总（可复现产物）

完整 run 完成后，脚本已自动生成：
- `summary.md`
- `scalability_summary.csv`
- `time_qps.csv`
- `time_qps.xlsx`
- `recompute_choices.csv`
- `reproduce.md`（含该次实验可重放命令）

如只想重算报告，可执行：

```bash
bash scripts/run_scalability.sh --result-root results/<your-root> --summary-only
```

或：

```bash
python3 scripts/summarize_scalability_results.py \
  --layout modelsize \
  --result-root results/<your-root>
```
