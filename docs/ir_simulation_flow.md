# IR Simulation Flow

This document describes the current calibrated IR scalability flow.

## Entry Point

Use the canonical scalability runner:

```bash
bash scripts/run_scalability.sh \
  --result-root results/hstu_modelsize_scalability_$(date +%Y%m%d_%H%M%S) \
  --docker-container gr-simulator-mini \
  --max-concurrent 45
```

For summary regeneration only:

```bash
bash scripts/run_scalability.sh \
  --result-root <existing-result-root> \
  --summary-only
```

## Main Files

- `scripts/run_scalability.sh`: top-level orchestration for the current single-NPU HSTU-small/middle/large sweep.
- `scripts/calibrate_memory_bandwidth.py`: memory bandwidth calibration through simulator `mem_bench`.
- `scripts/recompute_ratio_cost_model_new.py`: formula-based IR estimate and calibrated final `history_recompute_len`.
- `scripts/summarize_scalability_results.py`: shared summary writer for current model-size runs and older NPU-layout results.
- `scripts/run_hstu.sh`: builds one HSTU run directory and invokes `./build/bin/Simulator`.
- `scripts/recompute_ratio_calibration.json`: base calibration file merged with per-run calibration.
- `configs/910A.json`, `configs/910B.json`, `configs/910C.json`: simulator hardware configs.

## Inputs

Current scalability defaults follow `AGENTS.md`:

- single NPU
- single user
- model sizes: HSTU-small, HSTU-middle, HSTU-large
- `kv_len=4096`
- `candidates_per_user=128`
- source media: `ddr`, `ssd`
- chips: `910A`, `910B`, `910C`
- schemes: `Full_Cache`, `Full_Recompute`, `w_AR`, `w_IR`, `w_both`
- `w_both` uses `kv_reuse_ratio=0.4360`

Source user mode is selected by medium:

- `ddr`: hot user
- `ssd`: cold user

## Flow

1. `run_scalability.sh` parses model sizes, chips, source media, and schemes.
2. Unless `--skip-calibration` is set, it calls `calibrate_memory_bandwidth.py`.
3. Memory calibration writes a merged calibration JSON under `MISC/hstu_modelsize_calibration_cache`.
4. Final scalability cases compute `w_IR` and `w_both` k through `recompute_ratio_cost_model_new.py`, using the merged memory calibration.
5. Final cases run in parallel through `scripts/run_hstu.sh`.
6. `summarize_scalability_results.py --layout modelsize` writes `summary.md`, `scalability_summary.csv`, `time_qps.csv`, `time_qps.xlsx`, and `reproduce.md`.

## Output Layout

Important outputs:

- `<result-root>/summary.md`
- `<result-root>/scalability_summary.csv`
- `<result-root>/time_qps.csv`
- `<result-root>/time_qps.xlsx`
- `<result-root>/recompute_choices.csv`
- `<result-root>/cases/<small|middle|large>/<DRAM|SSD>/910*/<scheme>/hardware_summary.csv`
- `<result-root>/cases/<small|middle|large>/<DRAM|SSD>/910*/<scheme>/layer_breakdown.csv`

完整版复现文档见：

- `docs/reproducibility_hstu_scalability.md`

Calibration cache:

- `MISC/hstu_modelsize_calibration_cache/<cache-key>/memory`

重算公式与修正细节见：

- `IR_Recompute.md`
