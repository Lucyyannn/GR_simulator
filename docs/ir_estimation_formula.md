# IR Estimation Formula

This document summarizes the calibrated IR estimator.  
For a non-time-first derivation aligned to current implementation, please also see:

- `IR_Recompute.md`

## Objective

IR chooses `history_recompute_len = k`: the number of recent history rows
recomputed on the NPU instead of loaded as KV cache from DDR or SSD.

The goal is to hide KV preload behind useful compute without adding excessive
recompute.

## Previous Estimate

The earlier estimator scanned candidate k and minimized:

```text
abs(T_kv_layer(k) - T_compute_layer(k))
```

Where:

- `T_kv_layer(k)` is the per-layer KV preload time for history rows that remain cached.
- `T_compute_layer(k)` is the whole-layer NPU compute time.

This was too coarse because not all layer compute can hide KV preload. In the
actual trace schedule, the late cached-attention part needs the KV data and
therefore cannot hide that same KV preload. Counting the whole layer made the
model overestimate overlap capacity.

The previous logic also had two practical issues:

- `w_both`/AR was not fully reflected in all relevant terms: AR reduces cached
  KV reads, attention compute, and HBM traffic.
- `k=0` was parsed incorrectly in calibration. With no recompute split, the
  trace has plain `hstu::attention` instead of `hstu::attention.recompute_early`
  and `hstu::attention.cached_late`. Treating plain attention start as the
  overlap point hides the DDR/SSD wait inside the measured start time.

## Current Formula

The current estimator scans k and minimizes:

```text
abs(T_kv_layer(k) - T_pre_cached_compute_layer(k))
```

Where:

- `T_kv_layer(k)` is the remaining cached KV preload time.
- `T_pre_cached_compute_layer(k)` is the compute before cached-late attention.
- Cached-late attention is excluded from the overlap budget because it needs
  the cached KV preload to be complete.

The reported pipeline estimate is:

```text
T_pipeline_layer(k) =
  max(T_kv_layer(k), T_pre_cached_compute_layer(k)) + T_late_layer(k)
```

This makes the balancing point match the simulator schedule more closely.

## Memory Terms

For a given k:

```text
item_count = ceil(kv_len / 2)
action_count = floor(kv_len / 2)
recomputed_items = min(k, item_count)
recomputed_actions = max(0, k - item_count)
remaining_items = item_count - recomputed_items
remaining_actions = action_count - recomputed_actions
```

Without AR:

```text
effective_cached_rows = remaining_items + remaining_actions
```

With AR/w_both:

```text
action_reuse_ratio = clamp(kv_reuse_ratio * kv_len / action_count, 0, 1)
physical_remaining_actions =
  round(remaining_actions * (1 - action_reuse_ratio))
effective_cached_rows = remaining_items + physical_remaining_actions
```

Then:

```text
KV preload bytes =
  batch_per_npu * effective_cached_rows * 2 * hidden * precision_bytes

T_kv_layer = KV preload bytes / effective_source_bandwidth
```

The effective source bandwidth comes from the simulator config and calibration.
Config metadata/target bandwidth is preferred over raw channel math so the
estimator stays aligned with nbl-adjusted simulator accounting.

## Compute Terms

The estimator splits attention work into:

- early recompute-side attention work
- late cached-side attention work

Early work participates in overlap:

```text
T_pre_cached_compute_layer =
  T_linear/silu/split before attention
  + T_recompute_early
```

Late work is added after the overlap max:

```text
T_late_layer =
  T_cached_late
  + T_layer_norm/mul/final_linear after attention
```

AR/w_both reduces the cached-side score elements and the associated HBM traffic,
because fewer cached action rows are physically loaded and attended.

## Calibration Corrections

The formula alone still does not perfectly match the simulator because effective
bandwidth and effective compute throughput are workload-dependent. The
calibration step corrects this with small 2-layer cases.

For each context:

```text
context = chip / source_medium / batch_per_npu / scheme
```

Calibration records:

- `preload_peak_util`
- `hbm_peak_util`
- `compute_scale_mult`
- `pre_cached_compute_scale_mult`
- exact `history_recompute_len`
- model shape guards: hidden, kv_len, candidates, batch size

When the final scalability run matches the guarded shape, the exact calibrated
k is used directly. CLI `--fixed-recompute-len` still has the highest priority.

## k=0 Handling

For `k=0`, the simulator does not emit split attention ops. It emits:

```text
hstu::attention
```

The corrected calibration and summary logic do not use the delayed attention
start as proof that preload is hidden. Instead:

```text
pre_cached_compute_end =
  end time of the last compute op before hstu::attention

gap = preload_end - pre_cached_compute_end
wait = max(0, attention_start - pre_cached_compute_end)
```

This exposes the real DDR/SSD wait. For example, in the final NPU1/DRAM results:

- `910A/w_IR, k=0`: average wait is about `166.67 us`
- `910B/w_IR, k=0`: average wait is about `179.33 us`
- `910C/w_IR, k=0`: average wait is about `239.80 us`

So k=0 is not automatically “perfect overlap”; it is only one candidate. The
current calibration can now see its true wait and can add midpoint candidates
between sign-changing k values.

## Adaptive Candidate Expansion

Initial candidates are centered around the formula estimate:

```text
base, base +/- radius, base +/- radius/2
```

Extra boundary candidates are added near 0 or max k. Calibration now also adds
midpoints when two tested candidates bracket the target:

```text
gap(left) * gap(right) <= 0
```

This avoids choosing between only `k=0` and `k=255` when the true best point is
between them.

## Remaining Limitation

The calibration objective is local preload overlap, not end-to-end latency.
Some DDR multi-NPU cases can choose a larger k that improves the local overlap
metric but increases total latency. If final latency becomes the primary target,
the next step should add a second-stage latency-aware cap or a blended objective.
