# AR Attention-Compute Rerun Plan

## Objective

Regenerate every result whose method contains Action Reuse using
`--enable-ar-reduce-attention-compute`. The reuse ratio remains `0.4802`.
Full Cache and Full Recompute are AR-independent baselines and may be reused
after input-digest validation.

## Phase 0: semantics and safeguards

1. Change every AR-capable batch runner to pass
   `--enable-ar-reduce-attention-compute` explicitly.
2. Keep the CLI switch explicit rather than relying on the `run_hstu.sh`
   default.
3. Include the AR compute mode in case identity/input digests so an old result
   cannot be resumed accidentally.
4. Reject a completed AR result unless trace metadata reports
   `ar_reduce_attention_compute=true` and the recorded command contains the
   enable flag but not the disable flag.
5. Update the item-KV model so its Cube/Vector attention work uses the reduced
   effective history length consistently with the simulator.

## Phase 1: representative validation

Run a small validation set before launching a matrix:

- 910A, HSTU-small, seq8192, bs4, hot;
- 910A, HSTU-middle, seq8192, bs4, hot;
- 910C, HSTU-large, seq8192, bs4, cold;
- MTIA2, HSTU-middle, seq6144, bs2, hot.

For each point, compare Full Cache, old AR, and new AR. Verify that:

- cached KV bytes fall according to the 0.4802 reuse ratio;
- QK/AV score elements and attention latency also use the reduced history;
- candidate count and model output shapes do not change;
- hot/cold source placement remains DDR/SSD respectively;
- no result is read from a 0903 legacy AR directory.

## Phase 2: new AR baseline matrix

Rerun `w_AR` over the full 216-case matrix:

- NPU: 910A, 910B, 910C, MTIA2;
- model: HSTU-small, middle, large;
- sequence length: 4096, 6144, 8192;
- batch size: 1, 2, 4;
- user: hot, cold.

Matching Full Cache and Full Recompute outputs can be linked by manifest rather
than simulated again. Missing or failed RE/CA cases should be completed in the
new baseline root.

## Phase 3: AR+IR ratio data and calibration

Rerun ratios `0.0, 0.1, ..., 1.0` for the same matrix, totaling 2,376 cases.
Submit small and middle before large. Calibrate hardware-only efficiency,
saturation, and startup parameters from the new data. Do not introduce
model-specific coefficients.

Validate both latency prediction and optimum selection with held-out cases.
The final optimizer continues to enumerate every integer item-token count, so
its selected ratio is not restricted to the 0.1 grid.

## Phase 4: final w_both matrix

Use the recalibrated cost model to choose a fresh ratio for every workload and
run the 216-case `w_both` matrix. Compare predicted and measured latency and
also report the gap from the best neighboring ratio-sweep points.

## Phase 5: NPU reconfiguration and GRACE

Repeat analytic search, pilot simulation, and full guard validation separately
for each baseline NPU. Every candidate/workload pair must recompute its own
`w_both` ratio under the candidate Cube/Vector throughput. Keep the existing
area, power-recording, concurrency, memory, and 0.95x guard conventions.

## Phase 6: QPS and figures

Regenerate all comparisons containing AR, `w_both`, or GRACE. RE/CA QPS may be
reused only when configuration and input digests match. Figures must record the
new result roots and AR compute mode in their data manifest.

## Execution limits

Use at most 196 CPU cores and less than 460 GB total memory. A single scheduler
must account for all active Simulator processes. All new phases use fresh
result roots and support successful-result reuse only within the same AR
compute semantics.
