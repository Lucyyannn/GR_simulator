# Activity-weighted QPS estimate

`scripts/analyze_activity_weighted_qps.py` estimates QPS when the most active
users' persistent K/V is placed in a finite DDR capacity and the remaining
users' K/V stays on SSD.

The default command analyzes the historical 0903 matrices:

```bash
python3 scripts/analyze_activity_weighted_qps.py
```

Defaults are 128 GiB of DDR, FP16 K/V, and 100 copies of every user in
`configs/kuairand_1k_user_activity_distribution.csv`, producing 100K users
without changing the empirical activity distribution. Full Recompute and Full
KV Cache always use the measured cold/SSD latency. AR and `w_both` use their
measured hot/DDR and cold/SSD endpoints.

For a method's hot case, persistent storage per user is

```text
cached_rows = item_rows - recomputed_item_rows + retained_action_rows
KV_bytes    = 2 * layers * hidden * cached_rows * bytes_per_element
```

The leading 2 represents K and V. Recomputed Item K/V and Action K/V removed
by reuse therefore consume no DDR capacity. Users are admitted by decreasing
interaction count. If capacity ends within one replicated activity bucket,
only the users that fit are admitted from that bucket.

Let `p_hot` be the fraction of all interactions issued by DDR-resident users.
The estimate is

```text
weighted_latency = p_hot * hot_latency + (1 - p_hot) * cold_latency
weighted_QPS     = batch_size * 1e6 / weighted_latency_us
```

To analyze refreshed results, supply their roots without editing the script:

```bash
python3 scripts/analyze_activity_weighted_qps.py \
  --base-root results/NEW_BASE_MATRIX \
  --both-root results/NEW_BOTH_MATRIX \
  --output-root results/analysis/activity_weighted_qps_new
```

The output contains per-case values, per-chip geometric means, a common-case
per-chip summary for fair AR-versus-`both` comparison, missing-case records,
and analysis metadata. The historical 0903 AR results disabled AR-induced
Attention-compute reduction, so their output is provisional and must not be
used as the corrected paper result.
