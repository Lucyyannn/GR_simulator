# User-Activity-Aware KV Placement

This analysis compares `RE`, randomly placed `CA`, randomly placed `w/o IKM`, and
activity-aware `w/ IKM`. It currently uses the representative
910C / HSTU-large / length=8K / batch=1 workload and a full 128 GiB DDR budget.

`w/o IKM` accounts for Action KV eliminated by AR when calculating capacity,
but selects its DRAM users randomly. `w/ IKM` accounts for both Action KV
eliminated by AR and recomputed Item KV, and selects the most active users.
The activity-aware policy uses the empirical interaction counts in
`configs/kuairand_1k_user_activity_distribution.csv`. The 1K distribution is
replicated unchanged to form 10K, 20K, and 100K populations. CA placement is
random, so its expected interaction hit rate equals its resident-user fraction.

The persistent storage per user is

```
KV bytes = 2 * layers * hidden_size * cached_rows * bytes_per_element
```

where the factor two represents K and V. The weighted latency and throughput
are

```
latency = hit_rate * latency_DRAM + (1-hit_rate) * latency_SSD
QPS     = batch_size * 1e6 / latency_us
```

Inputs, including their provenance, are stored in
`configs/figure_data/user_activity_placement_inputs.json`. The current REFORGE
endpoints and 879 recomputed Item rows are provisional 0903-derived values;
replace them after the corrected AR-enabled large-model results finish.

Run:

```bash
python3 scripts/analyze_user_activity_placement.py
python3 scripts/plot_ikm_hit_rate.py
```

The table is written to `results/analysis/user_activity_placement/summary.csv`.
The PDF/SVG/PNG figure is written as `IKMHitRate` under
`results/figures/IKMHitRate/`. The old `plot_user_activity_placement.py` name is
retained as a compatibility entry point.
