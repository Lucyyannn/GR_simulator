# SystemThroughput figure

The canonical plotting entry is:

```bash
python3 scripts/plot_system_throughput.py \
  --input configs/figure_data/system_throughput.csv
```

It writes per-chip PDF, SVG, and PNG files to
`results/figures/SystemThroughput`. The figure uses CA on SSD as the normalized
baseline and the fixed legend names `RE`, `CA`, `O1`, `O1+O2`, and `REFORGE`.
The hollow red diamonds show REFORGE throughput in queries per second.

The input must contain one complete row per chip/model/sequence/batch case with
columns `chip`, `model`, `seq_len`, `batch_size`, `RE`, `CA`, `O1`, `O1+O2`,
`REFORGE`, and `reforge_qps`. The five method columns are normalized speedups;
`reforge_qps` is the absolute REFORGE throughput shown by the diamonds.

The plotting script uses all supplied values verbatim. It does not estimate
missing points, alter Small using Middle's trend, or generate REFORGE values.
It rejects incomplete tables. Both the normalized-speedup axis and each model's
QPS axis adjust automatically from the input data.
