# P99 latency figure

Run:

```bash
python3 scripts/plot_p99_latency.py
```

The current figure represents Ascend 910C, HSTU-Large, sequence length 8192,
and batch size 1. It uses the same steady-state M/M/1 definition as
`scripts/compute_mm1_p99.py`:

```text
P99 = ln(100) / (mu - lambda)
```

where `mu` is the service rate derived from each method's QPS and `lambda` is
the request rate. The plotted values are stored directly in
`configs/figure_data/p99_latency.csv`; the plotting script does not synthesize
or transform them. Curves retain their real P99 values above 500 ms and are
naturally clipped by the axis boundary, rather than replacing the first
out-of-range value with 500 ms. Denser sampling at 8--32 query/s separates RE
and CA before either queue saturates. The x-axis spans 0--100 query/s with
20-query/s major ticks while retaining the original sampled points. The current
REFORGE values remain
provisional until measured results replace the table.
