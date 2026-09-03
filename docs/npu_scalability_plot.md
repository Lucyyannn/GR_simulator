# NPU Scalability figure

The canonical plotting entry is:

```bash
python3 scripts/plot_npu_scalability.py
```

The default input is `configs/figure_data/npu_scalability.csv`. Each row gives
the absolute QPS for one HSTU-Large workload and has the following columns:

```text
chip,model,seq_len,batch_size,RE,CA,REFORGE
```

The table must contain every combination of four chips (`910A`, `910B`,
`910C`, and `MTIA2`), three sequence lengths (4096, 6144, and 8192), and three
batch sizes (1, 2, and 4). All QPS values must be positive.

The plotting script uses the supplied values verbatim. It does not fit missing
rows, interpolate values, normalize QPS, synthesize REFORGE results, or alter
cross-NPU trends. It validates the table and stops with an error if a required
row or column is absent. Axis limits are selected automatically from the input.

Three single-column figures are written for batch sizes 1, 2, and 4 in PDF,
SVG, and PNG formats under `results/figures/NPUScalability`. To use another
complete table or output prefix:

```bash
python3 scripts/plot_npu_scalability.py \
  --input path/to/npu_scalability.csv \
  --output path/to/NPUScalability
```

The checked-in table currently contains provisional values for layout review.
Replace those cells with measured results when they become available; no code
change is required.
