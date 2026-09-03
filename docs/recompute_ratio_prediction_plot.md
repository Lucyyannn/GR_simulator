# ItemKVRecomputeRatio Figure

The canonical plotting entry is
`scripts/plot_item_kv_recompute_ratio.py`. The older
`plot_recompute_ratio_prediction.py` name remains available as the underlying
compatibility implementation.

This figure currently uses **temporary layout data**, stored in
`configs/figure_data/recompute_ratio_prediction.csv`. It must not be reported
as a measured experimental result. Replace the table values directly when the
random- and continuous-embedding sweeps and the cost-model prediction are
available; the plotting script performs no data fitting or trend adjustment.

The table contains 11 grid ratios and one non-grid predicted ratio. The first
curve represents random history-embedding accesses (`w/o Embedding Opt.`), and
the second represents contiguous history-embedding accesses (`w/ Embedding
Opt.`). They coincide at ratio 0. The predicted ratio is inserted into both
smooth curves, but only its embedding-optimized point is marked. The input is
validated so that this marked point outperforms all 11 optimized grid points.

Run:

```bash
python3 scripts/plot_item_kv_recompute_ratio.py
```

The output figure is written to
`results/figures/ItemKVRecomputeRatio/ItemKVRecomputeRatio.pdf`.

The old `prepare_recompute_ratio_prediction.py` helper is archived locally and
excluded from Git. Its analytical subtraction must not be used to claim an
additional embedding optimization, because the current ratio sweep already
uses continuous history-embedding indices.
