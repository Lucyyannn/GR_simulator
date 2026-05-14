# report

配置：HSTU-small，cold user，embedding 初始在 SSD，KV Cache 初始在 SSD，开启 KV Reuse，reuse_ratio=0.4791，只对 item 做 recompute。

估算结果：
- stream: recompute_ratio=68.7012% (history_recompute_len=1407), 
- random: recompute_ratio=19.9707% (history_recompute_len=409), 

| group | index mode | target ratio | history_recompute_len | actual item ratio | simulation time (us) | attention op time (us) | NPU util (%) | HBM util (%) | DDR util (%) | DDR BW (GB/s) | SSD util (%) | SSD BW (GB/s) | result dir |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stream_0 | stream | 0% | 0 | 0.00% | 1539.130000 | 46.803700 | 0.436811 | 0.595529 | 0 | 0 | 92.055 | 6.03292 | `results/IRrecompute比例探究/stream_0` |
| stream_20 | stream | 20% | 410 | 20.02% | 1285.600000 | 123.254020 | 1.5015 | 1.60925 | 0 | 0 | 92.8511 | 6.08509 | `results/IRrecompute比例探究/stream_20` |
| stream_40 | stream | 40% | 819 | 39.99% | 1186.750000 | 163.508890 | 2.38982 | 2.57736 | 0 | 0 | 81.7446 | 5.35722 | `results/IRrecompute比例探究/stream_40` |
| stream_60 | stream | 60% | 1229 | 60.01% | 925.164000 | 202.775090 | 4.16872 | 4.4508 | 0 | 0 | 80.481 | 5.27441 | `results/IRrecompute比例探究/stream_60` |
| stream_80 | stream | 80% | 1638 | 79.98% | 943.055000 | 228.086920 | 4.79277 | 5.42754 | 0 | 0 | 55.3311 | 3.62618 | `results/IRrecompute比例探究/stream_80` |
| stream_100 | stream | 100% | 2048 | 100.00% | 1112.370000 | 274.733230 | 4.98598 | 5.54478 | 0 | 0 | 26.7279 | 1.75164 | `results/IRrecompute比例探究/stream_100` |
| stream_optimal | stream | optimal | 1407 | 68.70% | 879.708000 | 213.844340 | 4.64316 | 5.13502 | 0 | 0 | 73.4278 | 4.81216 | `results/IRrecompute比例探究/stream_optimal` |
| random_20 | random | 20% | 410 | 20.02% | 1525.600000 | 123.255340 | 1.26555 | 1.35632 | 0 | 0 | 92.8697 | 6.08631 | `results/IRrecompute比例探究/random_20` |
| random_40 | random | 40% | 819 | 39.99% | 1666.750000 | 163.515500 | 1.70206 | 1.83574 | 0 | 0 | 84.787 | 5.5566 | `results/IRrecompute比例探究/random_40` |
| random_60 | random | 60% | 1229 | 60.01% | 1605.160000 | 202.776300 | 2.40591 | 2.56864 | 0 | 0 | 87.608 | 5.74148 | `results/IRrecompute比例探究/random_60` |
| random_80 | random | 80% | 1638 | 79.98% | 1823.050000 | 228.092140 | 2.48104 | 2.80944 | 0 | 0 | 76.4672 | 5.01135 | `results/IRrecompute比例探究/random_80` |
| random_100 | random | 100% | 2048 | 100.00% | 2232.350000 | 274.729320 | 2.48528 | 2.76382 | 0 | 0 | 61.6872 | 4.04273 | `results/IRrecompute比例探究/random_100` |
| random_optimal | random | optimal | 409 | 19.97% | 1525.540000 | 123.187060 | 1.2648 | 1.35483 | 0 | 0 | 92.9152 | 6.08929 | `results/IRrecompute比例探究/random_optimal` |
