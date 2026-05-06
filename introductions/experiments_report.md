# Baseline：Layer-level pipeline (Fused HSTU Attention)

## 默认负载配置

| parameter | value |
| --- | --- |
| source_medium | ddr |
| base_config | configs/baseline.json |
| layers | 8 |
| hidden | 256 |
| kv_len | 8192 |
| num_users | 4 |
| users_per_batch | 4 |
| candidates_per_user | 512 |
| macro_batch_size | 512 |
| vocab | 262144 |
| seed | 1234 |
| op_modeling | split=materialize,view=materialize,concat=materialize |
| attention_modeling | fused |
| kv_reuse_variant | window_topk |
| kv_reuse_window_size | 1024 |
| kv_reuse_topk | 4 |
| kv_reuse_hot_share | 0.75 |
| log_level | warn |

## 仿真器配置

| field | value |
| --- | --- |
| config | configs/baseline.json |
| num_cores | 8 |
| core_freq_mhz | 763 |
| core_array | 256x256 |
| icnt_freq_mhz | 1021 |
| hbm_freq_mhz | 1563 |
| hbm_channels | 64 |
| hbm_req_size | 32 |
| hbm_size_gb | 128 |
| ddr_freq_mhz | 1600 |
| ddr_channels | 12 |
| ddr_req_size | 32 |
| ddr_size_gb | 768 |
| precision_bytes | 2 |

## 不同KV length（KV Reuse on/off）

| varying value | case | off status | on status | off e2e(us) | on e2e(us) | e2e improve(%) | off attn op(us) | on attn op(us) | attn op improve(%) | off attn movin(us) | on attn movin(us) | attn movin improve(%) | off attn movin(MiB) | on attn movin(MiB) | attn bytes improve(%) | off KV preload(us) | on KV preload(us) | KV preload improve(%) | off KV preload(MiB) | on KV preload(MiB) | KV bytes improve(%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | kv_len_2048 | ok | ok | 1773.648 | 1207.893 | 31.898 | 359.117 | 227.679 | 36.600 | 1058.727 | 794.757 | 24.933 | 88.000 | 64.250 | 26.989 | 1482.216 | 933.346 | 37.030 | 64.000 | 40.250 | 37.109 |
| 4096 | kv_len_4096 | ok | ok | 3299.065 | 2168.404 | 34.272 | 717.326 | 454.942 | 36.578 | 1764.357 | 1238.417 | 29.809 | 152.000 | 104.500 | 31.250 | 2965.965 | 1865.389 | 37.107 | 128.000 | 80.500 | 37.109 |
| 8192 | kv_len_8192 | ok | ok | 6350.895 | 4088.486 | 35.623 | 1433.811 | 909.551 | 36.564 | 3175.696 | 2126.439 | 33.040 | 280.000 | 185.000 | 33.929 | 5925.564 | 3728.212 | 37.083 | 256.000 | 161.000 | 37.109 |
| 12288 | kv_len_12288 | ok | ok | 9324.077 | 5962.988 | 36.047 | 1518.781 | 994.409 | 34.526 | 4759.351 | 3184.888 | 33.081 | 408.000 | 265.500 | 34.926 | 8889.260 | 5592.802 | 37.084 | 384.000 | 241.500 | 37.109 |
| 16384 | kv_len_16384 | ok | ok | 12285.699 | 7826.344 | 36.297 | 1520.706 | 996.073 | 34.499 | 6171.180 | 4073.156 | 33.997 | 536.000 | 346.000 | 35.448 | 11850.430 | 7456.581 | 37.078 | 512.000 | 322.000 | 37.109 |

## 不同batch size（KV Reuse on/off）

| varying value | case | off status | on status | off e2e(us) | on e2e(us) | e2e improve(%) | off attn op(us) | on attn op(us) | attn op improve(%) | off attn movin(us) | on attn movin(us) | attn movin improve(%) | off attn movin(MiB) | on attn movin(MiB) | attn bytes improve(%) | off KV preload(us) | on KV preload(us) | KV preload improve(%) | off KV preload(MiB) | on KV preload(MiB) | KV bytes improve(%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | users_per_batch_1 | ok | ok | 1685.512 | 1119.691 | 33.570 | 358.772 | 227.601 | 36.561 | 792.541 | 529.362 | 33.207 | 70.000 | 46.250 | 33.929 | 1481.765 | 932.360 | 37.078 | 64.000 | 40.250 | 37.109 |
| 4 | users_per_batch_4 | ok | ok | 6350.895 | 4088.486 | 35.623 | 1433.811 | 909.551 | 36.564 | 3175.696 | 2126.439 | 33.040 | 280.000 | 185.000 | 33.929 | 5925.564 | 3728.212 | 37.083 | 256.000 | 161.000 | 37.109 |
| 8 | users_per_batch_8 | ok | ok | 12414.007 | 7955.717 | 35.913 | 1608.658 | 1084.263 | 32.598 | 6700.497 | 4602.208 | 31.315 | 560.000 | 370.000 | 33.929 | 11850.540 | 7459.089 | 37.057 | 512.000 | 322.000 | 37.109 |
| 16 | users_per_batch_16 | ok | ok | 24547.197 | 15691.942 | 36.074 | 1953.012 | 1428.956 | 26.833 | 13756.358 | 9562.298 | 30.488 | 1120.000 | 740.000 | 33.929 | 23706.500 | 14917.600 | 37.074 | 1024.000 | 644.000 | 37.109 |
| 32 | users_per_batch_32 | ok | ok | 48975.202 | 31258.127 | 36.176 | 3904.903 | 2857.329 | 26.827 | 27515.845 | 19135.006 | 30.458 | 2240.000 | 1480.000 | 33.929 | 47423.310 | 29838.060 | 37.081 | 2048.000 | 1288.000 | 37.109 |

## 不同candidates（1 macrobatch per batch）（KV Reuse on/off）

| varying value | case | off status | on status | off e2e(us) | on e2e(us) | e2e improve(%) | off attn op(us) | on attn op(us) | attn op improve(%) | off attn movin(us) | on attn movin(us) | attn movin improve(%) | off attn movin(MiB) | on attn movin(MiB) | attn bytes improve(%) | off KV preload(us) | on KV preload(us) | KV preload improve(%) | off KV preload(MiB) | on KV preload(MiB) | KV bytes improve(%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | candidates_512 | ok | ok | 6350.895 | 4088.486 | 35.623 | 1433.811 | 909.551 | 36.564 | 3175.696 | 2126.439 | 33.040 | 280.000 | 185.000 | 33.929 | 5925.564 | 3728.212 | 37.083 | 256.000 | 161.000 | 37.109 |
| 1024 | candidates_1024 | ok | ok | 6493.349 | 4231.648 | 34.831 | 1433.865 | 909.618 | 36.562 | 3524.806 | 2475.264 | 29.776 | 304.000 | 209.000 | 31.250 | 5924.894 | 3729.251 | 37.058 | 256.000 | 161.000 | 37.109 |
| 2048 | candidates_2048 | ok | ok | 6706.970 | 4625.171 | 31.039 | 1433.975 | 926.701 | 35.375 | 4238.154 | 3188.799 | 24.760 | 352.000 | 257.000 | 26.989 | 5926.312 | 3728.103 | 37.092 | 256.000 | 161.000 | 37.109 |
| 4096 | candidates_4096 | ok | ok | 9059.301 | 8784.509 | 3.033 | 1938.375 | 1937.754 | 0.032 | 5649.265 | 4600.010 | 18.573 | 448.000 | 353.000 | 21.205 | 5925.171 | 3727.850 | 37.085 | 256.000 | 161.000 | 37.109 |
| 8192 | candidates_8192 | ok | ok | 15465.970 | 15191.831 | 1.773 | 3281.923 | 3283.063 | -0.035 | 9883.898 | 8833.941 | 10.623 | 640.000 | 545.000 | 14.844 | 5924.243 | 3727.522 | 37.080 | 256.000 | 161.000 | 37.109 |


## Summary Observations

1. 在 KV length sweep 中，KV Reuse 对 KV preload bytes 的降低稳定为约 37.1%，端到端时延改善随历史长度增大从 31.9% 提升到 36.3%。这说明在长历史序列下，KV 相关搬运逐渐成为更显著的系统瓶颈。
2. 在 users-per-batch sweep 中，users/batch 从 1 增至 32 时，端到端改善保持在 33.6% 到 36.2% 区间，KV preload bytes 与 fused attention MOVIN bytes 均稳定下降。
3. 在 candidates/macro-batch sweep 中，候选数从 512 增至 8192 时，KV preload bytes 仍稳定降低约 37.1%，但端到端改善从 35.6% 降至 1.8%。原因是候选规模增大后，当前候选相关计算和非 KV 数据搬运占比上升，KV Reuse 的收益更多体现为 KV preload 与 fused attention MOVIN 的局部改善，而不一定完全转化为端到端加速。

