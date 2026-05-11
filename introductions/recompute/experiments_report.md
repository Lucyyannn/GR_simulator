# KV Cache Recompute Report

参数：

```text
layers=4
hidden=256
kv_len=8192
num_users=1
users_per_batch=1
candidates_per_user=128
macro_batch_size=128
vocab=262144
attention_modeling=fused
```
### （1）SSD->HBM
结果：

| history_recompute_len | cached_kv_len | simulation time/us | 相对 baseline | wall-clock/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8192 | 22767.736 | + 0.00% | 48.518 |
| 512 | 7680 | 22588.547 | - 0.79% | 50.540 |
| 1024 | 7168 | 22730.229 | - 0.16% | 52.042 |
| 2048 | 6144 | 22978.088 | + 0.92% | 54.646 |
| 4096 | 4096 | 23374.991 | + 2.67% | 63.553 |

breakdown 汇总：

| history_recompute_len | stage preload/us | pre_attention preload/us | KV preload/us | post weights preload/us | attention 前 compute/us | 与 KV 重叠的 attention 前 compute/us | attention compute/us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 22760.030 | 1800.008 | 20640.000 | 20960.000 | 39.588 | 39.603 | 212.524 |
| 512 | 22560.010 | 2880.006 | 19360.030 | 19680.000 | 148.562 | 148.590 | 238.069 |
| 1024 | 22680.000 | 4280.006 | 18080.070 | 18400.000 | 281.557 | 281.590 | 270.105 |
| 2048 | 22880.010 | 7040.006 | 15520.110 | 15840.000 | 523.597 | 523.620 | 330.357 |
| 4096 | 23160.000 | 12440.006 | 10400.030 | 10720.000 | 1007.714 | 1007.600 | 435.120 |

### （2）DDR->HBM


| history_recompute_len | pre_attention preload/us | KV preload/us | post weights preload/us | simulation time/us | 相对baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 90.800 | 1480.988 | 1655.517 | 1763.050 | 0.000% |
| 512 | 102.176 | 1388.123 | 1562.598 | 1708.700 | -3.083% |
| 1024 | 113.771 | 1296.405 | 1470.953 | 1657.720 | -5.974% |
| 2048 | 137.940 | 1110.786 | 1285.310 | 1553.690 | -11.875% |
| 4096 | 186.252 | 742.026 | 801.577 | 2482.590 | +40.812% |

