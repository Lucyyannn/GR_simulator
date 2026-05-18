# task4 扩展性实验报告

配置：HSTU-small，kv_len=4096，candidates=128，batch_size=1，embedding 初始在 SSD，weights 在 HBM。
KV Reuse ratio=0.4360；Recompute 在 hot/cold 间复用同一组物理结果。

## 1. 910A/910B 配置对比

| variable | user | scheme | recompute_ratio | simulation time (ms) | NPU util (%) | HBM util (%) | DDR util (%) | SSD util (%) | memory util of interest (%) | result dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 910A | hot | Recompute | 100.0000% | 9.927860 | 2.37816 | 22.7454 | 0 | 4.02901 | 0 | `results/scalability/910A_hot/Recompute` |
| 910A | hot | FullCache | 0.0000% | 0.684262 | 1.98951 | 36.2275 | 17.954 | 11.687 | 17.954 | `results/scalability/910A_hot/FullCache` |
| 910A | hot | W_AR | 0.0000% | 0.401874 | 2.12686 | 41.593 | 15.925 | 19.897 | 15.925 | `results/scalability/910A_hot/W_AR` |
| 910A | hot | W_IR | 9.4727% | 1.344020 | 2.64282 | 28.927 | 8.71001 | 7.114 | 8.71001 | `results/scalability/910A_hot/W_IR` |
| 910A | hot | W_both | 0.0000% | 0.401874 | 2.12686 | 41.593 | 15.925 | 19.897 | 15.925 | `results/scalability/910A_hot/W_both` |
| 910A | cold | Recompute | 100.0000% | 9.927860 | 2.37816 | 22.7454 | 0 | 4.02901 | 4.02901 | `results/scalability/910A_cold/Recompute` |
| 910A | cold | FullCache | 0.0000% | 2.885130 | 0.471948 | 8.59448 | 0 | 91.582 | 91.582 | `results/scalability/910A_cold/FullCache` |
| 910A | cold | W_AR | 0.0000% | 1.573960 | 0.543243 | 10.6237 | 0 | 90.0481 | 90.0481 | `results/scalability/910A_cold/W_AR` |
| 910A | cold | W_IR | 90.0391% | 6.223140 | 2.43507 | 20.9487 | 0 | 26.2623 | 26.2623 | `results/scalability/910A_cold/W_IR` |
| 910A | cold | W_both | 38.0371% | 2.230900 | 2.24939 | 27.6163 | 0 | 44.3624 | 44.3624 | `results/scalability/910A_cold/W_both` |
| 910B | hot | Recompute | 100.0000% | 8.415900 | 2.19887 | 26.2679 | 0 | 4.75285 | 0 | `results/scalability/910B_hot/Recompute` |
| 910B | hot | FullCache | 0.0000% | 0.674084 | 1.63816 | 35.6612 | 18.2268 | 11.8646 | 18.2268 | `results/scalability/910B_hot/FullCache` |
| 910B | hot | W_AR | 0.0000% | 0.369159 | 1.88259 | 43.8519 | 17.3349 | 21.6585 | 17.3349 | `results/scalability/910B_hot/W_AR` |
| 910B | hot | W_IR | 9.4727% | 1.125010 | 2.38011 | 34.6519 | 10.4014 | 8.49543 | 10.4014 | `results/scalability/910B_hot/W_IR` |
| 910B | hot | W_both | 0.0000% | 0.369159 | 1.88259 | 43.8519 | 17.3349 | 21.6585 | 17.3349 | `results/scalability/910B_hot/W_both` |
| 910B | cold | Recompute | 100.0000% | 8.415900 | 2.19887 | 26.2679 | 0 | 4.75285 | 4.75285 | `results/scalability/910B_cold/Recompute` |
| 910B | cold | FullCache | 0.0000% | 2.874970 | 0.384179 | 8.3632 | 0 | 91.9078 | 91.9078 | `results/scalability/910B_cold/FullCache` |
| 910B | cold | W_AR | 0.0000% | 1.567100 | 0.443705 | 10.3346 | 0 | 90.4414 | 90.4414 | `results/scalability/910B_cold/W_AR` |
| 910B | cold | W_IR | 90.0391% | 5.224070 | 2.27211 | 23.7245 | 0 | 31.2848 | 31.2848 | `results/scalability/910B_cold/W_IR` |
| 910B | cold | W_both | 38.0371% | 1.890370 | 1.98576 | 31.1839 | 0 | 52.3345 | 52.3345 | `results/scalability/910B_cold/W_both` |

## 2. baseline core 数量扩展

| variable | user | scheme | recompute_ratio | simulation time (ms) | NPU util (%) | HBM util (%) | DDR util (%) | SSD util (%) | memory util of interest (%) | result dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cores1 | hot | Recompute | 100.0000% | 4.262570 | 20.4356 | 2.23526 | 0 | 9.37398 | 0 | `results/scalability/cores1_hot/Recompute` |
| cores1 | hot | FullCache | 0.0000% | 0.813879 | 9.31367 | 1.712 | 10.0567 | 9.821 | 10.0567 | `results/scalability/cores1_hot/FullCache` |
| cores1 | hot | W_AR | 0.0000% | 0.432490 | 12.1934 | 2.06013 | 9.85226 | 18.4672 | 9.85226 | `results/scalability/cores1_hot/W_AR` |
| cores1 | hot | W_IR | 9.4727% | 0.802049 | 18.417 | 2.52145 | 9.70993 | 11.8979 | 9.70993 | `results/scalability/cores1_hot/W_IR` |
| cores1 | hot | W_both | 0.0000% | 0.432490 | 12.1934 | 2.06013 | 9.85226 | 18.4672 | 9.85226 | `results/scalability/cores1_hot/W_both` |
| cores1 | cold | Recompute | 100.0000% | 4.262570 | 20.4356 | 2.23526 | 0 | 9.37398 | 9.37398 | `results/scalability/cores1_cold/Recompute` |
| cores1 | cold | FullCache | 0.0000% | 2.869600 | 2.64341 | 0.485859 | 0 | 92.0636 | 92.0636 | `results/scalability/cores1_cold/FullCache` |
| cores1 | cold | W_AR | 0.0000% | 1.563030 | 3.377 | 0.570717 | 0 | 90.6483 | 90.6483 | `results/scalability/cores1_cold/W_AR` |
| cores1 | cold | W_IR | 90.0391% | 2.480950 | 20.6869 | 2.29838 | 0 | 65.6215 | 65.6215 | `results/scalability/cores1_cold/W_IR` |
| cores1 | cold | W_both | 38.0371% | 1.257780 | 16.0446 | 2.17473 | 0 | 78.7219 | 78.7219 | `results/scalability/cores1_cold/W_both` |
| cores2 | hot | Recompute | 100.0000% | 2.830340 | 15.3802 | 3.36523 | 0 | 14.1097 | 0 | `results/scalability/cores2_hot/Recompute` |
| cores2 | hot | FullCache | 0.0000% | 0.789088 | 4.79803 | 1.77611 | 10.3724 | 10.1293 | 10.3724 | `results/scalability/cores2_hot/FullCache` |
| cores2 | hot | W_AR | 0.0000% | 0.418258 | 6.29442 | 2.14966 | 10.1869 | 19.0945 | 10.1869 | `results/scalability/cores2_hot/W_AR` |
| cores2 | hot | W_IR | 9.4727% | 0.775851 | 9.75837 | 2.655 | 10.0407 | 12.3032 | 10.0407 | `results/scalability/cores2_hot/W_IR` |
| cores2 | hot | W_both | 0.0000% | 0.418258 | 6.29442 | 2.14966 | 10.1869 | 19.0945 | 10.1869 | `results/scalability/cores2_hot/W_both` |
| cores2 | cold | Recompute | 100.0000% | 2.830340 | 15.3802 | 3.36523 | 0 | 14.1097 | 14.1097 | `results/scalability/cores2_cold/Recompute` |
| cores2 | cold | FullCache | 0.0000% | 2.844800 | 1.33237 | 0.492971 | 0 | 92.8657 | 92.8657 | `results/scalability/cores2_cold/FullCache` |
| cores2 | cold | W_AR | 0.0000% | 1.548800 | 1.70206 | 0.581243 | 0 | 91.4807 | 91.4807 | `results/scalability/cores2_cold/W_AR` |
| cores2 | cold | W_IR | 90.0391% | 1.852980 | 13.8688 | 3.08285 | 0 | 87.9874 | 87.9874 | `results/scalability/cores2_cold/W_IR` |
| cores2 | cold | W_both | 38.0371% | 1.197680 | 8.74873 | 2.34887 | 0 | 82.7785 | 82.7785 | `results/scalability/cores2_cold/W_both` |
| cores4 | hot | Recompute | 100.0000% | 2.153480 | 10.0882 | 4.4227 | 0 | 18.5355 | 0 | `results/scalability/cores4_hot/Recompute` |
| cores4 | hot | FullCache | 0.0000% | 0.774155 | 2.48016 | 1.83147 | 10.5723 | 10.3245 | 10.5723 | `results/scalability/cores4_hot/FullCache` |
| cores4 | hot | W_AR | 0.0000% | 0.408592 | 3.28668 | 2.24045 | 10.4275 | 19.5455 | 10.4275 | `results/scalability/cores4_hot/W_AR` |
| cores4 | hot | W_IR | 9.4727% | 0.753064 | 5.05668 | 2.7898 | 10.344 | 12.6748 | 10.344 | `results/scalability/cores4_hot/W_IR` |
| cores4 | hot | W_both | 0.0000% | 0.408592 | 3.28668 | 2.24045 | 10.4275 | 19.5455 | 10.4275 | `results/scalability/cores4_hot/W_both` |
| cores4 | cold | Recompute | 100.0000% | 2.153480 | 10.0882 | 4.4227 | 0 | 18.5355 | 18.5355 | `results/scalability/cores4_cold/Recompute` |
| cores4 | cold | FullCache | 0.0000% | 2.829870 | 0.678933 | 0.501358 | 0 | 93.3557 | 93.3557 | `results/scalability/cores4_cold/FullCache` |
| cores4 | cold | W_AR | 0.0000% | 1.539130 | 0.873623 | 0.595529 | 0 | 92.055 | 92.055 | `results/scalability/cores4_cold/W_AR` |
| cores4 | cold | W_IR | 90.0391% | 1.838000 | 7.21845 | 3.20364 | 0 | 88.8209 | 88.8209 | `results/scalability/cores4_cold/W_IR` |
| cores4 | cold | W_both | 38.0371% | 1.189680 | 4.68008 | 2.49102 | 0 | 83.3813 | 83.3813 | `results/scalability/cores4_cold/W_both` |
