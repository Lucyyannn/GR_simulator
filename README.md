# GR\_simulator

[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

GR\_simulator是基于 [ONNXim](https://ieeexplore.ieee.org/document/10726822) 扩展得到的NPU仿真系统，可支持 **生成式推荐（GR）模型** 工作负载（如 HSTU）的推理仿真。系统具备面向 UB 的分层存储架构，由 UB 连接 NPU、DRAM 与 SSD，并统一协调不同存储层级中的用户数据进入 NPU 侧执行。

***

## 目录

- [系统架构图](#系统架构图)
- [代码结构说明](#代码结构说明)
- [环境配置](#环境配置)
- [编译运行](#编译运行)
- [实验复现说明](#实验复现说明)


***

## 系统架构图

![GR simulator system architecture](img/arch.png)

GR\_simulator 以 trace 作为主要输入，Trace Frontend 负责解析算子与访存描述，Pipe Scheduler 将计算与预取请求组织成流水执行。NPU 侧基于 ONNXim 的 tile scheduler 与多核执行模型，存储侧由 Storage Controller 统一管理 HBM、DDR 和 SSD 请求，其中 HBM/DDR 使用 Ramulator2 建模，SSD 使用 FEMU BlackBoxSSD 建模。NPU Core 通过 NoC 与多级存储连接。


***

## 代码结构说明

本节介绍仿真器代码中的核心目录和文件。

```text
GR_simulator/
├── CMakeLists.txt              # C++ 仿真器构建入口
├── Dockerfile                  # Docker 运行环境
├── conanfile.txt               # C++ 依赖声明
├── configs/                    # 仿真器配置文件
│   ├── 910A.json
│   ├── 910B.json
│   ├── 910C.json
│   ├── booksim2_configs/       # Booksim2 NoC 配置
│   └── ramulator2_configs/     # Ramulator2 DDR/HBM 配置
├── src/                        # 仿真器主体源码
│   ├── main.cc                 # 程序入口
│   ├── Simulator.*             # 仿真主循环与组件协调
│   ├── Core.* / Systolic*.cc   # NPU core 与 systolic array 建模
│   ├── Dram.* / Hbm.* / Ssd.*  # DDR、HBM、SSD 接口封装
│   ├── Interconnect.*          # NoC/互连建模
│   ├── frontend/trace/         # trace 解析
│   ├── memory/                 # storage controller建模
│   ├── operations/             # Gemm、Embedding、HSTUAttention 等算子模型
│   ├── scheduler/              # 算子/语言模型调度器
│   ├── models/                 # 语言模型 workload 支持
│   └── benchmark/              # 单项内存带宽 benchmark
├── scripts/                    # trace 生成、校准、实验运行和结果处理脚本
│   ├── run_hstu.sh             # HSTU 实验的通用入口
│   ├── generate_hstu_baseline_trace.py
│   ├── recompute_ratio_cost_model_new.py
│   ├── recompute_ratio_calibration.json
│   ├── run_ActionReuse.sh
│   ├── run_ItemRecompute.sh
│   ├── run_OoO_pipeline_Ablation.sh
│   ├── run_SpeedupComparison.sh
│   └── run_scalability.sh
├── docs/                       # 公式、流程与可复现性说明
├── introductions/              # 论文/报告用实验结果、表格和图
├── example/                    # models list 和 trace 示例入口
├── extern/                     # protobuf、Ramulator2 等第三方组件
└── img/                        # README 和文档图片资源
```

***

## 环境配置

### 1. Docker方式（推荐）

使用项目提供的Dockerfile构建镜像：

```bash
# 克隆仓库
git clone https://github.com/Lucyyannn/GR_simulator.git
cd GR_simulator
git submodule update --recursive --init

# 构建Docker镜像
docker build . -t gr-simulator
```

启动容器并挂载项目目录：

```bash
docker run -it --name gr-simulator-mini \
  -v $(pwd):/workspace/GR_simulator \
  -w /workspace/GR_simulator \
  onnxim bash

# 在容器内安装相关依赖并构建目标
(docker) mkdir -p build && cd build
(docker) conan install .. --build=missing
(docker) cmake ..
(docker) make -j$(nproc)
```

### 2. 手动安装方式

**系统要求：**

| 依赖项       | 最低版本             |
| --------- | ---------------- |
| 操作系统      | Ubuntu 20.04（推荐） |
| GCC / G++ | >= 10.5.0        |
| CMake     | >= 3.22.1        |
| Python    | >= 3.8           |
| Conan     | 1.57.0           |


**Conan依赖**：

| 包名                 | 版本     |
| ------------------ | ------ |
| boost              | 1.79.0 |
| robin-hood-hashing | 3.11.5 |
| spdlog             | 1.11.0 |
| nlohmann\_json     | 3.11.2 |

### 3.环境验证

运行简单的算子来验证环境配置成功：
```bash
bash scripts/run_embedding.sh # 一个简单的embedding算子测试
bash scripts/run_gemm.sh      # 一个简单的gemm算子测试
```
运行脚本后，终端输出日志信息，并在最后报告仿真时间。

***

## 编译运行

### 1.编译

```bash
cd /path/to/GR_simulator
mkdir -p build && cd build
conan install .. 
cmake .. 
make -j$(nproc)
```

### 2.运行 HSTU 模型推理

仿真器使用`run_hstu.sh`脚本作为HSTU推理仿真的主入口，脚本通过参数指定模型层数、请求数量、优化策略等配置，详细说明可见 `introductions/run_experiments.md`。


## 实验复现说明

### 1. Action KV Cache Reuse 参数网格实验

```bash
bash scripts/run_ActionReuse.sh
```

该脚本用于复现 action KV reuse 的参数网格实验。脚本固定 HSTU-small、`kv_len=4096`、cold/SSD 场景，遍历 `window_size={64,128,256,512}` 与 `top_k={1,2,3,4,5}`，并为每组参数设置对应的 `kv_reuse_ratio`，该值取自HSTU模型侧[recsys](https://github.com/cry-daniel/recsys)在各参数配置下的实际复用率。默认输出目录为 `results/ActionReuse`。

### 2. Item Recompute 参数实验

```bash
bash scripts/run_ItemRecompute.sh
```

该脚本用于复现 Item Recompute 比例与索引模式实验。它会运行 `continuous` 与 `random` 两种历史 embedding 索引模式，覆盖 `0%/20%/40%/60%/80%/100%` 以及由 `scripts/recompute_ratio_cost_model_new.py` 估算出的 `optimal` recompute 长度，对比不同的Item Recompute设置的效果。默认输出目录为 `results/ItemRecompute`。

### 3. Out-of-Order Pipeline 消融实验

```bash
bash scripts/run_OoO_pipeline_Ablation.sh
```

该脚本用于复现 Out-of-Order pipeline 的消融实验，比较开启和关闭 out-of-order pipeline 时，action reuse 与 item recompute 组合后的表现。脚本默认遍历 `cold/hot`、`batch_size={1,4,8}`、`kv_len={4096,8192,16384}`，并调用 recompute ratio 估算脚本为每个 case 选择 `history_recompute_len`。默认输出目录为 `results/OoO_pipeline_ablation`。


### 4. Speedup Comparison 实验

```bash
bash scripts/run_SpeedupComparison.sh
```

该脚本用于复现各方法加速比对比实验，覆盖 `Recompute`、`FullCache`、`W_AR`、`W_IR`、`W_both` 五类方法，并遍历 HSTU-small/middle/large、`kv_len={4096,8192,16384}`、`batch_size={1,4,8}`、hot/cold 用户。W_IR 和 W_both 会自动调用 recompute ratio 估算脚本生成每个 case 的 recompute 长度。默认输出目录为 `results/SpeedupComparison`。

### 5. Scalability 实验

```bash
bash scripts/run_scalability.sh \
  --result-root results/hstu_scalability_$(date +%Y%m%d_%H%M%S) \
  --max-concurrent 45 \
  --docker-container gr-simulator \
```

该脚本用于复现 HSTU 模型规模可扩展性实验，默认在单 NPU、单用户设置下运行 HSTU-small/middle/large，并覆盖 `910A/910B/910C`三种配置、Cold/Hot用户以及 `Full_Cache`、`Full_Recompute`、`w_AR`、`w_IR`、`w_both` 五类方法。脚本会先执行内存带宽校准，并使用代价模型的估算脚本为 `w_IR` 和 `w_both` 自动估算 `history_recompute_len`，然后执行HSTU模型推理。


