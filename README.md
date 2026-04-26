# GR\_simulator

[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

GR\_simulator 是基于 [ONNXim](https://ieeexplore.ieee.org/document/10726822) 扩展的NPU仿真器，可支持 **生成式推荐模型（GR）** 工作负载（如 HSTU）的多种算子。当前支持三种运行模式：ONNX算子图、语言模型trace、**算子trace**，可在多核NPU上进行周期精确的 DRAM/NoC/SSD 仿真。

***

## 目录

- [系统架构图](#系统架构图)
- [环境配置](#环境配置)
- [编译和执行](#编译和执行测试)


***

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│  输入前端                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ ONNX算子图   │  │ 语言模型     │  │ ★ 算子Trace (PyTorch)   │    │
│  │   (.onnx)    │  │ (LLM .csv)   │  │     (.json)             │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
└─────────┼─────────────────┼───────────────────────┼────────────────┘
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  解析层                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ ONNX解析器   │  │ 语言模型     │  │ ★ Trace解析器            │  │
│  │  (protobuf)  │  │ 解析器       │  │  TraceParser             │  │
│  │              │  │(LanguageModel)│  │  → TraceOpConverter      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
└─────────┼─────────────────┼───────────────────────┼────────────────┘
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  模型层                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Model        │  │ LanguageModel│  │ ★ TraceModel             │  │
│  │(ONNX图→算子) │  │(LLM算子+调度)│  │  (trace→算子)            │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
└─────────┼─────────────────┼───────────────────────┼────────────────┘
          └─────────────────┼───────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  算子层                                                             │
│  GemmWS/OS │ ConvWS/OS │ Attention │ SkipLayerNorm │ BiasGelu/Act │
│  MaxPool   │ AdaptiveAvgPool │ GlobalAvgPool │ Softmax │ Flatten  │
│  Concat    │ KVCacheConcat   │ EmbedLayerNorm │ Dummy              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  映射与分块    MappingTable → 回退映射 (gemm_mapping / conv_mapping) │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  仿真核心                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 调度器 (simple / spatial_split / time_multiplex)            │   │
│  └──────────┬──────────┬──────────┬──────────┬─────────────────┘   │
│             ▼          ▼          ▼          ▼                     │
│  ┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐     │
│  │  Core 0     ││  Core 1     ││  Core 2     ││  Core 3     │     │
│  │ SystolicWS  ││ SystolicWS  ││ SystolicWS  ││ SystolicWS  │     │
│  │  128×128    ││  128×128    ││  128×128    ││  128×128    │     │
│  └──────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬──────┘     │
└─────────┼──────────────┼──────────────┼──────────────┼────────────┘
          └──────────────┼──────────────┼──────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  存储层次                                                           │
│  SRAM/Scratchpad (每核16MB)                                         │
│         ▼                                                           │
│  互连网络 (Simple互连 / Booksim2 NoC)                                │
│         ▼                                                           │
│  HBM ———— Ramulator 2.0 (HBM2)                                      │
│         ▼                                                           │
│  DDR ─── Ramulator 2.0  (DDR4)                                      │
│         ▼                                                           │
│  SSD ──── FEMU (8通道NAND)                                          │
└─────────────────────────────────────────────────────────────────────┘
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
docker run -it \
  -v $(pwd):/workspace/GR_simulator \
  -w /workspace/GR_simulator \
  gr-simulator

# 在容器内安装相关依赖并构建目标
(docker) mkdir -p build && cd build
(docker) conan install .. --build=missing
(docker) cmake 
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


***

## 编译和执行测试

### 编译

```bash
cd /path/to/GR_simulator
mkdir -p build && cd build
conan install .. 
cmake .. 
make -j$(nproc)
```

### 运行仿真

#### 模式1：ONNX算子图（默认）

```bash
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json \
  --models_list ./example/models_list.json
```

#### 模式2：语言模型（自回归LLM）

```bash
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json \
  --models_list ./example/language_models.json \
  --mode language
```

#### 模式3：算子Trace（PyTorch trace JSON）

```bash
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd.json \
  --models_list ./example/trace_models_list.json \
  --mode trace
```

#### 模式4：存储微基准（mem_bench）

```bash
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd.json \
  --mode mem_bench \
  --bench_config ./configs/mem_benchmark_default.json \
  --bench_output_dir ./results/hbmddrssd/mem_bench
```

**命令行参数：**

| 参数              | 说明                                                 | 默认值         |
| --------------- | -------------------------------------------------- | ----------- |
| `--config`      | 硬件配置JSON文件路径                                       | *必填*        |
| `--models_list` | 模型列表JSON文件路径；`mem_bench` 模式下不需要                    | *模型模式必填*   |
| `--mode`        | 运行模式：`default` / `language` / `trace` / `mem_bench` | `default`   |
| `--log_level`   | 日志级别：`trace` / `debug` / `info` / `warn` / `error` | `info`      |
| `--trace_file`  | LLM请求trace文件（language模式）                           | `input.csv` |
| `--trace_path`  | 算子trace JSON路径（trace模式）                            | —           |
| `--bench_config` | `mem_bench` 配置文件路径                                | —           |
| `--bench_output_dir` | `mem_bench` 输出目录                             | `results/hbmddrssd/mem_bench` |

### 运行Trace测试

单个trace测试：

```bash
# 临时生成单模型列表
echo '{"models":[{"name":"test_gemm","trace_path":"example/trace_tests/test_gemm.json"}]}' > /tmp/test.json

./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd.json \
  --mode trace \
  --models_list /tmp/test.json \
  --log_level info
```

`example/trace_tests/` 中可用的测试用例：

| 测试文件                         | 算子                          | 输入形状                          | 说明             |
| ---------------------------- | --------------------------- | ----------------------------- | -------------- |
| `test_gemm.json`             | Gemm（aten::linear）          | \[128,512] × \[512,256]       | 带偏置的线性层        |
| `test_matmul.json`           | Gemm（aten::mm）              | \[128,512] × \[512,512]       | 矩阵乘法           |
| `test_conv2d.json`           | Conv（aten::conv2d）          | \[128,8,8,64]，卷积核3×3          | 2D卷积           |
| `test_maxpool.json`          | MaxPool（aten::max\_pool2d）  | \[128,16,16,64]，核2×2          | 最大池化           |
| `test_adaptive_avgpool.json` | AdaptiveAvgPool             | \[128,8,8,64] → \[128,1,1,64] | 自适应平均池化        |
| `test_flatten.json`          | Flatten（aten::flatten）      | \[128,1,1,64] → \[128,64]     | 张量展平           |
| `test_softmax.json`          | Softmax（aten::softmax）      | \[128,256]                    | Softmax（dim=1） |
| `test_layernorm_gelu.json`   | SkipLayerNorm + BiasGelu    | \[128,512]                    | 融合LN + GELU流水线 |
| `test_pipeline.json`         | Linear → LN → GELU → Linear | \[128,512]                    | 多算子流水线         |




