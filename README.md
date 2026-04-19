# GR\_simulator

[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

GR\_simulator 是基于 [ONNXim](https://ieeexplore.ieee.org/document/10726822) 扩展的NPU仿真器，专为 **生成式推荐模型（GR）** 工作负载（如 HSTU）定制。支持三种输入模式——ONNX算子图、语言模型trace、**算子trace**——可在多核NPU上进行周期精确的 DRAM/NoC/SSD 仿真。

***

## 目录

- [系统架构图](#系统架构)
- [GR算子支持列表](#目前支持的gr算子)
- [环境配置](#环境配置)
- [编译和执行](#编译和执行测试)
- [硬件配置](#硬件配置)
- [Trace输入格式](#trace输入格式)

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
│  DRAM ─── Ramulator 2.0 (HBM2)                                      │
│         ▼                                                           │
│  SSD ──── FEMU (8通道NAND)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

> ★ 标记为当前采用的 **Trace模式** 线路组件。

**架构图例：**

| 模块          | 实现方案                           | 说明                             |
| ----------- | ------------------------------ | ------------------------------ |
| **DRAM**    | Ramulator 2.0                  | 周期精确DRAM仿真器，支持DDR4、HBM2等       |
| **NoC**     | Booksim2 / Simple              | 片上网络，用于核间及核-存储通信               |
| **SSD**     | FEMU（Black-Box SSD）            | NAND闪存SSD模型，用于大张量卸载            |
| **计算核心**    | SystolicWS / SystolicOS        | 脉动阵列，支持权重驻留或输出驻留数据流            |
| **Trace前端** | TraceParser + TraceOpConverter | 解析PyTorch算子trace JSON并转换为NPU算子 |
| **映射**      | MappingTable + 回退映射            | 层次化分块，支持自动映射生成                 |

***

## GR算子支持

### 矩阵乘法类

| 算子        | 类名       | 数据流  | PyTorch Trace算子               | 功能描述         | 输入                               | 输出        | 精度                 | 动态形状 | 量化 |
| --------- | -------- | ---- | ----------------------------- | ------------ | -------------------------------- | --------- | ------------------ | ---- | -- |
| Gemm（线性层） | `GemmWS` | 权重驻留 | `aten::linear`, `aten::addmm` | 通用矩阵乘法（可选偏置） | A: \[N,K], B: \[K,M], bias: \[M] | C: \[N,M] | FP16 / BF16 / FP32 | ✅    | ❌  |
| Gemm（矩阵乘） | `GemmWS` | 权重驻留 | `aten::mm`                    | 矩阵乘法（无偏置）    | A: \[N,K], B: \[K,M]             | C: \[N,M] | FP16 / BF16 / FP32 | ✅    | ❌  |
| Gemm（OS）  | `GemmOS` | 输出驻留 | —                             | 输出驻留GEMM变体   | A: \[N,K], B: \[K,M]             | C: \[N,M] | FP16 / BF16 / FP32 | ✅    | ❌  |

### 卷积类

| 算子         | 类名       | 数据流  | PyTorch Trace算子 | 功能描述                 | 输入                                       | 输出                    | 精度                 | 动态形状 | 量化 |
| ---------- | -------- | ---- | --------------- | -------------------- | ---------------------------------------- | --------------------- | ------------------ | ---- | -- |
| Conv2D     | `ConvWS` | 权重驻留 | `aten::conv2d`  | 2D卷积，支持融合激活/BN/残差/池化 | input: \[N,H,W,Ci], weight: \[Co,Ci,R,S] | output: \[N,H',W',Co] | FP16 / BF16 / FP32 | ✅    | ❌  |
| Conv2D（OS） | `ConvOS` | 输出驻留 | —               | 输出驻留卷积变体             | input: \[N,H,W,Ci], weight: \[Co,Ci,R,S] | output: \[N,H',W',Co] | FP16 / BF16 / FP32 | ✅    | ❌  |

**Conv2D支持的属性：** `kernel_shape`、`strides`、`pads`、`dilations`、`group`，融合 `activation`（ReLU/SiLU）、融合 `batchnorm`、融合 `skip_connection`、融合 `pool`（max/avg）。

### 池化类

| 算子                | 类名                | PyTorch Trace算子             | 功能描述            | 输入         | 输出           | 精度                 | 动态形状 |
| ----------------- | ----------------- | --------------------------- | --------------- | ---------- | ------------ | ------------------ | ---- |
| MaxPool2D         | `MaxPool`         | `aten::max_pool2d`          | 空间维度最大池化        | \[N,H,W,C] | \[N,H',W',C] | FP16 / BF16 / FP32 | ✅    |
| AdaptiveAvgPool2D | `AdaptiveAvgPool` | `aten::adaptive_avg_pool2d` | 自适应平均池化至目标输出尺寸  | \[N,H,W,C] | \[N,H',W',C] | FP16 / BF16 / FP32 | ✅    |
| GlobalAvgPool     | `GlobalAvgPool`   | —                           | 全局平均池化（输出H=W=1） | \[N,H,W,C] | \[N,1,1,C]   | FP16 / BF16 / FP32 | ✅    |

### 归一化类

| 算子             | 类名               | PyTorch Trace算子    | 功能描述                       | 输入                                                        | 输出               | 精度                 | 动态形状 |
| -------------- | ---------------- | ------------------ | -------------------------- | --------------------------------------------------------- | ---------------- | ------------------ | ---- |
| SkipLayerNorm  | `SkipLayerNorm`  | `aten::layer_norm` | 融合Skip + LayerNorm（BERT风格） | input: \[B,S,D], skip: \[B,S,D], weight: \[D], bias: \[D] | output: \[B,S,D] | FP16 / BF16 / FP32 | ✅    |
| EmbedLayerNorm | `EmbedLayerNorm` | —                  | 融合Embedding + LayerNorm    | token\_ids, segment\_ids, position\_ids                   | output: \[B,S,D] | FP16 / BF16 / FP32 | ✅    |

### 激活类

| 算子       | 类名         | PyTorch Trace算子            | 功能描述                       | 输入                          | 输出               | 精度                 | 动态形状 |
| -------- | ---------- | -------------------------- | -------------------------- | --------------------------- | ---------------- | ------------------ | ---- |
| BiasGelu | `BiasGelu` | `aten::gelu`               | 融合偏置 + GELU激活              | input: \[B,S,D], bias: \[D] | output: \[B,S,D] | FP16 / BF16 / FP32 | ✅    |
| BiasAct  | `BiasAct`  | `aten::silu`, `aten::relu` | 融合偏置 + 通用激活（SiLU/ReLU/...） | input: \[B,S,D], bias: \[D] | output: \[B,S,D] | FP16 / BF16 / FP32 | ✅    |

### 注意力类

| 算子        | 类名          | PyTorch Trace算子 | 功能描述                      | 输入                                      | 输出               | 精度                 | 动态形状 |
| --------- | ----------- | --------------- | ------------------------- | --------------------------------------- | ---------------- | ------------------ | ---- |
| Attention | `Attention` | —               | 多头自注意力，支持KV-cache和融合QKV投影 | Q: \[B,S,D], K: \[B,S,Dk], V: \[B,S,Dv] | output: \[B,S,D] | FP16 / BF16 / FP32 | ✅    |

### 形状/工具类

| 算子            | 类名              | PyTorch Trace算子 | 功能描述             | 输入                                  | 输出          | 精度                 | 动态形状 |
| ------------- | --------------- | --------------- | ---------------- | ----------------------------------- | ----------- | ------------------ | ---- |
| Softmax       | `Softmax`       | `aten::softmax` | 沿指定维度Softmax     | \[B,S,D]                            | \[B,S,D]    | FP16 / BF16 / FP32 | ✅    |
| Flatten       | `Flatten`       | `aten::flatten` | 从start\_dim起展平维度 | \[N,...]                            | \[N,...]    | FP16 / BF16 / FP32 | ✅    |
| Concat        | `Concat`        | `aten::cat`     | 沿指定轴拼接张量         | tensors\[]                          | tensor      | FP16 / BF16 / FP32 | ✅    |
| KVCacheConcat | `KVCacheConcat` | —               | 将新KV token与缓存拼接  | new\_kv: \[B,S,D], cache: \[B,S',D] | \[B,S+S',D] | FP16 / BF16 / FP32 | ✅    |
| Dummy         | `Dummy`         | *（不支持的算子）*      | 不支持算子的占位符        | 任意                                  | 同输入形状       | 任意                 | ✅    |

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
# 将项目挂载到容器内，实现宿主机与容器代码实时同步
docker run -it \
  -v $(pwd):/workspace/GR_simulator \
  -w /workspace/GR_simulator \
  gr-simulator

# 在容器内安装相关依赖并构建目标
(docker) mkdir -p build && cd build
(docker) conan install .. --build=missing
(docker) cmake .. -DCMAKE_BUILD_TYPE=Release
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
conan install .. --build=missing
cmake .. -DCMAKE_BUILD_TYPE=Release
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
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd_fast.json \
  --models_list ./example/trace_models_list.json \
  --mode trace
```

**命令行参数：**

| 参数              | 说明                                                 | 默认值         |
| --------------- | -------------------------------------------------- | ----------- |
| `--config`      | 硬件配置JSON文件路径                                       | *必填*        |
| `--models_list` | 模型列表JSON文件路径                                       | *必填*        |
| `--mode`        | 输入模式：`default` / `language` / `trace`              | `default`   |
| `--log_level`   | 日志级别：`trace` / `debug` / `info` / `warn` / `error` | `info`      |
| `--trace_file`  | LLM请求trace文件（language模式）                           | `input.csv` |
| `--trace_path`  | 算子trace JSON路径（trace模式）                            | —           |

### 运行Trace测试

单个trace测试：

```bash
# 临时生成单模型列表
echo '{"models":[{"name":"test_gemm","trace_path":"example/trace_tests/test_gemm.json"}]}' > /tmp/test.json

./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd_fast.json \
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

### 单元测试

```bash
cd build
./bin/Simulator_test
```

### 结果解读

仿真成功时输出示例：

```
[info] Simulation time: 1.086484 seconds
[info] Total tile: 2, simulated tile per seconds(TPS): 300887.618474
```

| 指标                  | 说明               |
| ------------------- | ---------------- |
| **Simulation time** | 仿真运行的实际耗时（秒）     |
| **Total tile**      | 所有核心执行的计算tile总数  |
| **TPS**             | 每秒tile数——仿真吞吐量指标 |

***

## 硬件配置

配置文件位于 `configs/` 目录，主要参数：

```json
{
  "num_cores": 4,
  "core_freq": 1000,
  "core_config": {
    "core_0": {
      "core_type": "systolic_ws",
      "core_width": 128,
      "core_height": 128,
      "spad_size": 16384,
      "accum_spad_size": 4096,
      "sram_width": 32,
      "vector_process_bit": 16384,
      "add_latency": 1,
      "mul_latency": 1,
      "gelu_latency": 1,
      "exp_latency": 1
    }
  },
  "dram_type": "ramulator2",
  "dram_freq": 1200,
  "dram_channels": 16,
  "dram_config_path": "../configs/ramulator2_configs/HBM2.yaml",
  "icnt_type": "simple",
  "icnt_latency": 1,
  "ssd": {
    "enabled": true,
    "place_threshold_bytes": 1048576,
    "address_base": "0x800000000",
    "nchs": 8,
    "luns_per_ch": 8
  },
  "precision": 2,
  "layout": "NHWC",
  "scheduler": "simple"
}
```

**可用配置：**

| 配置文件                                                                    | 核心阵列          | DRAM              | NoC      | SSD    |
| ----------------------------------------------------------------------- | ------------- | ----------------- | -------- | ------ |
| `systolic_ws_128x128_c4_simple_noc_tpuv4.json`                          | 4× WS 128×128 | Ramulator 1（简单模式） | Simple   | ❌      |
| `systolic_ws_128x128_c4_booksim2_tpuv4.json`                            | 4× WS 128×128 | Ramulator 1（简单模式） | Booksim2 | ❌      |
| `systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2.json`          | 4× WS 128×128 | Ramulator 2（HBM2） | Simple   | ❌      |
| `systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2_ssd_fast.json` | 4× WS 128×128 | Ramulator 2（HBM2） | Simple   | ✅ FEMU |

***

## Trace输入格式

算子trace前端接受符合PyTorch算子trace格式的JSON文件：

```json
{
  "metadata": {
    "format_version": "1.0",
    "model_name": "my_model",
    "layout": "NHWC"
  },
  "operators": [
    {
      "id": 0,
      "name": "aten::linear",
      "inputs": [
        { "name": "input", "shape": [128, 512], "dtype": "float16" },
        { "name": "weight", "shape": [512, 256], "dtype": "float16", "is_weight": true },
        { "name": "bias", "shape": [256], "dtype": "float16", "is_weight": true }
      ],
      "outputs": [
        { "name": "output", "shape": [128, 256], "dtype": "float16" }
      ],
      "attrs": {
        "has_bias": "1"
      }
    }
  ]
}
```

**支持的PyTorch算子名称映射：**

| PyTorch算子                   | NPU算子           | 说明                |
| --------------------------- | --------------- | ----------------- |
| `aten::linear`              | Gemm            | 线性层（A×B^T + bias） |
| `aten::addmm`               | Gemm            | 带偏置矩阵乘法           |
| `aten::mm`                  | Gemm            | 矩阵乘法（无偏置）         |
| `aten::conv2d`              | Conv            | 2D卷积              |
| `aten::max_pool2d`          | MaxPool         | 最大池化              |
| `aten::adaptive_avg_pool2d` | AdaptiveAvgPool | 自适应平均池化           |
| `aten::avg_pool2d`          | AdaptiveAvgPool | 平均池化              |
| `aten::flatten`             | Flatten         | 张量展平              |
| `aten::layer_norm`          | SkipLayerNorm   | 层归一化（skip自动合成）    |
| `aten::gelu`                | BiasGelu        | GELU激活（偏置自动合成）    |
| `aten::silu`                | BiasAct         | SiLU激活            |
| `aten::softmax`             | Softmax         | Softmax           |
| *（其他）*                      | Dummy           | 不支持的算子→占位符        |



