#include "TraceOpConverter.h"

#include <sstream>
#include <spdlog/spdlog.h>

namespace trace_frontend {

std::string TraceOpConverter::shape_to_str(const std::vector<uint32_t>& shape) {
  if (shape.empty()) return "";
  std::string result = std::to_string(shape[0]);
  for (size_t i = 1; i < shape.size(); i++)
    result += "," + std::to_string(shape[i]);
  return result;
}

std::vector<uint32_t> TraceOpConverter::parse_dims(const std::string& s) {
  std::vector<uint32_t> dims;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    if (!token.empty())
      dims.push_back(static_cast<uint32_t>(std::stoul(token)));
  }
  return dims;
}

ConvertedOp TraceOpConverter::convert(const OpEntry& entry) {
  const auto& name = entry.name;

  if (name == "aten::conv2d" || name == "conv2d")
    return convert_conv2d(entry);
  if (name == "aten::linear" || name == "linear")
    return convert_linear(entry);
  if (name == "aten::mm" || name == "aten::matmul" || name == "mm" || name == "matmul")
    return convert_matmul(entry);
  if (name == "aten::addmm" || name == "addmm")
    return convert_addmm(entry);
  if (name == "aten::max_pool2d" || name == "max_pool2d")
    return convert_max_pool2d(entry);
  if (name == "aten::adaptive_avg_pool2d" || name == "adaptive_avg_pool2d")
    return convert_adaptive_avg_pool2d(entry);
  if (name == "aten::avg_pool2d" || name == "avg_pool2d")
    return convert_avg_pool2d(entry);
  if (name == "aten::flatten" || name == "flatten")
    return convert_flatten(entry);
  if (name == "aten::layer_norm" || name == "layer_norm")
    return convert_layer_norm(entry);
  if (name == "aten::gelu" || name == "gelu")
    return convert_gelu(entry);
  if (name == "aten::silu" || name == "silu")
    return convert_silu(entry);
  if (name == "aten::softmax" || name == "softmax")
    return convert_softmax(entry);

  spdlog::debug("[TraceOpConverter] Unknown op '{}' -> Dummy", name);
  return convert_dummy(entry);
}

ConvertedOp TraceOpConverter::convert_conv2d(const OpEntry& entry) {
  ConvertedOp conv;
  conv.optype = "Conv";
  conv.attrs = entry.attrs;

  if (!entry.inputs.empty())
    conv.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2)
    conv.attrs["weight_shape"] = shape_to_str(entry.inputs[1].shape);
  if (!entry.outputs.empty())
    conv.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (conv.attrs.find("kernel_shape") == conv.attrs.end() && entry.inputs.size() >= 2) {
    auto& ws = entry.inputs[1].shape;
    if (ws.size() >= 4)
      conv.attrs["kernel_shape"] = std::to_string(ws[2]) + "," + std::to_string(ws[3]);
  }

  if (conv.attrs.count("stride") && !conv.attrs.count("strides"))
    conv.attrs["strides"] = conv.attrs["stride"];
  if (conv.attrs.find("strides") == conv.attrs.end())
    conv.attrs["strides"] = "1,1";

  if (conv.attrs.count("padding") && !conv.attrs.count("pads")) {
    auto pad_str = conv.attrs["padding"];
    auto dims = parse_dims(pad_str);
    if (dims.size() == 1)
      conv.attrs["pads"] = std::to_string(dims[0]) + "," + std::to_string(dims[0]) + "," + std::to_string(dims[0]) + "," + std::to_string(dims[0]);
    else if (dims.size() == 2)
      conv.attrs["pads"] = std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[0]) + "," + std::to_string(dims[1]);
    else
      conv.attrs["pads"] = pad_str;
  }
  if (conv.attrs.find("pads") == conv.attrs.end())
    conv.attrs["pads"] = "0,0,0,0";

  if (conv.attrs.count("dilation") && !conv.attrs.count("dilations"))
    conv.attrs["dilations"] = conv.attrs["dilation"];
  if (conv.attrs.find("dilations") == conv.attrs.end())
    conv.attrs["dilations"] = "1,1";

  if (conv.attrs.count("groups") && !conv.attrs.count("group"))
    conv.attrs["group"] = conv.attrs["groups"];
  if (conv.attrs.find("group") == conv.attrs.end())
    conv.attrs["group"] = "1";

  return conv;
}

ConvertedOp TraceOpConverter::convert_linear(const OpEntry& entry) {
  ConvertedOp gemm;
  gemm.optype = "Gemm";
  gemm.attrs = entry.attrs;

  if (!entry.inputs.empty())
    gemm.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2)
    gemm.attrs["weight_shape"] = shape_to_str(entry.inputs[1].shape);
  if (!entry.outputs.empty())
    gemm.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  gemm.attrs["has_bias"] = (entry.inputs.size() >= 3) ? "1" : "0";
  return gemm;
}

ConvertedOp TraceOpConverter::convert_matmul(const OpEntry& entry) {
  ConvertedOp gemm;
  gemm.optype = "Gemm";
  gemm.attrs = entry.attrs;

  if (!entry.inputs.empty())
    gemm.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2)
    gemm.attrs["weight_shape"] = shape_to_str(entry.inputs[1].shape);
  if (!entry.outputs.empty())
    gemm.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  gemm.attrs["has_bias"] = "0";
  return gemm;
}

ConvertedOp TraceOpConverter::convert_addmm(const OpEntry& entry) {
  ConvertedOp gemm;
  gemm.optype = "Gemm";
  gemm.attrs = entry.attrs;

  if (entry.inputs.size() >= 2)
    gemm.attrs["input_shape"] = shape_to_str(entry.inputs[1].shape);
  if (entry.inputs.size() >= 3)
    gemm.attrs["weight_shape"] = shape_to_str(entry.inputs[2].shape);
  if (!entry.outputs.empty())
    gemm.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  gemm.attrs["has_bias"] = (entry.inputs.size() >= 1) ? "1" : "0";
  return gemm;
}

ConvertedOp TraceOpConverter::convert_max_pool2d(const OpEntry& entry) {
  ConvertedOp pool;
  pool.optype = "MaxPool";
  pool.attrs = entry.attrs;

  if (!entry.inputs.empty())
    pool.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    pool.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (pool.attrs.find("kernel_shape") == pool.attrs.end())
    pool.attrs["kernel_shape"] = "2,2";
  if (pool.attrs.find("strides") == pool.attrs.end())
    pool.attrs["strides"] = "2,2";
  if (pool.attrs.find("pads") == pool.attrs.end())
    pool.attrs["pads"] = "0,0,0,0";
  return pool;
}

ConvertedOp TraceOpConverter::convert_adaptive_avg_pool2d(const OpEntry& entry) {
  ConvertedOp pool;
  pool.optype = "AdaptiveAvgPool";
  pool.attrs = entry.attrs;

  if (!entry.inputs.empty())
    pool.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    pool.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (pool.attrs.find("output_size") == pool.attrs.end() && !entry.outputs.empty())
    pool.attrs["output_size"] = shape_to_str(entry.outputs[0].shape);
  return pool;
}

ConvertedOp TraceOpConverter::convert_avg_pool2d(const OpEntry& entry) {
  ConvertedOp pool;
  pool.optype = "AdaptiveAvgPool";
  pool.attrs = entry.attrs;

  if (!entry.inputs.empty())
    pool.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    pool.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (pool.attrs.find("kernel_shape") == pool.attrs.end())
    pool.attrs["kernel_shape"] = "2,2";
  if (pool.attrs.find("strides") == pool.attrs.end())
    pool.attrs["strides"] = "2,2";
  if (pool.attrs.find("pads") == pool.attrs.end())
    pool.attrs["pads"] = "0,0,0,0";
  return pool;
}

ConvertedOp TraceOpConverter::convert_flatten(const OpEntry& entry) {
  ConvertedOp flat;
  flat.optype = "Flatten";
  flat.attrs = entry.attrs;

  if (!entry.inputs.empty())
    flat.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    flat.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (flat.attrs.find("start_dim") == flat.attrs.end())
    flat.attrs["start_dim"] = "1";
  if (flat.attrs.find("end_dim") == flat.attrs.end())
    flat.attrs["end_dim"] = "-1";
  return flat;
}

ConvertedOp TraceOpConverter::convert_layer_norm(const OpEntry& entry) {
  ConvertedOp ln;
  ln.optype = "SkipLayerNorm";
  ln.attrs = entry.attrs;

  if (!entry.inputs.empty())
    ln.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2)
    ln.attrs["weight_shape"] = shape_to_str(entry.inputs[1].shape);
  if (entry.inputs.size() >= 3)
    ln.attrs["bias_shape"] = shape_to_str(entry.inputs[2].shape);
  if (!entry.outputs.empty())
    ln.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (ln.attrs.find("normalized_shape") == ln.attrs.end() && !entry.inputs.empty())
    ln.attrs["normalized_shape"] = std::to_string(entry.inputs[0].shape.back());
  return ln;
}

ConvertedOp TraceOpConverter::convert_gelu(const OpEntry& entry) {
  ConvertedOp act;
  act.optype = "BiasGelu";
  act.attrs = entry.attrs;

  if (!entry.inputs.empty())
    act.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    act.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  return act;
}

ConvertedOp TraceOpConverter::convert_silu(const OpEntry& entry) {
  ConvertedOp act;
  act.optype = "BiasGelu";
  act.attrs = entry.attrs;
  act.attrs["activation_type"] = "silu";

  if (!entry.inputs.empty())
    act.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    act.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  return act;
}

ConvertedOp TraceOpConverter::convert_softmax(const OpEntry& entry) {
  ConvertedOp sm;
  sm.optype = "Softmax";
  sm.attrs = entry.attrs;

  if (!entry.inputs.empty())
    sm.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    sm.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);

  if (sm.attrs.find("dim") == sm.attrs.end())
    sm.attrs["dim"] = "-1";
  return sm;
}

ConvertedOp TraceOpConverter::convert_dummy(const OpEntry& entry) {
  ConvertedOp dummy;
  dummy.optype = "Dummy";
  dummy.attrs = entry.attrs;

  if (!entry.inputs.empty())
    dummy.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty())
    dummy.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  return dummy;
}

} // namespace trace_frontend
