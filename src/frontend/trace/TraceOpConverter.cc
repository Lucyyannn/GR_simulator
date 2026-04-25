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

std::string TraceOpConverter::join_shapes(const std::vector<TensorEntry>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (i != 0) result += ";";
    result += shape_to_str(tensors[i].shape);
  }
  return result;
}

std::string TraceOpConverter::join_names(const std::vector<TensorEntry>& tensors) {
  std::string result;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (i != 0) result += ";";
    result += tensors[i].name;
  }
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
  if (name == "aten::split" || name == "aten::chunk" || name == "split" ||
      name == "chunk")
    return convert_split(entry);
  if (name == "aten::cat" || name == "cat")
    return convert_cat(entry);
  if (name == "aten::mul" || name == "mul")
    return convert_mul(entry);
  if (name == "aten::transpose" || name == "aten::permute" ||
      name == "aten::reshape" || name == "aten::view" || name == "transpose" ||
      name == "permute" || name == "reshape" || name == "view")
    return convert_view(entry);
  if (name == "aten::softmax" || name == "softmax")
    return convert_softmax(entry);
  if (name == "aten::embedding" || name == "embedding")
    return convert_embedding(entry);

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
  if (!entry.outputs.empty())
    conv.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    gemm.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    gemm.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    gemm.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    pool.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    pool.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    pool.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    flat.attrs["output_name"] = entry.outputs[0].name;

  if (flat.attrs.find("start_dim") == flat.attrs.end())
    flat.attrs["start_dim"] = "1";
  if (flat.attrs.find("end_dim") == flat.attrs.end())
    flat.attrs["end_dim"] = "-1";
  return flat;
}

ConvertedOp TraceOpConverter::convert_layer_norm(const OpEntry& entry) {
  ConvertedOp ln;
  ln.optype = "LayerNorm";
  ln.attrs = entry.attrs;

  if (!entry.inputs.empty())
    ln.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2)
    ln.attrs["weight_shape"] = shape_to_str(entry.inputs[1].shape);
  if (entry.inputs.size() >= 3)
    ln.attrs["bias_shape"] = shape_to_str(entry.inputs[2].shape);
	  if (!entry.outputs.empty())
	    ln.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  if (!entry.outputs.empty())
    ln.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    act.attrs["output_name"] = entry.outputs[0].name;
	  return act;
}

ConvertedOp TraceOpConverter::convert_silu(const OpEntry& entry) {
	  ConvertedOp act;
	  act.optype = "BiasAct";
	  act.attrs = entry.attrs;
  act.attrs["activation"] = "silu";
  act.attrs["has_bias"] = "0";
  act.attrs["llama_mlp"] = "0";

	  if (!entry.inputs.empty())
	    act.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
	  if (!entry.outputs.empty())
	    act.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  if (!entry.outputs.empty())
    act.attrs["output_name"] = entry.outputs[0].name;
	  return act;
}

ConvertedOp TraceOpConverter::convert_split(const OpEntry& entry) {
  ConvertedOp split;
  split.optype = "Split";
  split.attrs = entry.attrs;
  if (!entry.inputs.empty())
    split.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  split.attrs["output_shapes"] = join_shapes(entry.outputs);
  split.attrs["output_names"] = join_names(entry.outputs);
  if (!split.attrs.count("axis") && split.attrs.count("dim"))
    split.attrs["axis"] = split.attrs["dim"];
  if (!split.attrs.count("axis")) split.attrs["axis"] = "-1";
  if (split.attrs["axis"] == "-1" && !entry.inputs.empty())
    split.attrs["axis"] = std::to_string(entry.inputs[0].shape.size() - 1);
  return split;
}

ConvertedOp TraceOpConverter::convert_cat(const OpEntry& entry) {
  ConvertedOp cat;
  cat.optype = "Concat";
  cat.attrs = entry.attrs;
  if (!cat.attrs.count("axis") && cat.attrs.count("dim"))
    cat.attrs["axis"] = cat.attrs["dim"];
  if (!cat.attrs.count("axis")) cat.attrs["axis"] = "0";
  if (!entry.outputs.empty()) {
    cat.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
    cat.attrs["output_name"] = entry.outputs[0].name;
  }
  return cat;
}

ConvertedOp TraceOpConverter::convert_mul(const OpEntry& entry) {
  ConvertedOp mul;
  mul.optype = "Elementwise";
  mul.attrs = entry.attrs;
  mul.attrs["elementwise_op"] = "mul";
  if (!entry.inputs.empty())
    mul.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty()) {
    mul.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
    mul.attrs["output_name"] = entry.outputs[0].name;
  }
  return mul;
}

ConvertedOp TraceOpConverter::convert_view(const OpEntry& entry) {
  ConvertedOp view;
  view.optype = "View";
  view.attrs = entry.attrs;
  if (!entry.inputs.empty())
    view.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
  if (!entry.outputs.empty()) {
    view.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
    view.attrs["output_name"] = entry.outputs[0].name;
  }
  return view;
}

ConvertedOp TraceOpConverter::convert_embedding(const OpEntry& entry) {
  ConvertedOp embedding;
  embedding.optype = "Embedding";
  embedding.attrs = entry.attrs;

  if (!entry.inputs.empty())
    embedding.attrs["weight_shape"] = shape_to_str(entry.inputs[0].shape);
  if (entry.inputs.size() >= 2) {
    embedding.attrs["indices_shape"] = shape_to_str(entry.inputs[1].shape);
    embedding.attrs["indices_dtype"] = entry.inputs[1].dtype;
  }
	  if (!entry.outputs.empty()) {
	    embedding.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
    embedding.attrs["output_name"] = entry.outputs[0].name;
  }

  return embedding;
}

ConvertedOp TraceOpConverter::convert_softmax(const OpEntry& entry) {
  ConvertedOp sm;
  sm.optype = "Softmax";
  sm.attrs = entry.attrs;

  if (!entry.inputs.empty())
    sm.attrs["input_shape"] = shape_to_str(entry.inputs[0].shape);
	  if (!entry.outputs.empty())
	    sm.attrs["output_shape"] = shape_to_str(entry.outputs[0].shape);
  if (!entry.outputs.empty())
    sm.attrs["output_name"] = entry.outputs[0].name;

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
  if (!entry.outputs.empty())
    dummy.attrs["output_name"] = entry.outputs[0].name;
  return dummy;
}

} // namespace trace_frontend
