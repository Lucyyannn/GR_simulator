#include "TraceModel.h"
#include "Ssd.h"
#include "operations/OperationFactory.h"
#include "frontend/trace/TraceParser.h"
#include "frontend/trace/TraceOpConverter.h"

TraceModel::TraceModel(const std::string& trace_path,
                       json model_config,
                       SimulationConfig config,
                       const std::string& name,
                       MappingTable& mapping_table)
    : Model("", model_config, config, name, mapping_table),
      _trace_path(trace_path) {
}

uint32_t TraceModel::register_tensor(const trace_frontend::TensorEntry& entry, bool produced) {
  Tensor* existing = find_tensor(entry.name);
  if (existing) return existing->get_id();

  std::vector<uint32_t> dims = entry.shape;
  auto tensor = std::make_unique<Tensor>(
      _root_node_id, entry.name, dims, _config.precision, produced);
  uint32_t id = tensor->get_id();
  if (produced) tensor->set_produced();
  _tensor_map[id] = std::move(tensor);
  return id;
}

void TraceModel::initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  weight_table.clear();
  _graph = trace_frontend::TraceParser::parse(_trace_path);

  std::set<std::string> seen;
  for (auto& op : _graph.operators) {
    for (auto& inp : op.inputs) {
      if (!inp.is_weight || seen.count(inp.name)) continue;
      seen.insert(inp.name);
      std::vector<uint32_t> dims = inp.shape;
      auto tensor = std::make_unique<Tensor>(
          _root_node_id, inp.name, dims, _config.precision, true);
      tensor->set_produced();
      weight_table.push_back(std::move(tensor));
    }
  }
  spdlog::info("[TraceModel] initialize_weight: {} weight tensors registered", weight_table.size());
}

void TraceModel::initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  auto start = std::chrono::high_resolution_clock::now();

  if (_graph.operators.empty())
    _graph = trace_frontend::TraceParser::parse(_trace_path);

  for (auto& wt : weight_table) {
    auto tensor = std::make_unique<Tensor>(*wt.get());
    tensor->set_produced();
    uint32_t id = tensor->get_id();
    _tensor_map[id] = std::move(tensor);
  }

  std::set<std::string> input_names;
  for (auto& op : _graph.operators) {
    for (auto& inp : op.inputs) {
      if (inp.is_weight) continue;
      if (input_names.count(inp.name)) continue;
      input_names.insert(inp.name);
      if (!find_tensor(inp.name)) {
        std::vector<uint32_t> dims = inp.shape;
        auto tensor = std::make_unique<Tensor>(
            _root_node_id, inp.name, dims, _config.precision, true);
        tensor->set_produced();
        _tensor_map[tensor->get_id()] = std::move(tensor);
      }
    }
  }

  for (auto& op_entry : _graph.operators) {
    auto converted = trace_frontend::TraceOpConverter::convert(op_entry);

    for (auto& inp : op_entry.inputs) {
      if (!find_tensor(inp.name)) {
        register_tensor(inp, false);
      }
    }

    auto op = OperationFactory::create_from_trace(
        this, converted, op_entry, _target_core);
    if (op) {
      // Add all declared inputs first
      for (auto& inp : op_entry.inputs) {
        Tensor* t = find_tensor(inp.name);
        if (t) op->add_input(t->get_id());
      }
      // For SkipLayerNorm from layer_norm (only 1 activation input),
      // synthesize a zero-valued skip tensor so _INPUT_OPERAND+1 is valid.
      if (converted.optype == "SkipLayerNorm" && op_entry.inputs.size() < 4) {
        Tensor* input_tensor = find_tensor(op_entry.inputs[0].name);
        if (input_tensor) {
          std::string skip_name = name_gen(op_entry.inputs[0].name, "skip_syn");
          Tensor* existing_skip = find_tensor(skip_name);
          if (!existing_skip) {
            auto skip_dims = input_tensor->get_dims();
            auto skip_tensor = std::make_unique<Tensor>(
                _id, skip_name, skip_dims, _config.precision, true);
            skip_tensor->set_produced();
            uint32_t skip_id = skip_tensor->get_id();
            _tensor_map[skip_id] = std::move(skip_tensor);
            op->add_input(skip_id);
          } else {
            op->add_input(existing_skip->get_id());
          }
        }
      }
      // For BiasGelu (aten::gelu) with no bias input, synthesize one.
      if (converted.optype == "BiasGelu" && op_entry.inputs.size() < 2) {
        Tensor* input_tensor = find_tensor(op_entry.inputs[0].name);
        if (input_tensor) {
          std::string bias_name = name_gen(op_entry.inputs[0].name, "gelu_bias_syn");
          Tensor* existing_bias = find_tensor(bias_name);
          if (!existing_bias) {
            std::vector<uint32_t> bias_dims = {input_tensor->get_dims().back()};
            auto bias_tensor = std::make_unique<Tensor>(
                _id, bias_name, bias_dims, _config.precision, true);
            bias_tensor->set_produced();
            uint32_t bias_id = bias_tensor->get_id();
            _tensor_map[bias_id] = std::move(bias_tensor);
            op->add_input(bias_id);
          } else {
            op->add_input(existing_bias->get_id());
          }
        }
      }
      _operation_map[op->get_id()] = std::move(op);
    }
  }

  for (auto& [key, val] : _operation_map) {
    val->initialize_tiles(_mapping_table);
  }

  for (auto& [key, val] : _operation_map) {
    if (val->check_executable()) {
      spdlog::debug("[TraceModel] runnable op: {}", val->get_optype());
      _executable_layer.push_back(val.get());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("[TraceModel] {} initialization time: {:2f} seconds, {} ops, {} runnable",
               _name, duration.count(), _operation_map.size(), _executable_layer.size());
}

void TraceModel::prefill_ssd_tensors(Ssd* ssd) {
  if (ssd == nullptr) return;

  uint64_t prefilled_tensors = 0;
  uint64_t prefilled_bytes = 0;
  for (const auto& [_, tensor] : _tensor_map) {
    if (tensor == nullptr) continue;
    if (!ssd->owns_address(tensor->get_address())) continue;
    ssd->prefill_range(tensor->get_address(), tensor->get_size());
    prefilled_tensors++;
    prefilled_bytes += tensor->get_size();
  }

  if (prefilled_tensors > 0) {
    spdlog::info(
        "[TraceModel] {} prefilling {} SSD tensors ({} bytes) without timing",
        _name, prefilled_tensors, prefilled_bytes);
  }
}
