#include "TraceModel.h"
#include "Ssd.h"
#include "memory/StorageController.h"
#include "memory/ResidencyManager.h"
#include "operations/OperationFactory.h"
#include "frontend/trace/TraceParser.h"
#include "frontend/trace/TraceOpConverter.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <numeric>

TraceModel::TraceModel(const std::string& trace_path,
                       json model_config,
                       SimulationConfig config,
                       const std::string& name,
                       MappingTable& mapping_table)
    : Model("", model_config, config, name, mapping_table),
      _trace_path(trace_path) {
}

namespace {

MemoryMedium parse_medium_name(const std::string& name) {
  if (name == "hbm" || name == "HBM") return MemoryMedium::HBM;
  if (name == "ddr" || name == "DDR" || name == "dram" || name == "DRAM")
    return MemoryMedium::DDR;
  if (name == "ssd" || name == "SSD") return MemoryMedium::SSD;
  return MemoryMedium::UNKNOWN;
}

bool is_resident_role(const std::string& role) {
  return role == "weight" || role == "kv_cache_k" || role == "kv_cache_v";
}

uint64_t tensor_bytes_for_shape(const std::vector<uint32_t>& shape,
                                uint32_t precision) {
  if (shape.empty()) return precision;
  uint64_t elems = 1;
  for (uint32_t dim : shape) elems *= dim;
  return elems * precision;
}

uint32_t flattened_elems(const std::vector<uint32_t>& shape) {
  if (shape.empty()) return 1;
  return std::accumulate(shape.begin(), shape.end(), 1u,
                         std::multiplies<uint32_t>());
}

}  // namespace

void TraceModel::remember_tensor_entry(const trace_frontend::TensorEntry& entry) {
  if (!entry.name.empty()) _tensor_entries[entry.name] = entry;
}

uint32_t TraceModel::effective_user_id(
    const trace_frontend::TensorEntry& entry) const {
  if (_model_config.contains("user_id"))
    return _model_config["user_id"].get<uint32_t>();
  return entry.user_id;
}

uint32_t TraceModel::effective_batch_id(
    const trace_frontend::TensorEntry& entry) const {
  if (_model_config.contains("batch_id"))
    return _model_config["batch_id"].get<uint32_t>();
  return entry.batch_id;
}

uint32_t TraceModel::effective_macro_batch_id(
    const trace_frontend::TensorEntry& entry) const {
  if (_model_config.contains("macro_batch_id"))
    return _model_config["macro_batch_id"].get<uint32_t>();
  return entry.macro_batch_id;
}

std::string TraceModel::effective_logical_id(
    const trace_frontend::TensorEntry& entry) const {
  if (entry.role == "kv_cache_k" || entry.role == "kv_cache_v") {
    const char* suffix = entry.role == "kv_cache_k" ? "kc" : "vc";
    return "user" + std::to_string(effective_user_id(entry)) + ".layer" +
           std::to_string(entry.layer_id) + "." + suffix;
  }
  if (!entry.logical_id.empty()) return entry.logical_id;
  return entry.name;
}

uint32_t TraceModel::register_tensor(const trace_frontend::TensorEntry& entry, bool produced) {
  Tensor* existing = find_tensor(entry.name);
  if (existing) return existing->get_id();

  std::vector<uint32_t> dims = entry.shape;
  auto tensor = std::make_unique<Tensor>(
      _root_node_id, entry.name, dims, _config.precision, produced);
	  uint32_t id = tensor->get_id();
	  if (produced) tensor->set_produced();
  Tensor* tensor_ptr = tensor.get();
	  _tensor_map[id] = std::move(tensor);
  apply_trace_storage(tensor_ptr, entry);
	  return id;
}

void TraceModel::apply_trace_storage(Tensor* tensor,
                                     const trace_frontend::TensorEntry& entry) {
  if (tensor == nullptr) return;

  MemoryMedium runtime_medium = parse_medium_name(entry.runtime_medium);
  if (runtime_medium == MemoryMedium::UNKNOWN)
    runtime_medium = parse_medium_name(entry.initial_medium);

  MemoryMedium initial_medium = parse_medium_name(entry.initial_medium);
  const bool needs_preload =
      _graph.metadata.baseline_preload &&
      initial_medium != MemoryMedium::UNKNOWN &&
      runtime_medium != MemoryMedium::UNKNOWN &&
      initial_medium != runtime_medium;

  if (needs_preload && entry.role == "embedding_rows") {
    tensor->relocate(runtime_medium);
    const uint64_t row_bytes =
        entry.shape.empty() ? tensor->get_size()
                            : static_cast<uint64_t>(entry.shape.back()) *
                                  _config.precision;
    const uint32_t rows = entry.indices_values.empty()
                              ? flattened_elems(entry.shape) /
                                    std::max<uint32_t>(entry.shape.empty()
                                                           ? 1
                                                           : entry.shape.back(),
                                                       1)
                              : entry.indices_values.size();
    uint64_t source_bytes = tensor_bytes_for_shape(entry.source_shape,
                                                  _config.precision);
    if (source_bytes == _config.precision && !entry.source_shape.empty())
      source_bytes = tensor->get_size();
    std::string source_id = entry.source_logical_id.empty()
                                ? "embedding.table"
                                : entry.source_logical_id;
    addr_type source_base =
        _residency_manager
            ? _residency_manager->source_addr(source_id, source_bytes,
                                             initial_medium)
            : allocate_address_in_medium(static_cast<uint32_t>(source_bytes),
                                         initial_medium);

    for (uint32_t i = 0; i < rows; ++i) {
      uint32_t row = entry.indices_values.empty() ? i : entry.indices_values[i];
      if (!entry.source_shape.empty() && entry.source_shape[0] > 0)
        row %= entry.source_shape[0];
      _data_movements.push_back(PlannedDataMovement{
          .tensor_name = tensor->get_name(),
          .logical_id = source_id,
          .role = entry.role,
          .source = initial_medium,
          .destination = runtime_medium,
          .src_addr = source_base + static_cast<addr_type>(row) * row_bytes,
          .dst_addr = tensor->get_address() + static_cast<addr_type>(i) * row_bytes,
          .bytes = row_bytes,
          .batch_id = effective_batch_id(entry),
          .macro_batch_id = effective_macro_batch_id(entry),
          .user_id = effective_user_id(entry),
      });
    }
    return;
  }

  if (needs_preload && is_resident_role(entry.role) && _residency_manager &&
      runtime_medium == MemoryMedium::HBM) {
    std::string logical_id = effective_logical_id(entry);
    if (_residency_manager->is_resident(logical_id)) {
      tensor->set_address(_residency_manager->resident_addr(logical_id));
      spdlog::debug("[TraceModel] {} uses resident {} at 0x{:x}",
                    tensor->get_name(), logical_id, tensor->get_address());
      return;
    }

    addr_type hbm_addr = _residency_manager->reserve_destination(
        logical_id, tensor->get_size(), runtime_medium);
    tensor->set_address(hbm_addr);
    addr_type source_addr = _residency_manager->source_addr(
        logical_id, tensor->get_size(), initial_medium);
    _data_movements.push_back(PlannedDataMovement{
        .tensor_name = tensor->get_name(),
        .logical_id = logical_id,
        .role = entry.role,
        .source = initial_medium,
        .destination = runtime_medium,
        .src_addr = source_addr,
        .dst_addr = tensor->get_address(),
        .bytes = tensor->get_size(),
        .batch_id = effective_batch_id(entry),
        .macro_batch_id = effective_macro_batch_id(entry),
        .user_id = effective_user_id(entry),
        .makes_resident = true,
    });
    return;
  }

  if (runtime_medium != MemoryMedium::UNKNOWN) {
    tensor->relocate(runtime_medium);
  }

  if (!_graph.metadata.baseline_preload ||
      initial_medium == MemoryMedium::UNKNOWN ||
      runtime_medium == MemoryMedium::UNKNOWN ||
      initial_medium == runtime_medium) {
    return;
  }

  addr_type source_addr =
      allocate_address_in_medium(static_cast<uint32_t>(tensor->get_size()),
                                 initial_medium);
  _data_movements.push_back(PlannedDataMovement{
      .tensor_name = tensor->get_name(),
      .logical_id = entry.logical_id,
      .role = entry.role,
      .source = initial_medium,
      .destination = runtime_medium,
      .src_addr = source_addr,
      .dst_addr = tensor->get_address(),
      .bytes = tensor->get_size(),
      .batch_id = effective_batch_id(entry),
      .macro_batch_id = effective_macro_batch_id(entry),
      .user_id = effective_user_id(entry),
  });
}

void TraceModel::initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  weight_table.clear();
  _graph = trace_frontend::TraceParser::parse(_trace_path);

  std::set<std::string> seen;
	  for (auto& op : _graph.operators) {
    for (auto& inp : op.inputs) remember_tensor_entry(inp);
    for (auto& out : op.outputs) remember_tensor_entry(out);
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
  _data_movements.clear();
  _submitted_movement_ids.clear();
  _resident_loads.clear();
  _data_movements_submitted = false;

  for (auto& op : _graph.operators) {
    for (auto& inp : op.inputs) remember_tensor_entry(inp);
    for (auto& out : op.outputs) remember_tensor_entry(out);
  }

  std::set<std::string> produced_names;
  for (auto& op : _graph.operators) {
    for (auto& out : op.outputs) produced_names.insert(out.name);
  }

	  for (auto& wt : weight_table) {
	    auto tensor = std::make_unique<Tensor>(*wt.get());
	    tensor->set_produced();
	    uint32_t id = tensor->get_id();
    Tensor* tensor_ptr = tensor.get();
	    _tensor_map[id] = std::move(tensor);
    auto entry_it = _tensor_entries.find(tensor_ptr->get_name());
    if (entry_it != _tensor_entries.end())
      apply_trace_storage(tensor_ptr, entry_it->second);
	  }

	  std::set<std::string> input_names;
	  for (auto& op : _graph.operators) {
	    for (auto& inp : op.inputs) {
	      if (inp.is_weight) continue;
      if (produced_names.count(inp.name)) continue;
	      if (input_names.count(inp.name)) continue;
	      input_names.insert(inp.name);
	      if (!find_tensor(inp.name)) {
        register_tensor(inp, true);
	      }
	    }
	  }

	  for (auto& op_entry : _graph.operators) {
	    auto converted = trace_frontend::TraceOpConverter::convert(op_entry);
    if ((converted.optype == "Split" || converted.optype == "View" ||
         converted.optype == "Concat") &&
        !converted.attrs.count("modeling_mode")) {
      std::string key = converted.optype;
      std::transform(key.begin(), key.end(), key.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      auto mode_it = _graph.metadata.op_modeling.find(key);
      if (mode_it != _graph.metadata.op_modeling.end())
        converted.attrs["modeling_mode"] = mode_it->second;
    }
    if (_graph.metadata.fail_on_unknown_op && converted.optype == "Dummy") {
      spdlog::error("[TraceModel] Unsupported trace op '{}' in fail-fast mode",
                    op_entry.name);
      std::exit(EXIT_FAILURE);
    }

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
      for (auto& out : op_entry.outputs) {
        Tensor* t = find_tensor(out.name);
        if (t) apply_trace_storage(t, out);
      }
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

uint64_t TraceModel::prepare_baseline_storage(StorageController* controller,
                                              uint64_t now_ps) {
  if (controller == nullptr || _data_movements.empty()) return now_ps;

  submit_data_movements(controller, now_ps);

  uint64_t current_ps = now_ps;
  while (!data_movements_ready(controller)) {
    uint64_t next_ps = controller->next_event_time_ps();
    if (next_ps == std::numeric_limits<uint64_t>::max()) break;
    if (next_ps <= current_ps) next_ps = current_ps + 1;
    current_ps = next_ps;
    controller->advance_to(current_ps);
    while (controller->has_ready_response()) {
      MemoryAccess* response = controller->top_ready_response();
      controller->pop_ready_response();
      delete response;
    }
  }

  spdlog::info("[TraceModel] {} baseline preload finished at {} ps",
               _name, current_ps);
  return current_ps;
}

std::vector<uint64_t> TraceModel::submit_data_movements(
    StorageController* controller, uint64_t now_ps) {
  if (controller == nullptr || _data_movements.empty())
    return _submitted_movement_ids;
  if (_data_movements_submitted) return _submitted_movement_ids;

  spdlog::info("[TraceModel] {} submitting {} data movements",
               _name, _data_movements.size());
  for (const auto& movement : _data_movements) {
    MigrationRequest request;
    request.src_medium = movement.source;
    request.dst_medium = movement.destination;
    request.src_addr = movement.src_addr;
    request.dst_addr = movement.dst_addr;
    request.bytes = movement.bytes;
    uint64_t movement_id = controller->submit_migration_request(request, now_ps);
    _submitted_movement_ids.push_back(movement_id);
    if (movement.makes_resident) {
      _resident_loads.push_back(ResidentLoad{
          .logical_id = movement.logical_id,
          .hbm_addr = movement.dst_addr,
          .bytes = movement.bytes,
          .movement_id = movement_id,
      });
    }
    spdlog::debug(
        "[TraceModel] movement {} {}: 0x{:x} -> 0x{:x}, {} bytes",
        movement_id, movement.tensor_name, movement.src_addr, movement.dst_addr,
        movement.bytes);
  }
  _data_movements_submitted = true;
  return _submitted_movement_ids;
}

bool TraceModel::data_movements_ready(StorageController* controller) const {
  if (_submitted_movement_ids.empty()) return true;
  if (controller == nullptr) return true;
  return controller->movements_done(_submitted_movement_ids);
}

void TraceModel::complete_data_movements(StorageController* controller) {
  if (controller == nullptr || _residency_manager == nullptr) return;
  for (const auto& load : _resident_loads) {
    if (controller->movement_done(load.movement_id)) {
      _residency_manager->mark_resident(load.logical_id, load.hbm_addr,
                                        load.bytes);
    }
  }
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
