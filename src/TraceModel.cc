#include "TraceModel.h"
#include "Ssd.h"
#include "memory/StorageController.h"
#include "memory/ResidencyManager.h"
#include "operations/OperationFactory.h"
#include "frontend/trace/TraceParser.h"
#include "frontend/trace/TraceOpConverter.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

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
  return role == "weight" || role.rfind("kv_cache_", 0) == 0;
}

bool is_batched_kv_role(const std::string& role) {
  return role == "kv_cache_k_batch" || role == "kv_cache_v_batch";
}

std::string kv_suffix_for_role(const std::string& role) {
  return role.find("_v") != std::string::npos ? "vc" : "kc";
}

uint64_t tensor_bytes_for_shape(const std::vector<uint32_t>& shape,
                                uint32_t precision) {
  if (shape.empty()) return precision;
  uint64_t elems = 1;
  for (uint32_t dim : shape) elems *= dim;
  return elems * precision;
}

uint64_t round_up_to(uint64_t value, uint64_t alignment) {
  if (alignment == 0) return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t row_bytes_for_axis(const std::vector<uint32_t>& shape,
                            uint32_t axis,
                            uint32_t precision) {
  if (shape.empty()) return precision;
  uint64_t elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == axis) continue;
    elems *= shape[i];
  }
  return elems * precision;
}

uint32_t flattened_elems(const std::vector<uint32_t>& shape) {
  if (shape.empty()) return 1;
  return std::accumulate(shape.begin(), shape.end(), 1u,
                         std::multiplies<uint32_t>());
}

std::vector<uint32_t> users_for_entry(
    const trace_frontend::TensorEntry& entry) {
  if (!entry.user_ids.empty()) return entry.user_ids;
  uint32_t count = entry.shape.empty() ? 1 : entry.shape.front();
  std::vector<uint32_t> users;
  users.reserve(count);
  for (uint32_t i = 0; i < count; ++i)
    users.push_back(entry.user_id + i);
  return users;
}

std::string medium_to_string(MemoryMedium medium) {
  if (medium == MemoryMedium::HBM) return "hbm";
  if (medium == MemoryMedium::DDR) return "ddr";
  if (medium == MemoryMedium::SSD) return "ssd";
  return "unknown";
}

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') escaped += "\"\"";
    else escaped += c;
  }
  escaped += "\"";
  return escaped;
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

bool TraceModel::apply_reuse_layout(
    Tensor* tensor, const trace_frontend::TensorEntry& entry) {
  if (tensor == nullptr) return false;
  if (!_graph.metadata.kv_reuse_enabled) return false;
  if (entry.reuse_mode != "row_reuse") return false;
  if (entry.reuse_logical_to_physical.empty()) return false;
  if (entry.reuse_axis >= entry.shape.size()) return false;
  if (entry.reuse_logical_to_physical.size() != entry.shape[entry.reuse_axis]) {
    spdlog::error(
        "[TraceModel] reuse mapping size {} does not match tensor {} axis {} extent {}",
        entry.reuse_logical_to_physical.size(), entry.name, entry.reuse_axis,
        entry.shape[entry.reuse_axis]);
    std::exit(EXIT_FAILURE);
  }

  uint32_t physical_rows = entry.reuse_physical_rows;
  for (uint32_t row : entry.reuse_logical_to_physical)
    physical_rows = std::max(physical_rows, row + 1);
  if (physical_rows == 0) return false;

  const uint64_t row_bytes =
      row_bytes_for_axis(entry.shape, entry.reuse_axis, _config.precision);
  const uint64_t row_stride =
      round_up_to(row_bytes, std::max<uint64_t>(_config.hbm.req_size, 1));
  tensor->set_reuse_layout(entry.reuse_axis, physical_rows, row_stride,
                           entry.reuse_logical_to_physical);
  return true;
}

void TraceModel::apply_trace_storage(Tensor* tensor,
                                     const trace_frontend::TensorEntry& entry) {
  if (tensor == nullptr) return;
  const bool has_reuse_layout = apply_reuse_layout(tensor, entry);

  MemoryMedium runtime_medium = parse_medium_name(entry.runtime_medium);
  if (runtime_medium == MemoryMedium::UNKNOWN)
    runtime_medium = parse_medium_name(entry.initial_medium);

  MemoryMedium initial_medium = parse_medium_name(entry.initial_medium);
  const bool needs_preload =
      _graph.metadata.baseline_preload &&
      initial_medium != MemoryMedium::UNKNOWN &&
      runtime_medium != MemoryMedium::UNKNOWN &&
      initial_medium != runtime_medium;

  if (needs_preload && entry.role == "embedding_rows" &&
      entry.shape.size() >= 3 && !entry.indices_values_per_user.empty()) {
    tensor->relocate(runtime_medium);
    const std::vector<uint32_t> users = users_for_entry(entry);
    const uint32_t batch_size = entry.shape[0];
    const uint32_t rows_per_user = entry.shape[1];
    const uint64_t row_bytes =
        static_cast<uint64_t>(entry.shape.back()) * _config.precision;
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

    if (layer_preload_enabled()) tensor->clear_produced();
    for (uint32_t b = 0; b < batch_size; ++b) {
      const uint32_t user_id = b < users.size() ? users[b] : entry.user_id + b;
      PlannedDataMovement movement{
          .tensor_name = tensor->get_name(),
          .logical_id = "batch" + std::to_string(effective_batch_id(entry)) +
                        ".macro" +
                        std::to_string(effective_macro_batch_id(entry)) +
                        ".user" + std::to_string(user_id) +
                        ".embedding_rows",
          .role = entry.role,
          .preload_group = entry.preload_group,
          .source = initial_medium,
          .destination = runtime_medium,
          .src_addr = source_base,
          .dst_addr = tensor->get_address() +
                      static_cast<addr_type>(b) * rows_per_user * row_bytes,
          .bytes = static_cast<uint64_t>(rows_per_user) * row_bytes,
          .batch_id = effective_batch_id(entry),
          .macro_batch_id = effective_macro_batch_id(entry),
          .user_id = user_id,
          .layer_id = static_cast<int32_t>(entry.layer_id),
          .tensor_id = tensor->get_id(),
      };
      const auto& per_user_indices =
          b < entry.indices_values_per_user.size()
              ? entry.indices_values_per_user[b]
              : entry.indices_values;
      auto add_segment = [&movement](addr_type src_addr, addr_type dst_addr,
                                     uint64_t bytes) {
        if (bytes == 0) return;
        if (!movement.segments.empty()) {
          auto& last = movement.segments.back();
          if (last.src_addr + last.bytes == src_addr &&
              last.dst_addr + last.bytes == dst_addr) {
            last.bytes += bytes;
            return;
          }
        }
        movement.segments.push_back(PlannedDataMovement::Segment{
            .src_addr = src_addr,
            .dst_addr = dst_addr,
            .bytes = bytes,
        });
      };
      for (uint32_t i = 0; i < rows_per_user; ++i) {
        uint32_t row = i;
        if (!per_user_indices.empty())
          row = per_user_indices[i % per_user_indices.size()];
        else if (!entry.indices_values.empty())
          row = entry.indices_values[(b * rows_per_user + i) %
                                     entry.indices_values.size()];
        if (!entry.source_shape.empty() && entry.source_shape[0] > 0)
          row %= entry.source_shape[0];
        add_segment(source_base + static_cast<addr_type>(row) * row_bytes,
                    movement.dst_addr + static_cast<addr_type>(i) * row_bytes,
                    row_bytes);
      }
      _data_movements.push_back(std::move(movement));
    }
    return;
  }

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

    if (layer_preload_enabled()) tensor->clear_produced();
    for (uint32_t i = 0; i < rows; ++i) {
      uint32_t row = entry.indices_values.empty() ? i : entry.indices_values[i];
      if (!entry.source_shape.empty() && entry.source_shape[0] > 0)
        row %= entry.source_shape[0];
      _data_movements.push_back(PlannedDataMovement{
          .tensor_name = tensor->get_name(),
          .logical_id = source_id,
          .role = entry.role,
          .preload_group = entry.preload_group,
          .source = initial_medium,
          .destination = runtime_medium,
          .src_addr = source_base + static_cast<addr_type>(row) * row_bytes,
          .dst_addr = tensor->get_address() + static_cast<addr_type>(i) * row_bytes,
          .bytes = row_bytes,
          .batch_id = effective_batch_id(entry),
          .macro_batch_id = effective_macro_batch_id(entry),
          .user_id = effective_user_id(entry),
          .layer_id = static_cast<int32_t>(entry.layer_id),
          .tensor_id = tensor->get_id(),
      });
    }
    return;
  }

  if (needs_preload && is_batched_kv_role(entry.role) && _residency_manager &&
      runtime_medium == MemoryMedium::HBM && entry.shape.size() == 3) {
    const std::vector<uint32_t> users = users_for_entry(entry);
    const uint32_t batch_size = entry.shape[0];
    const uint32_t logical_rows = entry.shape[1];
    const uint64_t row_bytes =
        static_cast<uint64_t>(entry.shape[2]) * _config.precision;
    const bool use_reuse =
        _graph.metadata.kv_reuse_enabled &&
        entry.reuse_mode == "row_reuse" &&
        entry.reuse_logical_to_physical_per_user.size() >= batch_size;
    const uint64_t row_stride =
        use_reuse ? round_up_to(row_bytes, std::max<uint64_t>(_config.hbm.req_size, 1))
                  : row_bytes;

    std::vector<std::string> logical_ids;
    std::vector<uint32_t> physical_rows;
    std::vector<uint64_t> physical_bytes;
    std::vector<uint64_t> logical_bytes;
    std::vector<std::vector<uint32_t>> maps(batch_size);
    logical_ids.reserve(batch_size);
    physical_rows.reserve(batch_size);
    physical_bytes.reserve(batch_size);
    logical_bytes.reserve(batch_size);
    for (uint32_t b = 0; b < batch_size; ++b) {
      const uint32_t user_id = b < users.size() ? users[b] : entry.user_id + b;
      logical_ids.push_back("user" + std::to_string(user_id) + ".layer" +
                            std::to_string(entry.layer_id) + "." +
                            kv_suffix_for_role(entry.role));
      uint32_t rows = logical_rows;
      if (use_reuse) {
        maps[b] = entry.reuse_logical_to_physical_per_user[b];
        rows = b < entry.reuse_physical_rows_per_user.size()
                   ? entry.reuse_physical_rows_per_user[b]
                   : 0;
        for (uint32_t row : maps[b]) rows = std::max(rows, row + 1);
      }
      physical_rows.push_back(rows);
      physical_bytes.push_back(static_cast<uint64_t>(rows) * row_stride);
      logical_bytes.push_back(static_cast<uint64_t>(logical_rows) * row_stride);
    }

    std::vector<addr_type> group_bases(batch_size, 0);
    bool any_known_addr = false;
    for (uint32_t b = 0; b < batch_size; ++b) {
      group_bases[b] = _residency_manager->resident_addr(logical_ids[b]);
      any_known_addr = any_known_addr || group_bases[b] != 0;
    }
    if (!any_known_addr) {
      std::vector<std::pair<std::string, uint64_t>> allocations;
      allocations.reserve(batch_size);
      for (uint32_t b = 0; b < batch_size; ++b)
        allocations.push_back({logical_ids[b], physical_bytes[b]});
      group_bases =
          _residency_manager->reserve_packed_destinations(allocations,
                                                          runtime_medium);
    } else {
      for (uint32_t b = 0; b < batch_size; ++b) {
        if (group_bases[b] == 0) {
          group_bases[b] = _residency_manager->reserve_destination(
              logical_ids[b], physical_bytes[b], runtime_medium);
        }
      }
    }

    tensor->set_address(group_bases.empty() ? tensor->get_address()
                                            : group_bases.front());
    tensor->set_group_layout(0, 1, row_stride, group_bases, physical_rows, maps);
    if (layer_preload_enabled()) tensor->clear_produced();

    for (uint32_t b = 0; b < batch_size; ++b) {
      const uint32_t user_id = b < users.size() ? users[b] : entry.user_id + b;
      addr_type source_addr = _residency_manager->source_addr(
          logical_ids[b], logical_bytes[b], initial_medium);
      PlannedDataMovement movement{
          .tensor_name = tensor->get_name(),
          .logical_id = logical_ids[b],
          .role = entry.role,
          .preload_group = entry.preload_group,
          .source = initial_medium,
          .destination = runtime_medium,
          .src_addr = source_addr,
          .dst_addr = group_bases[b],
          .bytes = physical_bytes[b],
          .batch_id = effective_batch_id(entry),
          .macro_batch_id = effective_macro_batch_id(entry),
          .user_id = user_id,
          .layer_id = static_cast<int32_t>(entry.layer_id),
          .tensor_id = tensor->get_id(),
          .makes_resident = true,
          .reuse_if_resident = true,
          .defer_tensor_ready = true,
          .resident_bytes = physical_bytes[b],
      };
      if (use_reuse) {
        std::vector<uint32_t> canonical_rows(
            physical_rows[b], std::numeric_limits<uint32_t>::max());
        for (uint32_t logical_row = 0; logical_row < maps[b].size();
             ++logical_row) {
          uint32_t physical_row = maps[b][logical_row];
          if (physical_row >= canonical_rows.size()) continue;
          canonical_rows[physical_row] =
              std::min(canonical_rows[physical_row], logical_row);
        }
        for (uint32_t physical_row = 0; physical_row < canonical_rows.size();
             ++physical_row) {
          uint32_t logical_row = canonical_rows[physical_row];
          if (logical_row == std::numeric_limits<uint32_t>::max()) continue;
          movement.segments.push_back(PlannedDataMovement::Segment{
              .src_addr = source_addr +
                          static_cast<addr_type>(logical_row) * row_stride,
              .dst_addr = group_bases[b] +
                          static_cast<addr_type>(physical_row) * row_stride,
              .bytes = row_stride,
          });
        }
      }
      note_residency_entry(movement);
      _data_movements.push_back(std::move(movement));
      if (use_reuse) {
        _reuse_logical_bytes += logical_bytes[b];
        _reuse_physical_bytes += physical_bytes[b];
      }
    }
    return;
  }

  if (needs_preload && is_resident_role(entry.role) && _residency_manager &&
      runtime_medium == MemoryMedium::HBM) {
    std::string logical_id = effective_logical_id(entry);
    if (_residency_manager->is_resident(logical_id)) {
      tensor->set_address(_residency_manager->resident_addr(logical_id));
      uint64_t resident_bytes =
          tensor->has_reuse_layout()
              ? static_cast<uint64_t>(tensor->reuse_physical_rows()) *
                    tensor->reuse_row_stride_bytes()
              : tensor->get_size();
      _residency_manager->note_entry(
          logical_id, resident_bytes,
          effective_user_id(entry), static_cast<int32_t>(entry.layer_id),
          entry.role, residency_next_use_rank(entry.layer_id));
      if (layer_preload_enabled()) {
        addr_type source_addr =
            _residency_manager->source_addr(logical_id,
                                            tensor->has_reuse_layout()
                                                ? tensor_bytes_for_shape(
                                                      entry.shape,
                                                      _config.precision)
                                                : tensor->get_size(),
                                            initial_medium);
        PlannedDataMovement movement{
            .tensor_name = tensor->get_name(),
            .logical_id = logical_id,
            .role = entry.role,
            .preload_group = entry.preload_group,
            .source = initial_medium,
            .destination = runtime_medium,
            .src_addr = source_addr,
            .dst_addr = tensor->get_address(),
            .bytes = resident_bytes,
            .batch_id = effective_batch_id(entry),
            .macro_batch_id = effective_macro_batch_id(entry),
            .user_id = effective_user_id(entry),
            .layer_id = static_cast<int32_t>(entry.layer_id),
            .tensor_id = tensor->get_id(),
            .makes_resident = true,
            .reuse_if_resident = true,
            .resident_bytes = resident_bytes,
        };
        if (has_reuse_layout && tensor->has_reuse_layout()) {
          const uint64_t row_stride = tensor->reuse_row_stride_bytes();
          std::vector<uint32_t> canonical_rows(
              tensor->reuse_physical_rows(),
              std::numeric_limits<uint32_t>::max());
          const auto& map = tensor->reuse_logical_to_physical();
          for (uint32_t logical_row = 0; logical_row < map.size();
               ++logical_row) {
            uint32_t physical_row = map[logical_row];
            if (physical_row >= canonical_rows.size()) continue;
            canonical_rows[physical_row] =
                std::min(canonical_rows[physical_row], logical_row);
          }
          movement.segments.clear();
          movement.bytes = resident_bytes;
          for (uint32_t physical_row = 0;
               physical_row < canonical_rows.size(); ++physical_row) {
            uint32_t logical_row = canonical_rows[physical_row];
            if (logical_row == std::numeric_limits<uint32_t>::max()) continue;
            movement.segments.push_back(PlannedDataMovement::Segment{
                .src_addr = source_addr +
                            static_cast<addr_type>(logical_row) * row_stride,
                .dst_addr = tensor->get_address() +
                            static_cast<addr_type>(physical_row) * row_stride,
                .bytes = row_stride,
            });
          }
        }
        tensor->clear_produced();
        _data_movements.push_back(std::move(movement));
      } else {
        pin_resident_use(static_cast<int32_t>(entry.layer_id), logical_id);
      }
      spdlog::debug("[TraceModel] {} uses resident {} at 0x{:x}",
                    tensor->get_name(), logical_id, tensor->get_address());
      return;
    }

    if (has_reuse_layout && tensor->has_reuse_layout()) {
      const uint64_t logical_rows = entry.shape[entry.reuse_axis];
      const uint64_t row_stride = tensor->reuse_row_stride_bytes();
      const uint64_t physical_bytes =
          static_cast<uint64_t>(tensor->reuse_physical_rows()) * row_stride;
      const uint64_t logical_bytes = logical_rows * row_stride;
      addr_type hbm_addr = _residency_manager->reserve_destination(
          logical_id, physical_bytes, runtime_medium);
      tensor->set_address(hbm_addr);
      addr_type source_addr =
          _residency_manager->source_addr(logical_id, logical_bytes,
                                          initial_medium);

      std::vector<uint32_t> canonical_rows(tensor->reuse_physical_rows(),
                                           std::numeric_limits<uint32_t>::max());
      const auto& map = tensor->reuse_logical_to_physical();
      for (uint32_t logical_row = 0; logical_row < map.size(); ++logical_row) {
        uint32_t physical_row = map[logical_row];
        if (physical_row >= canonical_rows.size()) continue;
        canonical_rows[physical_row] =
            std::min(canonical_rows[physical_row], logical_row);
      }

      PlannedDataMovement movement{
          .tensor_name = tensor->get_name(),
          .logical_id = logical_id,
          .role = entry.role,
          .preload_group = entry.preload_group,
          .source = initial_medium,
          .destination = runtime_medium,
          .src_addr = source_addr,
          .dst_addr = tensor->get_address(),
          .bytes = physical_bytes,
          .batch_id = effective_batch_id(entry),
          .macro_batch_id = effective_macro_batch_id(entry),
          .user_id = effective_user_id(entry),
          .layer_id = static_cast<int32_t>(entry.layer_id),
          .tensor_id = tensor->get_id(),
          .makes_resident = true,
          .resident_bytes = physical_bytes,
      };
      if (layer_preload_enabled()) tensor->clear_produced();
      auto add_segment = [&movement](addr_type src_addr, addr_type dst_addr,
                                     uint64_t bytes) {
        if (bytes == 0) return;
        if (!movement.segments.empty()) {
          auto& last = movement.segments.back();
          if (last.src_addr + last.bytes == src_addr &&
              last.dst_addr + last.bytes == dst_addr) {
            last.bytes += bytes;
            return;
          }
        }
        movement.segments.push_back(PlannedDataMovement::Segment{
            .src_addr = src_addr,
            .dst_addr = dst_addr,
            .bytes = bytes,
        });
      };

      for (uint32_t physical_row = 0; physical_row < canonical_rows.size();
           ++physical_row) {
        uint32_t logical_row = canonical_rows[physical_row];
        if (logical_row == std::numeric_limits<uint32_t>::max()) continue;
        add_segment(source_addr + static_cast<addr_type>(logical_row) * row_stride,
                    tensor->get_address() +
                        static_cast<addr_type>(physical_row) * row_stride,
                    row_stride);
      }
      note_residency_entry(movement);
      _data_movements.push_back(std::move(movement));
      _reuse_logical_bytes += logical_bytes;
      _reuse_physical_bytes += physical_bytes;
      return;
    }

    addr_type hbm_addr = _residency_manager->reserve_destination(
        logical_id, tensor->get_size(), runtime_medium);
    tensor->set_address(hbm_addr);
    addr_type source_addr = _residency_manager->source_addr(
        logical_id, tensor->get_size(), initial_medium);
    if (layer_preload_enabled()) tensor->clear_produced();
    PlannedDataMovement movement{
        .tensor_name = tensor->get_name(),
        .logical_id = logical_id,
        .role = entry.role,
        .preload_group = entry.preload_group,
        .source = initial_medium,
        .destination = runtime_medium,
        .src_addr = source_addr,
        .dst_addr = tensor->get_address(),
        .bytes = tensor->get_size(),
        .batch_id = effective_batch_id(entry),
        .macro_batch_id = effective_macro_batch_id(entry),
        .user_id = effective_user_id(entry),
        .layer_id = static_cast<int32_t>(entry.layer_id),
        .tensor_id = tensor->get_id(),
        .makes_resident = true,
        .resident_bytes = tensor->get_size(),
    };
    note_residency_entry(movement);
    _data_movements.push_back(std::move(movement));
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
  if (layer_preload_enabled()) tensor->clear_produced();
  _data_movements.push_back(PlannedDataMovement{
      .tensor_name = tensor->get_name(),
      .logical_id = entry.logical_id,
      .role = entry.role,
      .preload_group = entry.preload_group,
      .source = initial_medium,
      .destination = runtime_medium,
      .src_addr = source_addr,
      .dst_addr = tensor->get_address(),
      .bytes = tensor->get_size(),
      .batch_id = effective_batch_id(entry),
      .macro_batch_id = effective_macro_batch_id(entry),
      .user_id = effective_user_id(entry),
      .layer_id = static_cast<int32_t>(entry.layer_id),
      .tensor_id = tensor->get_id(),
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
  _preload_stages.clear();
  _next_stage_to_submit = 0;
  _operation_layer_ids.clear();
  _operation_names.clear();
  _compute_events.clear();
  _dispatched_layers.clear();
  _resident_uses_by_layer.clear();
  _remaining_ops_by_layer.clear();
  _data_movements_submitted = false;
  _reuse_logical_bytes = 0;
  _reuse_physical_bytes = 0;

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
         converted.optype == "Concat" || converted.optype == "LayerNorm") &&
        !converted.attrs.count("modeling_mode")) {
      std::string key = converted.optype;
      std::transform(key.begin(), key.end(), key.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      auto mode_it = _graph.metadata.op_modeling.find(key);
      if (mode_it != _graph.metadata.op_modeling.end())
        converted.attrs["modeling_mode"] = mode_it->second;
    }
    if (converted.optype == "LayerNorm" &&
        converted.attrs.count("modeling_mode") &&
        converted.attrs["modeling_mode"] == "skip") {
      converted.optype = "Dummy";
    }
    if (_graph.metadata.fail_on_unknown_op && converted.optype == "Dummy" &&
        (!converted.attrs.count("modeling_mode") ||
         converted.attrs["modeling_mode"] != "skip")) {
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
      int32_t op_layer_id = -1;
      for (const auto& inp : op_entry.inputs) {
        if (!inp.role.empty() && inp.role != "indices" &&
            inp.role != "embedding_table") {
          op_layer_id = static_cast<int32_t>(inp.layer_id);
          break;
        }
      }
      if (op_layer_id < 0) {
        for (const auto& out : op_entry.outputs) {
          if (!out.role.empty()) {
            op_layer_id = static_cast<int32_t>(out.layer_id);
            break;
          }
        }
      }
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
      uint32_t op_id = op->get_id();
      _operation_layer_ids[op_id] = op_layer_id;
      _operation_names[op_id] = op->get_name();
	      _operation_map[op_id] = std::move(op);
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
  for (const auto& [_, layer_id] : _operation_layer_ids) {
    if (layer_id >= 0) _remaining_ops_by_layer[layer_id]++;
  }
  build_preload_stages();

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
  if (_reuse_logical_bytes > 0) {
    spdlog::info(
        "[TraceModel] {} kv reuse logical={}B physical={}B saved={}B ({:.2f}%)",
        _name, _reuse_logical_bytes, _reuse_physical_bytes,
        _reuse_logical_bytes - _reuse_physical_bytes,
        100.0 * static_cast<double>(_reuse_logical_bytes - _reuse_physical_bytes) /
            static_cast<double>(_reuse_logical_bytes));
  }
  std::vector<MigrationRequest> requests;
  std::vector<size_t> request_movement_indices;
  requests.reserve(_data_movements.size());
  request_movement_indices.reserve(_data_movements.size());
  for (size_t movement_idx = 0; movement_idx < _data_movements.size();
       ++movement_idx) {
    const auto& movement = _data_movements[movement_idx];
    MigrationRequest request;
    request.src_medium = movement.source;
    request.dst_medium = movement.destination;
    request.src_addr = movement.src_addr;
    request.dst_addr = movement.dst_addr;
    request.bytes = movement.bytes;
    request.segments.reserve(movement.segments.size());
    for (const auto& segment : movement.segments) {
      request.segments.push_back(MigrationSegment{
          .src_addr = segment.src_addr,
          .dst_addr = segment.dst_addr,
          .bytes = segment.bytes,
      });
    }
    uint64_t request_bytes = request.segments.empty() ? request.bytes : 0;
    for (const auto& segment : request.segments) request_bytes += segment.bytes;
    if (request_bytes == 0) continue;
    request_movement_indices.push_back(movement_idx);
    requests.push_back(std::move(request));
  }

  std::vector<uint64_t> movement_ids =
      controller->submit_migration_requests(requests, now_ps);
  _submitted_movement_ids.insert(_submitted_movement_ids.end(),
                                 movement_ids.begin(), movement_ids.end());

  for (size_t i = 0; i < request_movement_indices.size() &&
                     i < movement_ids.size(); ++i) {
    const auto& movement = _data_movements[request_movement_indices[i]];
    uint64_t movement_id = movement_ids[i];
    if (movement.makes_resident) {
      _resident_loads.push_back(ResidentLoad{
          .logical_id = movement.logical_id,
          .hbm_addr = movement.dst_addr,
          .bytes = movement.resident_bytes == 0 ? movement.bytes
                                                 : movement.resident_bytes,
          .movement_id = movement_id,
          .layer_id = movement.layer_id,
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
  for (auto& load : _resident_loads) {
    if (!load.completed && controller->movement_done(load.movement_id)) {
      _residency_manager->mark_resident(load.logical_id, load.hbm_addr,
                                        load.bytes);
      pin_resident_use(load.layer_id, load.logical_id);
      load.completed = true;
    }
  }
}

bool TraceModel::layer_preload_enabled() const {
  return _config.layer_preload_enabled && _graph.metadata.baseline_preload;
}

bool TraceModel::uses_layer_preload() const {
  return layer_preload_enabled() && !_preload_stages.empty();
}

int64_t TraceModel::residency_next_use_rank(int32_t layer_id) const {
  if (layer_id < 0) return std::numeric_limits<int64_t>::max() / 2;
  return static_cast<int64_t>(layer_id);
}

void TraceModel::note_residency_entry(
    const PlannedDataMovement& movement) const {
  if (_residency_manager == nullptr || !movement.makes_resident) return;
  const uint64_t bytes =
      movement.resident_bytes == 0 ? movement.bytes : movement.resident_bytes;
  _residency_manager->note_entry(
      movement.logical_id, bytes, movement.user_id, movement.layer_id,
      movement.role, residency_next_use_rank(movement.layer_id));
}

void TraceModel::pin_resident_use(int32_t layer_id,
                                  const std::string& logical_id) {
  if (_residency_manager == nullptr || logical_id.empty()) return;
  auto& logical_ids = _resident_uses_by_layer[layer_id];
  if (logical_ids.insert(logical_id).second) {
    _residency_manager->pin(logical_id);
  }
}

void TraceModel::release_layer_residency_pins(int32_t layer_id) {
  if (_residency_manager == nullptr) return;
  auto it = _resident_uses_by_layer.find(layer_id);
  if (it == _resident_uses_by_layer.end()) return;
  for (const auto& logical_id : it->second) {
    _residency_manager->unpin(logical_id);
  }
  spdlog::debug("[TraceModel] {} release {} residency pins for layer {}",
                _name, it->second.size(), layer_id);
  _resident_uses_by_layer.erase(it);
}

void TraceModel::release_residency_pins() {
  if (_residency_manager == nullptr) return;
  for (const auto& [_, logical_ids] : _resident_uses_by_layer) {
    for (const auto& logical_id : logical_ids)
      _residency_manager->unpin(logical_id);
  }
  _resident_uses_by_layer.clear();
}

int32_t TraceModel::stage_layer_for_movement(
    const PlannedDataMovement& movement) const {
  if (movement.role == "embedding_rows") return 0;
  if (movement.layer_id >= 0) return movement.layer_id;
  return 0;
}

std::string TraceModel::preload_type_for_role(const std::string& role) const {
  if (role == "embedding_rows") return "candidate_embedding";
  if (role.rfind("kv_cache_", 0) == 0) return "kvcache";
  if (role == "weight") return "weights";
  return role.empty() ? "other" : role;
}

std::string TraceModel::preload_type_for_movement(
    const PlannedDataMovement& movement) const {
  if (!movement.preload_group.empty()) return movement.preload_group;
  return preload_type_for_role(movement.role);
}

void TraceModel::build_preload_stages() {
  _preload_stages.clear();
  _next_stage_to_submit = 0;
  if (!layer_preload_enabled() || _data_movements.empty()) return;

  const std::vector<std::string> preload_type_order = {
      "pre_attention", "candidate_embedding", "kvcache",
      "post_attention_weights", "weights", "other"};
  std::map<int32_t, std::vector<size_t>> movement_indices_by_layer;
  for (size_t movement_idx = 0; movement_idx < _data_movements.size();
       ++movement_idx) {
    const auto& movement = _data_movements[movement_idx];
    int32_t layer_id = stage_layer_for_movement(movement);
    movement_indices_by_layer[layer_id].push_back(movement_idx);
  }

  for (const auto& [layer_id, movement_indices] : movement_indices_by_layer) {
    PreloadStage stage;
    stage.layer_id = layer_id;
    stage.name = layer_id == 0 ? "bootstrap_layer0"
                               : "layer" + std::to_string(layer_id);
    stage.movement_indices = movement_indices;
    std::map<std::string, std::vector<size_t>> movement_indices_by_type;
    for (size_t movement_idx : movement_indices) {
      const auto& movement = _data_movements[movement_idx];
      movement_indices_by_type[preload_type_for_movement(movement)]
          .push_back(movement_idx);
      if (movement.segments.empty()) {
        stage.physical_bytes += movement.bytes;
      } else {
        for (const auto& segment : movement.segments)
          stage.physical_bytes += segment.bytes;
      }
    }
    std::set<std::string> emitted_types;
    for (const auto& preload_type : preload_type_order) {
      auto type_it = movement_indices_by_type.find(preload_type);
      if (type_it == movement_indices_by_type.end()) continue;
      stage.subtasks.push_back(PreloadSubtask{
          .preload_type = preload_type,
          .movement_indices = type_it->second,
      });
      emitted_types.insert(preload_type);
    }
    for (const auto& [preload_type, type_movement_indices] :
         movement_indices_by_type) {
      if (emitted_types.count(preload_type)) continue;
      stage.subtasks.push_back(PreloadSubtask{
          .preload_type = preload_type,
          .movement_indices = type_movement_indices,
      });
    }
    _preload_stages.push_back(std::move(stage));
  }
}

bool TraceModel::has_data_movement_stage_to_submit() const {
  if (!uses_layer_preload()) return false;
  if (_next_stage_to_submit >= _preload_stages.size()) return false;
  const auto& stage = _preload_stages[_next_stage_to_submit];
  if (stage.completed) return false;
  if (!stage_compute_frontier_ready(stage)) {
    for (const auto& subtask : stage.subtasks) {
      if (subtask.submitted && !subtask.completed) return false;
    }
  }
  return stage.next_subtask_to_submit < stage.subtasks.size();
}

std::vector<uint64_t> TraceModel::submit_next_data_movement_stage(
    StorageController* controller, uint64_t now_ps) {
  if (controller == nullptr || !has_data_movement_stage_to_submit()) return {};

  PreloadStage& stage = _preload_stages[_next_stage_to_submit];
  PreloadSubtask& subtask = stage.subtasks[stage.next_subtask_to_submit];
  std::vector<ResidencyManager::StageLoad> stage_loads;
  std::set<std::string> protected_ids;
  for (size_t movement_idx : subtask.movement_indices) {
    const auto& movement = _data_movements[movement_idx];
    if (!movement.makes_resident || movement.destination != MemoryMedium::HBM)
      continue;
    const uint64_t bytes =
        movement.resident_bytes == 0 ? movement.bytes : movement.resident_bytes;
    stage_loads.push_back(ResidencyManager::StageLoad{
        .logical_id = movement.logical_id,
        .bytes = bytes,
        .user_id = movement.user_id,
        .layer_id = movement.layer_id,
        .role = movement.role,
        .next_use_rank = residency_next_use_rank(movement.layer_id),
    });
    protected_ids.insert(movement.logical_id);
  }
  if (_residency_manager != nullptr &&
      !_residency_manager->ensure_capacity_for(
          stage_loads, protected_ids,
          _name + "." + stage.name + "." + subtask.preload_type)) {
    return {};
  }

  std::vector<MigrationRequest> requests;
  std::vector<size_t> request_movement_indices;
  requests.reserve(subtask.movement_indices.size());
  request_movement_indices.reserve(subtask.movement_indices.size());
  uint64_t submitted_physical_bytes = 0;

  for (size_t movement_idx : subtask.movement_indices) {
    const auto& movement = _data_movements[movement_idx];
    if (movement.reuse_if_resident && _residency_manager != nullptr &&
        _residency_manager->is_resident(movement.logical_id)) {
      pin_resident_use(movement.layer_id, movement.logical_id);
      Tensor* tensor = get_tensor(movement.tensor_id);
      if (tensor != nullptr && !movement.defer_tensor_ready)
        tensor->set_produced();
      continue;
    }
    MigrationRequest request;
    request.src_medium = movement.source;
    request.dst_medium = movement.destination;
    request.src_addr = movement.src_addr;
    request.dst_addr = movement.dst_addr;
    request.bytes = movement.bytes;
    request.segments.reserve(movement.segments.size());
    for (const auto& segment : movement.segments) {
      request.segments.push_back(MigrationSegment{
          .src_addr = segment.src_addr,
          .dst_addr = segment.dst_addr,
          .bytes = segment.bytes,
      });
    }
    uint64_t request_bytes = request.segments.empty() ? request.bytes : 0;
    for (const auto& segment : request.segments) request_bytes += segment.bytes;
    if (request_bytes == 0) continue;
    submitted_physical_bytes += request_bytes;
    request_movement_indices.push_back(movement_idx);
    requests.push_back(std::move(request));
  }

  if (!stage.submitted) {
    stage.submitted = true;
    stage.submitted_time_ps = now_ps;
    stage.physical_bytes = 0;
    stage.movement_ids.clear();
    stage.physical_bytes_by_type.clear();
    stage.movement_count_by_type.clear();
  }
  subtask.submitted = true;
  subtask.submitted_time_ps = now_ps;
  subtask.physical_bytes = submitted_physical_bytes;
  subtask.movement_count = static_cast<uint32_t>(requests.size());
  stage.physical_bytes += submitted_physical_bytes;
  stage.physical_bytes_by_type[subtask.preload_type] += submitted_physical_bytes;
  stage.movement_count_by_type[subtask.preload_type] +=
      static_cast<uint32_t>(requests.size());
  stage.next_subtask_to_submit++;

  std::vector<uint64_t> movement_ids =
      controller->submit_migration_requests(requests, now_ps);
  subtask.movement_ids = movement_ids;
  stage.movement_ids.insert(stage.movement_ids.end(), movement_ids.begin(),
                            movement_ids.end());
  _submitted_movement_ids.insert(_submitted_movement_ids.end(),
                                 movement_ids.begin(), movement_ids.end());

  for (size_t i = 0; i < request_movement_indices.size() &&
                     i < movement_ids.size(); ++i) {
    const auto& movement = _data_movements[request_movement_indices[i]];
    uint64_t movement_id = movement_ids[i];
    if (movement.makes_resident) {
      _resident_loads.push_back(ResidentLoad{
          .logical_id = movement.logical_id,
          .hbm_addr = movement.dst_addr,
          .bytes = movement.resident_bytes == 0 ? movement.bytes
                                                 : movement.resident_bytes,
          .movement_id = movement_id,
          .layer_id = movement.layer_id,
      });
    }
  }

  spdlog::info("[TraceModel] {} submit preload {}.{} (layer={}, "
               "movements={}, bytes={}) at {:.6f} us",
               _name, stage.name, subtask.preload_type, stage.layer_id,
               movement_ids.size(), subtask.physical_bytes,
               static_cast<double>(now_ps) / 1e6);
  return movement_ids;
}

bool TraceModel::initial_data_movement_stage_ready(
    StorageController* controller) const {
  if (!uses_layer_preload()) return data_movements_ready(controller);
  if (_preload_stages.empty()) return true;
  const auto& stage = _preload_stages.front();
  return stage.completed || !_executable_layer.empty();
}

void TraceModel::mark_stage_tensors_ready(const PreloadStage& stage) {
  for (size_t movement_idx : stage.movement_indices) {
    const auto& movement = _data_movements[movement_idx];
    Tensor* tensor = get_tensor(movement.tensor_id);
    if (tensor != nullptr) tensor->set_produced();
  }
}

void TraceModel::mark_subtask_tensors_ready(const PreloadSubtask& subtask) {
  for (size_t movement_idx : subtask.movement_indices) {
    const auto& movement = _data_movements[movement_idx];
    Tensor* tensor = get_tensor(movement.tensor_id);
    if (tensor != nullptr) tensor->set_produced();
  }
}

bool TraceModel::stage_compute_frontier_ready(
    const PreloadStage& stage) const {
  if (stage.subtasks.empty()) return true;
  const PreloadSubtask& first_subtask = stage.subtasks.front();
  return first_subtask.completed;
}

void TraceModel::refresh_executable_layers() {
  for (auto& [op_id, op] : _operation_map) {
    if (op->check_finish()) continue;
    if (check_exist_in_exeutable(op_id)) continue;
    if (op->check_executable()) _executable_layer.push_back(op.get());
  }
}

bool TraceModel::complete_ready_data_movement_stages(
    StorageController* controller, uint64_t now_ps) {
  if (!uses_layer_preload() || controller == nullptr) return false;

  bool completed_any = false;
  for (auto& stage : _preload_stages) {
    if (!stage.submitted || stage.completed) continue;
    PreloadSubtask* ready_subtask = nullptr;
    for (auto& subtask : stage.subtasks) {
      if (!subtask.submitted || subtask.completed) continue;
      if (!subtask.movement_ids.empty() &&
          !controller->movements_done(subtask.movement_ids)) {
        continue;
      }
      ready_subtask = &subtask;
      break;
    }
    if (ready_subtask == nullptr) {
      continue;
    }

    ready_subtask->completed = true;
    ready_subtask->completed_time_ps = now_ps;
    complete_data_movements(controller);
    mark_subtask_tensors_ready(*ready_subtask);
    refresh_executable_layers();
    completed_any = true;
    append_preload_type_event(stage, *ready_subtask);

    bool all_subtasks_completed = true;
    for (const auto& subtask : stage.subtasks) {
      if (!subtask.completed) {
        all_subtasks_completed = false;
        break;
      }
    }
    if (!all_subtasks_completed) continue;

    stage.completed = true;
    stage.completed_time_ps = now_ps;
    mark_stage_tensors_ready(stage);
    refresh_executable_layers();
    append_pipeline_event(
        "preload", "stage", stage.name, stage.layer_id,
        stage.submitted_time_ps, stage.completed_time_ps, stage.physical_bytes,
        "movements=" + std::to_string(stage.movement_ids.size()));
    _next_stage_to_submit++;
    spdlog::info("[TraceModel] {} complete preload stage {} at {:.6f} us "
                 "(latency={:.6f} us)",
                 _name, stage.name, static_cast<double>(now_ps) / 1e6,
                 static_cast<double>(now_ps - stage.submitted_time_ps) / 1e6);
  }
  return completed_any;
}

void TraceModel::append_preload_type_event(
    const PreloadStage& stage, const PreloadSubtask& subtask) const {
  if (subtask.physical_bytes == 0) return;
  append_pipeline_event(
      "preload", subtask.preload_type,
      stage.name + "." + subtask.preload_type, stage.layer_id,
      subtask.submitted_time_ps, subtask.completed_time_ps,
      subtask.physical_bytes,
      "movements=" + std::to_string(subtask.movement_count));
}

bool TraceModel::all_data_movement_stages_done(
    StorageController* controller) const {
  if (!uses_layer_preload()) return data_movements_ready(controller);
  for (const auto& stage : _preload_stages) {
    if (!stage.completed) return false;
  }
  return true;
}

void TraceModel::record_compute_start(uint32_t op_id, uint64_t now_ps) {
  if (_config.pipeline_breakdown_csv.empty()) return;
  auto& event = _compute_events[op_id];
  if (!event.started) {
    event.started = true;
    event.start_time_ps = now_ps;
  }
}

void TraceModel::record_compute_finish(uint32_t op_id, uint64_t now_ps) {
  if (_config.pipeline_breakdown_csv.empty()) return;
  auto event_it = _compute_events.find(op_id);
  if (event_it == _compute_events.end() || !event_it->second.started) return;
  int32_t layer_id = -1;
  auto layer_it = _operation_layer_ids.find(op_id);
  if (layer_it != _operation_layer_ids.end()) layer_id = layer_it->second;
  std::string op_name = "op" + std::to_string(op_id);
  auto name_it = _operation_names.find(op_id);
  if (name_it != _operation_names.end()) op_name = name_it->second;
  append_pipeline_event("compute", "op", op_name, layer_id,
                        event_it->second.start_time_ps, now_ps, 0,
                        "op_id=" + std::to_string(op_id));
  _compute_events.erase(event_it);

  if (layer_id >= 0) {
    auto remaining_it = _remaining_ops_by_layer.find(layer_id);
    if (remaining_it != _remaining_ops_by_layer.end() &&
        remaining_it->second > 0) {
      remaining_it->second--;
      if (remaining_it->second == 0) {
        release_layer_residency_pins(layer_id);
      }
    }
  }

  auto phase_op_it = _core_phase_events.find(op_id);
  if (phase_op_it != _core_phase_events.end()) {
    for (const auto& [phase, by_core] : phase_op_it->second) {
      for (const auto& [core_id, aggregate] : by_core) {
        if (!aggregate.started) continue;
        append_pipeline_event(
            "compute", phase, op_name + ".core" + std::to_string(core_id),
            layer_id, aggregate.start_time_ps, aggregate.end_time_ps,
            aggregate.bytes,
            "op_id=" + std::to_string(op_id) +
                ";core_id=" + std::to_string(core_id) +
                ";events=" + std::to_string(aggregate.events) +
                ";sum_duration_ps=" +
                std::to_string(aggregate.total_duration_ps));
      }
    }
    _core_phase_events.erase(phase_op_it);
  }
}

void TraceModel::record_core_phase(uint32_t op_id, const std::string& phase,
                                   uint32_t core_id, uint64_t start_ps,
                                   uint64_t end_ps, uint64_t bytes) {
  if (_config.pipeline_breakdown_csv.empty()) return;
  if (end_ps < start_ps) end_ps = start_ps;
  auto& aggregate = _core_phase_events[op_id][phase][core_id];
  if (!aggregate.started) {
    aggregate.started = true;
    aggregate.start_time_ps = start_ps;
    aggregate.end_time_ps = end_ps;
  } else {
    aggregate.start_time_ps = std::min(aggregate.start_time_ps, start_ps);
    aggregate.end_time_ps = std::max(aggregate.end_time_ps, end_ps);
  }
  aggregate.total_duration_ps += end_ps - start_ps;
  aggregate.bytes += bytes;
  aggregate.events++;
}

void TraceModel::append_pipeline_event(const std::string& pipe,
                                       const std::string& phase,
                                       const std::string& name,
                                       int32_t layer_id,
                                       uint64_t start_ps,
                                       uint64_t end_ps,
                                       uint64_t bytes,
                                       const std::string& detail) const {
  if (_config.pipeline_breakdown_csv.empty()) return;
  std::filesystem::path path(_config.pipeline_breakdown_csv);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());

  bool write_header = true;
  if (std::filesystem::exists(path))
    write_header = std::filesystem::file_size(path) == 0;

  std::ofstream out(path, std::ios::app);
  if (!out.is_open()) return;
  if (write_header) {
    out << "model,user_id,batch_id,macro_batch_id,layer_id,pipe,phase,name,"
           "start_ps,end_ps,duration_ps,start_us,end_us,duration_us,bytes,detail\n";
  }
  uint32_t user_id = _model_config.value("user_id", 0u);
  uint32_t batch_id = _model_config.value("batch_id", 0u);
  uint32_t macro_id = _model_config.value("macro_batch_id", 0u);
  out << csv_escape(_name) << ',' << user_id << ',' << batch_id << ','
      << macro_id << ',' << layer_id << ',' << pipe << ',' << phase << ','
      << csv_escape(name) << ',' << start_ps << ',' << end_ps << ','
      << (end_ps >= start_ps ? end_ps - start_ps : 0) << ','
      << static_cast<double>(start_ps) / 1e6 << ','
      << static_cast<double>(end_ps) / 1e6 << ','
      << static_cast<double>(end_ps >= start_ps ? end_ps - start_ps : 0) /
             1e6
      << ',' << bytes << ',' << csv_escape(detail) << '\n';
}

void TraceModel::prefill_ssd_tensors(Ssd* ssd) {
  if (ssd == nullptr) return;

  uint64_t prefilled_ranges = 0;
  uint64_t prefilled_bytes = 0;
  std::set<addr_type> prefilled_pages;
  const uint64_t page_bytes = std::max<uint64_t>(1, ssd->prefill_page_bytes());

  auto prefill_segment = [&](addr_type src_addr, uint64_t bytes) {
    if (bytes == 0 || !ssd->owns_address(src_addr)) return;
    addr_type first_page = ssd->align_prefill_page(src_addr);
    uint64_t last_offset = bytes > 0 ? bytes - 1 : 0;
    addr_type last_page = ssd->align_prefill_page(src_addr + last_offset);
    for (addr_type page = first_page; page <= last_page;
         page += static_cast<addr_type>(page_bytes)) {
      if (!prefilled_pages.insert(page).second) continue;
      ssd->prefill_range(page, page_bytes);
      prefilled_ranges++;
      prefilled_bytes += page_bytes;
    }
  };

  for (const auto& movement : _data_movements) {
    if (movement.source != MemoryMedium::SSD) continue;
    if (!movement.segments.empty()) {
      for (const auto& segment : movement.segments)
        prefill_segment(segment.src_addr, segment.bytes);
      continue;
    }
    prefill_segment(movement.src_addr, movement.bytes);
  }

  if (prefilled_ranges > 0) {
    spdlog::info(
        "[TraceModel] {} prefilling {} SSD source pages ({} bytes) without timing",
        _name, prefilled_ranges, prefilled_bytes);
  }
}
