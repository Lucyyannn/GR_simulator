#include "Tensor.h"

#include "Model.h"
#include "operations/Operation.h"

#include <algorithm>

Tensor::Tensor(uint32_t src_node, onnx::TensorProto &tensor_proto, int precision,
               bool produced = false) {
  _id = generate_id();
  _temporal = false;
  _src_node = src_node;
  _name = tensor_proto.name();
  for (int dim : tensor_proto.dims()) {
    _dims.push_back(dim);
  }
  spdlog::trace("Tensor: {}", _name);
  _produced = produced;
  _precision = precision;
  allocate_tensor(precision);
}

Tensor::Tensor(uint32_t src_node, std::string name, std::vector<uint32_t> &dims,
               int precision, bool produced = false) {
  _temporal = false;
  _id = generate_id();
  _src_node = src_node;
  _name = name;
  for (int dim : dims) {
    _dims.push_back(dim);
  }
  spdlog::trace("Tensor: {} {}", _name, dims);
  _produced = produced;
  _precision = precision;
  allocate_tensor(precision);
}

Tensor::Tensor(const Tensor &tensor) {
  _temporal = false;
  _produced = tensor._produced;
  _id = tensor._id;
  _name = tensor._name;
  _dims = tensor._dims;
  _src_node = tensor._src_node;
  _child_nodes = tensor._child_nodes;
  _address = tensor._address;
  _size = tensor._size;
  _precision = tensor._precision;
  _has_reuse_layout = tensor._has_reuse_layout;
  _reuse_axis = tensor._reuse_axis;
  _reuse_physical_rows = tensor._reuse_physical_rows;
  _reuse_row_stride_bytes = tensor._reuse_row_stride_bytes;
  _reuse_logical_to_physical = tensor._reuse_logical_to_physical;
  _memory_ready_full = tensor._memory_ready_full;
  _memory_ready_ranges = tensor._memory_ready_ranges;
}

Tensor::Tensor(uint32_t src_node, std::string name, int precision) {
  //Temproal definition, need to define
  _temporal = true;
  _id = generate_id();
  _src_node = src_node;
  _name = name;
  _precision = precision;
}

void Tensor::define_tensor(addr_type address, std::vector<uint32_t> &dims) {
  if (_dims.empty()) {
    _temporal = false;
    _address = address;
    _size = _precision;
    for (int dim : dims) {
      _dims.push_back(dim);
      _size *= dim;
    }
  } else {
    throw("Error: cannot redefine already created tensor");
  }
}

void Tensor::redefine_tensor(uint32_t id, std::vector<uint32_t> &dims) {
  if (_dims.empty()) {
    _src_node = id;
    _size = _precision;
    for (int dim : dims) {
      _dims.push_back(dim);
      _size *= dim;
    }
  } else {
    bool condition = false;
    if (_dims.size() == dims.size() && id == _src_node) {
      condition = true;
      for (int i = 0; i < _dims.size(); i++) {
        condition = condition && (_dims[i] == dims[i]);
      }
    }
    if (!condition) throw("Error: cannot redefine already created tensor");
  }
}

void Tensor::resize_tensor(std::vector<uint32_t> &dims) {
  _dims.clear();
  _size = _precision;
  for (int dim : dims) {
    _dims.push_back(dim);
    _size *= dim;
  }
}

void Tensor::add_child_node(Operation *op) {
  _child_nodes.push_back(op->get_id());
}

void Tensor::set_reuse_layout(
    uint32_t axis, uint32_t physical_rows, uint64_t row_stride_bytes,
    const std::vector<uint32_t>& logical_to_physical) {
  _has_reuse_layout = true;
  _reuse_axis = axis;
  _reuse_physical_rows = physical_rows;
  _reuse_row_stride_bytes = row_stride_bytes;
  _reuse_logical_to_physical = logical_to_physical;
}

void Tensor::set_memory_pending() {
  _memory_ready_full = false;
  _memory_ready_ranges.clear();
}

void Tensor::mark_full_memory_ready() {
  _memory_ready_full = true;
  _memory_ready_ranges.clear();
}

void Tensor::mark_memory_ready(addr_type address, uint64_t bytes) {
  if (bytes == 0 || _memory_ready_full) return;
  addr_type tensor_begin = _address;
  addr_type tensor_end = _address + _size;
  addr_type ready_begin = std::max(address, tensor_begin);
  addr_type ready_end = std::min<addr_type>(address + bytes, tensor_end);
  if (ready_begin >= ready_end) return;

  _memory_ready_ranges.push_back({ready_begin, ready_end});
  std::sort(_memory_ready_ranges.begin(), _memory_ready_ranges.end());
  std::vector<std::pair<addr_type, addr_type>> merged;
  for (const auto& range : _memory_ready_ranges) {
    if (merged.empty() || range.first > merged.back().second) {
      merged.push_back(range);
    } else {
      merged.back().second = std::max(merged.back().second, range.second);
    }
  }
  _memory_ready_ranges = std::move(merged);
  if (_memory_ready_ranges.size() == 1 &&
      _memory_ready_ranges.front().first <= tensor_begin &&
      _memory_ready_ranges.front().second >= tensor_end) {
    mark_full_memory_ready();
  }
}

bool Tensor::memory_ready(addr_type address, uint64_t bytes) const {
  if (_memory_ready_full) return true;
  if (bytes == 0) return true;
  addr_type request_begin = address;
  addr_type request_end = address + bytes;
  for (const auto& range : _memory_ready_ranges) {
    if (range.first <= request_begin && range.second >= request_end)
      return true;
  }
  return false;
}

bool Tensor::memory_ready() const {
  if (_memory_ready_full) return true;
  return _memory_ready_ranges.size() == 1 &&
         _memory_ready_ranges.front().first <= _address &&
         _memory_ready_ranges.front().second >= _address + _size;
}

void Tensor::allocate_tensor(int precision) {
  uint32_t size = 1;
  for (auto dim : _dims) {
    size *= dim;
  }
  uint32_t total_bytes = size * precision;
  MemoryMedium medium = default_tensor_medium(total_bytes);
  _address = allocate_address_in_medium(total_bytes, medium);
  _size = total_bytes;
  const char* medium_name = "HBM";
  if (medium == MemoryMedium::DDR) medium_name = "DDR";
  if (medium == MemoryMedium::SSD) medium_name = "SSD";
  spdlog::debug("[TENSOR] {} ({} B) placed in {} at 0x{:x}",
                _name, total_bytes, medium_name, _address);
}

void Tensor::relocate(MemoryMedium medium) {
  _address = allocate_address_in_medium(static_cast<uint32_t>(_size), medium);
  mark_full_memory_ready();
  const char* medium_name = "HBM";
  if (medium == MemoryMedium::DDR) medium_name = "DDR";
  if (medium == MemoryMedium::SSD) medium_name = "SSD";
  spdlog::debug("[TENSOR] {} relocated to {} at 0x{:x}",
                _name, medium_name, _address);
}

void Tensor::print_tensor() {
  spdlog::info("Tensor: {} {} {} {}", _name, _src_node, _dims, _size);
}
