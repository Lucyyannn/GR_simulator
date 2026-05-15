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
  _has_group_layout = tensor._has_group_layout;
  _group_axis = tensor._group_axis;
  _group_row_axis = tensor._group_row_axis;
  _group_row_stride_bytes = tensor._group_row_stride_bytes;
  _group_base_addrs = tensor._group_base_addrs;
  _group_physical_rows = tensor._group_physical_rows;
  _group_logical_to_physical = tensor._group_logical_to_physical;
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

void Tensor::set_group_layout(
    uint32_t group_axis, uint32_t row_axis, uint64_t row_stride_bytes,
    const std::vector<addr_type>& group_base_addrs,
    const std::vector<uint32_t>& group_physical_rows,
    const std::vector<std::vector<uint32_t>>& logical_to_physical) {
  _has_group_layout = true;
  _group_axis = group_axis;
  _group_row_axis = row_axis;
  _group_row_stride_bytes = row_stride_bytes;
  _group_base_addrs = group_base_addrs;
  _group_physical_rows = group_physical_rows;
  _group_logical_to_physical = logical_to_physical;
}

addr_type Tensor::physical_address(const std::vector<uint32_t>& coords,
                                   uint32_t precision) const {
  if (!_has_group_layout || coords.size() != _dims.size() ||
      _group_axis >= coords.size() || _group_row_axis >= coords.size()) {
    addr_type offset = 0;
    for (size_t dim = 0; dim < coords.size(); ++dim)
      offset = offset * _dims[dim] + coords[dim];
    return _address + offset * static_cast<addr_type>(precision);
  }

  uint32_t group = coords[_group_axis];
  if (group >= _group_base_addrs.size()) return _address;
  uint32_t logical_row = coords[_group_row_axis];
  uint32_t physical_row = logical_row;
  if (group < _group_logical_to_physical.size() &&
      !_group_logical_to_physical[group].empty() &&
      logical_row < _group_logical_to_physical[group].size()) {
    physical_row = _group_logical_to_physical[group][logical_row];
  }

  uint64_t inner_index = 0;
  for (size_t dim = 0; dim < coords.size(); ++dim) {
    if (dim == _group_axis || dim == _group_row_axis) continue;
    inner_index = inner_index * std::max<uint32_t>(_dims[dim], 1) + coords[dim];
  }

  uint64_t dense_row_bytes = static_cast<uint64_t>(precision);
  for (size_t dim = 0; dim < _dims.size(); ++dim) {
    if (dim == _group_axis || dim == _group_row_axis) continue;
    dense_row_bytes *= std::max<uint32_t>(_dims[dim], 1);
  }
  const uint64_t row_stride =
      _group_row_stride_bytes == 0 ? dense_row_bytes : _group_row_stride_bytes;
  return _group_base_addrs[group] +
         static_cast<addr_type>(physical_row) * row_stride +
         inner_index * static_cast<addr_type>(precision);
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
  const char* medium_name = "HBM";
  if (medium == MemoryMedium::DDR) medium_name = "DDR";
  if (medium == MemoryMedium::SSD) medium_name = "SSD";
  spdlog::debug("[TENSOR] {} relocated to {} at 0x{:x}",
                _name, medium_name, _address);
}

void Tensor::relocate(MemoryMedium medium, uint32_t npu_id) {
  _address =
      allocate_address_in_medium_for_npu(static_cast<uint32_t>(_size), medium,
                                         npu_id);
  const char* medium_name = "HBM";
  if (medium == MemoryMedium::DDR) medium_name = "DDR";
  if (medium == MemoryMedium::SSD) medium_name = "SSD";
  spdlog::debug("[TENSOR] {} relocated to {} npu={} at 0x{:x}",
                _name, medium_name, npu_id, _address);
}

void Tensor::print_tensor() {
  spdlog::info("Tensor: {} {} {} {}", _name, _src_node, _dims, _size);
}
