#pragma once
#include "Common.h"

class Model;
class Operation;

class Tensor {
 public:
  Tensor(uint32_t src_node, onnx::TensorProto &tensor_proto, int precision, bool produced);
  Tensor(uint32_t src_node, std::string name, std::vector<uint32_t> &dims,
         int precision, bool produced);
  Tensor(uint32_t src_node, std::string name, int precision);
  Tensor(const Tensor &tensor);

  void define_tensor(addr_type address, std::vector<uint32_t> &dims);
  void redefine_tensor(uint32_t src_node, std::vector<uint32_t> &dims);
  void resize_tensor(std::vector<uint32_t> &dims);
  void add_child_node(Operation *op);

  uint32_t get_id() { return _id; }
  std::string get_name() { return _name; }
  uint32_t get_src_node() { return _src_node; }
  std::vector<uint32_t> get_dims() { return _dims; }
  bool has_reuse_layout() const { return _has_reuse_layout; }
  uint32_t reuse_axis() const { return _reuse_axis; }
  uint32_t reuse_physical_rows() const { return _reuse_physical_rows; }
  uint64_t reuse_row_stride_bytes() const { return _reuse_row_stride_bytes; }
  const std::vector<uint32_t>& reuse_logical_to_physical() const {
    return _reuse_logical_to_physical;
  }
  bool has_group_layout() const { return _has_group_layout; }
  uint32_t group_axis() const { return _group_axis; }
  uint32_t group_row_axis() const { return _group_row_axis; }
  uint64_t group_row_stride_bytes() const { return _group_row_stride_bytes; }
  const std::vector<addr_type>& group_base_addrs() const {
    return _group_base_addrs;
  }
  const std::vector<uint32_t>& group_physical_rows() const {
    return _group_physical_rows;
  }
  const std::vector<std::vector<uint32_t>>& group_logical_to_physical() const {
    return _group_logical_to_physical;
  }
  void set_reuse_layout(uint32_t axis, uint32_t physical_rows,
                        uint64_t row_stride_bytes,
                        const std::vector<uint32_t>& logical_to_physical);
  void set_group_layout(uint32_t group_axis, uint32_t row_axis,
                        uint64_t row_stride_bytes,
                        const std::vector<addr_type>& group_base_addrs,
                        const std::vector<uint32_t>& group_physical_rows,
                        const std::vector<std::vector<uint32_t>>& logical_to_physical);
  addr_type physical_address(const std::vector<uint32_t>& coords,
                             uint32_t precision) const;
  void set_produced() { _produced = true; }
  void clear_produced() { _produced = false; }
  bool get_produced() { return _produced; }
  uint32_t num_child_nodes() { return _child_nodes.size(); }
  uint32_t get_child_node(uint32_t id) { return _child_nodes[id]; }

	  void allocate_tensor(int precision);
	  void relocate(MemoryMedium medium);
	  void relocate(MemoryMedium medium, uint32_t npu_id);
	  void set_address(addr_type address) { _address = address; }
	  addr_type get_address() { return _address; }
  uint64_t get_size() { return _size; }
  void print_tensor();

 private:
  bool _temporal;
  uint32_t _precision;
  bool _produced;
  uint32_t _id;
  std::string _name;
  std::vector<uint32_t> _dims;
  uint32_t _src_node;
  std::vector<uint32_t> _child_nodes;
  addr_type _address;
  uint64_t _size;
  bool _has_reuse_layout = false;
  uint32_t _reuse_axis = 0;
  uint32_t _reuse_physical_rows = 0;
  uint64_t _reuse_row_stride_bytes = 0;
  std::vector<uint32_t> _reuse_logical_to_physical;
  bool _has_group_layout = false;
  uint32_t _group_axis = 0;
  uint32_t _group_row_axis = 0;
  uint64_t _group_row_stride_bytes = 0;
  std::vector<addr_type> _group_base_addrs;
  std::vector<uint32_t> _group_physical_rows;
  std::vector<std::vector<uint32_t>> _group_logical_to_physical;
  friend Model;
};
