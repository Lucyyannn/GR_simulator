#pragma once

#include "Operation.h"

#include <set>

class HSTUAttention : public Operation {
 public:
  HSTUAttention(SimulationConfig config, Model* model, std::string name,
                std::map<std::string, std::string>& attributes,
                uint32_t target_core = 0);
  HSTUAttention(const HSTUAttention& src);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void calculate_tiles();
  void initialize_dense_input_tiles(uint32_t input_idx);
  void initialize_kv_cache_tiles(uint32_t input_idx);
  void initialize_output_compute_tiles();
  void append_gemm_compute(Tile* tile, uint32_t n, uint32_t c, uint32_t m,
                           addr_type dest_addr);
  void append_silu_compute(Tile* tile, uint64_t score_elements);
  void initialize_movin_compute_tile(uint32_t operand_id,
                                     const std::set<addr_type>& input_addrs,
                                     uint32_t logical_request_count,
                                     uint32_t compute_size);
  void collect_dense_addresses(Tensor* tensor, uint64_t element_offset,
                               uint64_t elements,
                               std::set<addr_type>& addrs);
  void collect_row_addresses(Tensor* tensor, uint32_t row_start,
                             uint32_t rows, std::set<addr_type>& physical_addrs,
                             std::set<addr_type>& logical_addrs);
  void collect_output_addresses(uint64_t element_offset, uint64_t elements,
                                std::set<addr_type>& addrs);

  std::vector<uint32_t> _q_shape;
  std::vector<uint32_t> _k_shape;
  std::vector<uint32_t> _v_shape;
  std::vector<uint32_t> _k_cache_shape;
  std::vector<uint32_t> _v_cache_shape;
  std::vector<uint32_t> _output_shape;
  uint32_t _kv_axis = 1;
  uint32_t _logical_kv_len = 0;
  uint32_t _current_tokens = 0;
  uint32_t _hidden = 0;
  uint64_t _attention_score_elements = 0;
  uint64_t _dense_elements_per_tile = 1;
  uint32_t _kv_rows_per_tile = 1;
  uint64_t _output_elements_per_tile = 1;
};
