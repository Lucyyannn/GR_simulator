#pragma once

#include "Operation.h"

class Embedding : public Operation {
 public:
  Embedding(SimulationConfig config, Model* model, onnx::NodeProto& node_proto,
            uint32_t target_core = 0);
  Embedding(SimulationConfig config, Model* model, std::string name,
            std::map<std::string, std::string>& attrs,
            uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void infer_output_shape();
  void calculate_loops();
  void initialize_instructions(Tile* tile, Mapping mapping, uint32_t lookup_offset,
                               uint32_t lookups);
  uint32_t get_dtype_bytes(const std::string& dtype) const;
  uint32_t get_lookup_row(uint32_t lookup_idx) const;

  std::vector<uint32_t> _indices_shape;
  std::vector<uint32_t> _weight_shape;
  std::vector<uint32_t> _output_shape;
  std::vector<uint32_t> _indices_values;

  uint32_t _num_lookups = 0;
  uint32_t _embedding_dim = 0;
  uint32_t _vocab_size = 0;
  uint32_t _rows_per_tile = 1;
  uint32_t _index_element_bytes = 4;
  uint32_t _weight_input_idx = 0;
  uint32_t _indices_input_idx = 1;
  bool _preloaded_rows = false;
};
