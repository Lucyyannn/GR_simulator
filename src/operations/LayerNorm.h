#pragma once

#include "Operation.h"

class LayerNorm : public Operation {
 public:
  LayerNorm(SimulationConfig config, Model* model, std::string name,
            std::map<std::string, std::string>& attributes,
            uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void initialize_instructions(Tile* tile, uint32_t token_offset,
                               uint32_t tokens);
  void calculate_loops();

  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _output_shape;
  uint32_t _tokens = 0;
  uint32_t _hidden = 0;
  uint32_t _tokens_per_tile = 1;
};
