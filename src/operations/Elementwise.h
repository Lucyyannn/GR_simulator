#pragma once

#include "Operation.h"

class Elementwise : public Operation {
 public:
  Elementwise(SimulationConfig config, Model* model, std::string name,
              std::map<std::string, std::string>& attributes,
              uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void initialize_instructions(Tile* tile, uint32_t element_offset,
                               uint32_t elements);
  void calculate_loops();

  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _output_shape;
  uint32_t _num_elements = 0;
  uint32_t _elements_per_tile = 1;
  Opcode _opcode = Opcode::MUL;
};
