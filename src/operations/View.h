#pragma once

#include "Operation.h"

class View : public Operation {
 public:
  View(SimulationConfig config, Model* model, std::string name,
       std::map<std::string, std::string>& attributes,
       uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void initialize_copy_tile(uint64_t element_offset, uint64_t elements);
  uint64_t input_element_for_output(uint64_t output_element) const;
  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _output_shape;
  std::vector<uint32_t> _dims;
  std::string _output_name;
};
