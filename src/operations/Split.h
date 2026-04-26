#pragma once

#include "Operation.h"

class Split : public Operation {
 public:
  Split(SimulationConfig config, Model* model, std::string name,
        std::map<std::string, std::string>& attributes,
        uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void initialize_copy_tile(uint32_t output_idx, uint64_t element_offset,
                            uint64_t elements, uint64_t axis_base);
  std::vector<std::string> split_csv(const std::string& value) const;

  std::vector<uint32_t> _input_shape;
  std::vector<std::vector<uint32_t>> _output_shapes;
  std::vector<std::string> _output_names;
  uint32_t _axis = 0;
};
