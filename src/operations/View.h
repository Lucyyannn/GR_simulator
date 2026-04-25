#pragma once

#include "Operation.h"

class View : public Operation {
 public:
  View(SimulationConfig config, Model* model, std::string name,
       std::map<std::string, std::string>& attributes,
       uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  std::vector<uint32_t> _output_shape;
  std::string _output_name;
};
