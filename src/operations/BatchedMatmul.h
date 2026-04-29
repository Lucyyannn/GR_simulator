#pragma once

#include "Gemm.h"

class BatchedMatmul : public Gemm {
 public:
  BatchedMatmul(SimulationConfig config, Model* model, std::string name,
                std::map<std::string, std::string>& attributes,
                uint32_t target_core);
  virtual void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  virtual void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  void validate_shapes() const;

 private:
  uint32_t _batch_dim = 0;
  uint32_t _rows_per_batch = 0;
};
