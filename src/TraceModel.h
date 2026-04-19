#pragma once
#include "Model.h"
#include "frontend/trace/TraceTypes.h"
#include <string>

class TraceModel : public Model {
 public:
  TraceModel(const std::string& trace_path,
             json model_config,
             SimulationConfig config,
             const std::string& name,
             MappingTable& mapping_table);

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;

  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;

 private:
  std::string _trace_path;
  trace_frontend::TraceGraph _graph;

  uint32_t register_tensor(const trace_frontend::TensorEntry& entry, bool produced);
};
