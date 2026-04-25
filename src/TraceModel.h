#pragma once
#include "Model.h"
#include "frontend/trace/TraceTypes.h"
#include <string>
#include <map>
#include <vector>

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

  virtual void prefill_ssd_tensors(Ssd* ssd) override;
  virtual uint64_t prepare_baseline_storage(StorageController* controller,
                                            uint64_t now_ps) override;

	 private:
  struct BaselineMigration {
    std::string tensor_name;
    MemoryMedium source = MemoryMedium::UNKNOWN;
    MemoryMedium destination = MemoryMedium::UNKNOWN;
    addr_type src_addr = 0;
    addr_type dst_addr = 0;
    uint64_t bytes = 0;
  };

	  std::string _trace_path;
	  trace_frontend::TraceGraph _graph;
  std::map<std::string, trace_frontend::TensorEntry> _tensor_entries;
  std::vector<BaselineMigration> _baseline_migrations;

	  uint32_t register_tensor(const trace_frontend::TensorEntry& entry, bool produced);
  void remember_tensor_entry(const trace_frontend::TensorEntry& entry);
  void apply_trace_storage(Tensor* tensor, const trace_frontend::TensorEntry& entry);
};
