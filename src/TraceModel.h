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
  virtual std::vector<uint64_t> submit_data_movements(
      StorageController* controller, uint64_t now_ps) override;
  virtual bool data_movements_ready(StorageController* controller) const override;
  virtual void complete_data_movements(StorageController* controller) override;
  virtual uint64_t prepare_baseline_storage(StorageController* controller,
                                            uint64_t now_ps) override;

	 private:
  struct PlannedDataMovement {
    std::string tensor_name;
    std::string logical_id;
    std::string role;
    MemoryMedium source = MemoryMedium::UNKNOWN;
    MemoryMedium destination = MemoryMedium::UNKNOWN;
    addr_type src_addr = 0;
    addr_type dst_addr = 0;
    uint64_t bytes = 0;
    uint32_t batch_id = 0;
    uint32_t macro_batch_id = 0;
    uint32_t user_id = 0;
    bool makes_resident = false;
    uint64_t resident_bytes = 0;
  };

  struct ResidentLoad {
    std::string logical_id;
    addr_type hbm_addr = 0;
    uint64_t bytes = 0;
    uint64_t movement_id = 0;
  };

	  std::string _trace_path;
	  trace_frontend::TraceGraph _graph;
  std::map<std::string, trace_frontend::TensorEntry> _tensor_entries;
  std::vector<PlannedDataMovement> _data_movements;
  std::vector<uint64_t> _submitted_movement_ids;
  std::vector<ResidentLoad> _resident_loads;
  bool _data_movements_submitted = false;
  uint64_t _reuse_logical_bytes = 0;
  uint64_t _reuse_physical_bytes = 0;

  uint32_t register_tensor(const trace_frontend::TensorEntry& entry, bool produced);
  void remember_tensor_entry(const trace_frontend::TensorEntry& entry);
  void apply_trace_storage(Tensor* tensor, const trace_frontend::TensorEntry& entry);
  bool apply_reuse_layout(Tensor* tensor,
                          const trace_frontend::TensorEntry& entry);
  std::string effective_logical_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_user_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_batch_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_macro_batch_id(const trace_frontend::TensorEntry& entry) const;
};
