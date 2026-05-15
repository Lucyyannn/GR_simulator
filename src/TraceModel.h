#pragma once
#include "Model.h"
#include "frontend/trace/TraceTypes.h"
#include <string>
#include <map>
#include <set>
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
  virtual bool uses_layer_preload() const override;
  virtual bool has_data_movement_stage_to_submit() const override;
  virtual bool initial_data_movement_stage_ready(
      StorageController* controller) const override;
  virtual std::vector<uint64_t> submit_next_data_movement_stage(
      StorageController* controller, uint64_t now_ps) override;
  virtual bool complete_ready_data_movement_stages(
      StorageController* controller, uint64_t now_ps) override;
  virtual bool all_data_movement_stages_done(
      StorageController* controller) const override;
	  virtual void record_compute_start(uint32_t op_id, uint64_t now_ps) override;
	  virtual void record_compute_finish(uint32_t op_id, uint64_t now_ps) override;
	  virtual void record_core_phase(uint32_t op_id, const std::string& phase,
	                                 uint32_t core_id, uint64_t start_ps,
	                                 uint64_t end_ps, uint64_t bytes) override;
	  virtual void release_residency_pins() override;

	 private:
  struct PlannedDataMovement {
    struct Segment {
      addr_type src_addr = 0;
      addr_type dst_addr = 0;
      uint64_t bytes = 0;
    };

    std::string tensor_name;
    std::string logical_id;
    std::string role;
    std::string preload_group;
    MemoryMedium source = MemoryMedium::UNKNOWN;
    MemoryMedium destination = MemoryMedium::UNKNOWN;
    addr_type src_addr = 0;
    addr_type dst_addr = 0;
    uint64_t bytes = 0;
    uint32_t batch_id = 0;
    uint32_t macro_batch_id = 0;
    uint32_t user_id = 0;
    int32_t layer_id = -1;
    uint32_t tensor_id = 0;
    bool makes_resident = false;
    bool reuse_if_resident = false;
    bool defer_tensor_ready = false;
    uint64_t resident_bytes = 0;
    std::vector<Segment> segments;
  };

	  struct ResidentLoad {
	    std::string logical_id;
	    addr_type hbm_addr = 0;
	    uint64_t bytes = 0;
	    uint64_t movement_id = 0;
	    int32_t layer_id = -1;
	    bool completed = false;
	  };

  struct PreloadSubtask {
    std::string preload_type;
    std::vector<size_t> movement_indices;
    std::vector<uint64_t> movement_ids;
    uint64_t submitted_time_ps = 0;
    uint64_t completed_time_ps = 0;
    uint64_t physical_bytes = 0;
    uint32_t movement_count = 0;
    bool submitted = false;
    bool completed = false;
  };

  struct PreloadStage {
    int32_t layer_id = -1;
    std::string name;
    std::vector<size_t> movement_indices;
    std::vector<uint64_t> movement_ids;
    std::vector<PreloadSubtask> subtasks;
    size_t next_subtask_to_submit = 0;
    uint64_t submitted_time_ps = 0;
    uint64_t completed_time_ps = 0;
    uint64_t physical_bytes = 0;
    std::map<std::string, uint64_t> physical_bytes_by_type;
    std::map<std::string, uint32_t> movement_count_by_type;
    bool submitted = false;
    bool completed = false;
  };

	  struct ComputeEvent {
	    uint64_t start_time_ps = 0;
	    bool started = false;
	  };

	  struct CorePhaseAggregate {
	    uint64_t start_time_ps = 0;
	    uint64_t end_time_ps = 0;
	    uint64_t total_duration_ps = 0;
	    uint64_t bytes = 0;
	    uint64_t events = 0;
	    bool started = false;
	  };

	  std::string _trace_path;
	  trace_frontend::TraceGraph _graph;
  std::map<std::string, trace_frontend::TensorEntry> _tensor_entries;
  std::vector<PlannedDataMovement> _data_movements;
  std::vector<uint64_t> _submitted_movement_ids;
  std::vector<ResidentLoad> _resident_loads;
  std::vector<PreloadStage> _preload_stages;
  size_t _next_stage_to_submit = 0;
  std::map<uint32_t, int32_t> _operation_layer_ids;
	  std::map<uint32_t, std::string> _operation_names;
	  std::map<uint32_t, ComputeEvent> _compute_events;
	  std::map<uint32_t, std::map<std::string, std::map<uint32_t, CorePhaseAggregate>>>
	      _core_phase_events;
	  std::map<int32_t, std::set<std::string>> _resident_uses_by_layer;
	  std::map<int32_t, uint32_t> _remaining_ops_by_layer;
  bool _data_movements_submitted = false;
  bool _graph_sharded = false;
  uint64_t _reuse_logical_bytes = 0;
  uint64_t _reuse_physical_bytes = 0;

  uint32_t register_tensor(const trace_frontend::TensorEntry& entry, bool produced);
  void apply_npu_trace_shard();
  void remember_tensor_entry(const trace_frontend::TensorEntry& entry);
  void apply_trace_storage(Tensor* tensor, const trace_frontend::TensorEntry& entry);
  bool apply_reuse_layout(Tensor* tensor,
                          const trace_frontend::TensorEntry& entry);
  std::string effective_logical_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_user_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_batch_id(const trace_frontend::TensorEntry& entry) const;
  uint32_t effective_macro_batch_id(const trace_frontend::TensorEntry& entry) const;
	  bool layer_preload_enabled() const;
	  int32_t stage_layer_for_movement(const PlannedDataMovement& movement) const;
	  std::string preload_type_for_role(const std::string& role) const;
	  std::string preload_type_for_movement(
	      const PlannedDataMovement& movement) const;
	  int64_t residency_next_use_rank(int32_t layer_id) const;
	  void note_residency_entry(const PlannedDataMovement& movement) const;
	  void pin_resident_use(int32_t layer_id, const std::string& logical_id);
	  void release_layer_residency_pins(int32_t layer_id);
	  void build_preload_stages();
  std::set<MemoryMedium> preload_subtask_sources(
      const PreloadSubtask& subtask) const;
  bool preload_subtask_can_submit(const PreloadStage& stage,
                                  size_t subtask_idx) const;
  size_t next_ready_preload_subtask(const PreloadStage& stage) const;
  std::vector<uint64_t> submit_preload_subtask(PreloadStage& stage,
                                               size_t subtask_idx,
                                               StorageController* controller,
                                               uint64_t now_ps);
	  void append_preload_type_event(const PreloadStage& stage,
	                                 const PreloadSubtask& subtask) const;
  void mark_stage_tensors_ready(const PreloadStage& stage);
  void mark_subtask_tensors_ready(const PreloadSubtask& subtask);
  bool stage_compute_frontier_ready(const PreloadStage& stage) const;
  void refresh_executable_layers();
  void append_pipeline_event(const std::string& pipe,
                             const std::string& phase,
                             const std::string& name,
                             int32_t layer_id,
                             uint64_t start_ps,
                             uint64_t end_ps,
                             uint64_t bytes,
                             const std::string& detail) const;
};
