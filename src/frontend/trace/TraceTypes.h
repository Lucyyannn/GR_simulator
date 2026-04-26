#pragma once
#include <string>
#include <vector>
#include <map>

namespace trace_frontend {

struct TensorEntry {
  std::string name;
  std::vector<uint32_t> shape;
  std::string dtype;
  bool is_weight = false;
  std::string logical_id;
  std::string source_logical_id;
  std::vector<uint32_t> source_shape;
  std::vector<uint32_t> indices_values;
  std::string reuse_mode;
  uint32_t reuse_axis = 0;
  uint32_t reuse_physical_rows = 0;
  std::vector<uint32_t> reuse_logical_to_physical;
  std::string role;
  std::string initial_medium;
  std::string runtime_medium;
  uint32_t layer_id = 0;
  uint32_t user_id = 0;
  uint32_t batch_id = 0;
  uint32_t macro_batch_id = 0;
};

struct OpEntry {
  uint32_t id;
  std::string name;
  std::vector<TensorEntry> inputs;
  std::vector<TensorEntry> outputs;
  std::map<std::string, std::string> attrs;
};

struct TraceMetadata {
  std::string format_version;
  std::string model_name;
  std::string layout;
  bool fail_on_unknown_op = false;
  bool baseline_preload = false;
  bool pipeline_enabled = false;
  bool kv_reuse_enabled = false;
  std::map<std::string, std::string> op_modeling;
};

struct TraceGraph {
  TraceMetadata metadata;
  std::vector<OpEntry> operators;
};

} // namespace trace_frontend
