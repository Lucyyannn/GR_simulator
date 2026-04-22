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
};

struct TraceGraph {
  TraceMetadata metadata;
  std::vector<OpEntry> operators;
};

} // namespace trace_frontend
