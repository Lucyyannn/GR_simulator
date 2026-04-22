#include "TraceParser.h"

#include <fstream>
#include <spdlog/spdlog.h>
#include "../../Common.h"

namespace trace_frontend {

namespace {

std::string json_array_to_string(const nlohmann::json& arr) {
  if (!arr.is_array() || arr.empty())
    return "";
  std::string result = std::to_string(arr[0].get<uint32_t>());
  for (size_t i = 1; i < arr.size(); i++)
    result += "," + std::to_string(arr[i].get<uint32_t>());
  return result;
}

TensorEntry parse_tensor(const nlohmann::json& j) {
  TensorEntry entry;
  entry.name = j.value("name", "");
  if (j.contains("shape") && j["shape"].is_array()) {
    for (auto& dim : j["shape"])
      entry.shape.push_back(dim.get<uint32_t>());
  }
  entry.dtype = j.value("dtype", "float16");
  entry.is_weight = j.value("is_weight", false);
  return entry;
}

OpEntry parse_op(const nlohmann::json& j) {
  OpEntry entry;
  entry.id = j.value("id", 0);
  entry.name = j.value("name", "");

  if (j.contains("inputs") && j["inputs"].is_array()) {
    for (auto& inp : j["inputs"])
      entry.inputs.push_back(parse_tensor(inp));
  }
  if (j.contains("outputs") && j["outputs"].is_array()) {
    for (auto& out : j["outputs"])
      entry.outputs.push_back(parse_tensor(out));
  }
  if (j.contains("attrs") && j["attrs"].is_object()) {
    for (auto it = j["attrs"].begin(); it != j["attrs"].end(); ++it) {
      if (it->is_array()) {
        entry.attrs[it.key()] = json_array_to_string(*it);
      } else if (it->is_number_integer()) {
        entry.attrs[it.key()] = std::to_string(it->get<int>());
      } else if (it->is_number_float()) {
        entry.attrs[it.key()] = std::to_string(it->get<double>());
      } else if (it->is_string()) {
        entry.attrs[it.key()] = it->get<std::string>();
      } else if (it->is_boolean()) {
        entry.attrs[it.key()] = it->get<bool>() ? "1" : "0";
      }
    }
  }
  return entry;
}

} // anonymous namespace

TraceGraph TraceParser::parse(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    spdlog::error("[TraceParser] Cannot open trace file: {}", json_path);
    exit(EXIT_FAILURE);
  }

  nlohmann::json root;
  try {
    file >> root;
  } catch (const nlohmann::json::parse_error& e) {
    spdlog::error("[TraceParser] JSON parse error in {}: {}", json_path, e.what());
    exit(EXIT_FAILURE);
  }

  TraceGraph graph;

  if (root.contains("metadata")) {
    auto& meta = root["metadata"];
    graph.metadata.format_version = meta.value("format_version", "1.0");
    graph.metadata.model_name = meta.value("model_name", "unknown");
    graph.metadata.layout = meta.value("layout", "NHWC");
  }

  if (root.contains("operators") && root["operators"].is_array()) {
    for (auto& op : root["operators"])
      graph.operators.push_back(parse_op(op));
  }

  spdlog::info("[TraceParser] Parsed {} operators from {} (model={}, layout={})",
               graph.operators.size(), json_path,
               graph.metadata.model_name, graph.metadata.layout);
  return graph;
}

} // namespace trace_frontend
