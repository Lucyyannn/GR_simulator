#pragma once
#include "TraceTypes.h"
#include <string>
#include <map>

namespace trace_frontend {

struct ConvertedOp {
  std::string optype;
  std::map<std::string, std::string> attrs;
};

class TraceOpConverter {
 public:
  static ConvertedOp convert(const OpEntry& entry);

 private:
  static ConvertedOp convert_conv2d(const OpEntry& entry);
  static ConvertedOp convert_linear(const OpEntry& entry);
  static ConvertedOp convert_matmul(const OpEntry& entry);
  static ConvertedOp convert_addmm(const OpEntry& entry);
  static ConvertedOp convert_max_pool2d(const OpEntry& entry);
  static ConvertedOp convert_adaptive_avg_pool2d(const OpEntry& entry);
  static ConvertedOp convert_avg_pool2d(const OpEntry& entry);
  static ConvertedOp convert_flatten(const OpEntry& entry);
  static ConvertedOp convert_layer_norm(const OpEntry& entry);
  static ConvertedOp convert_gelu(const OpEntry& entry);
  static ConvertedOp convert_silu(const OpEntry& entry);
  static ConvertedOp convert_softmax(const OpEntry& entry);
  static ConvertedOp convert_dummy(const OpEntry& entry);

  static std::string shape_to_str(const std::vector<uint32_t>& shape);
  static std::vector<uint32_t> parse_dims(const std::string& s);
};

} // namespace trace_frontend
