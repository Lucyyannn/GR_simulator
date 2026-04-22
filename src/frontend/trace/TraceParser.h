#pragma once
#include "TraceTypes.h"
#include <string>

namespace trace_frontend {

class TraceParser {
 public:
  static TraceGraph parse(const std::string& json_path);
};

} // namespace trace_frontend
