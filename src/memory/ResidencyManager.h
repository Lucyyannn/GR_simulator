#pragma once

#include "../Common.h"

#include <string>
#include <unordered_map>

class ResidencyManager {
 public:
  struct Entry {
    addr_type resident_addr = 0;
    uint64_t bytes = 0;
    bool resident = false;
  };

  bool is_resident(const std::string& logical_id) const;
  addr_type resident_addr(const std::string& logical_id) const;
  addr_type reserve_destination(const std::string& logical_id,
                                uint64_t bytes,
                                MemoryMedium medium);
  void mark_resident(const std::string& logical_id,
                     addr_type resident_addr,
                     uint64_t bytes);

  addr_type source_addr(const std::string& logical_id,
                        uint64_t bytes,
                        MemoryMedium medium);

 private:
  std::unordered_map<std::string, Entry> _entries;
  std::unordered_map<std::string, addr_type> _source_addrs;
};
