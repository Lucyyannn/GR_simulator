#pragma once

#include "../Common.h"

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class ResidencyManager {
 public:
  struct Entry {
    addr_type resident_addr = 0;
    uint64_t bytes = 0;
    bool resident = false;
    bool loading = false;
    uint32_t pin_count = 0;
    uint32_t user_id = 0;
    int32_t layer_id = -1;
    std::string role;
    int64_t next_use_rank = 0;
    uint64_t last_touch = 0;
  };

  struct StageLoad {
    std::string logical_id;
    uint64_t bytes = 0;
    uint32_t user_id = 0;
    int32_t layer_id = -1;
    std::string role;
    int64_t next_use_rank = 0;
  };

  void configure_capacity(uint64_t capacity_bytes);
  bool is_resident(const std::string& logical_id) const;
  addr_type resident_addr(const std::string& logical_id) const;
  addr_type reserve_destination(const std::string& logical_id,
                                uint64_t bytes,
                                MemoryMedium medium);
  void note_entry(const std::string& logical_id,
                  uint64_t bytes,
                  uint32_t user_id,
                  int32_t layer_id,
                  const std::string& role,
                  int64_t next_use_rank);
  bool ensure_capacity_for(const std::vector<StageLoad>& loads,
                           const std::set<std::string>& protected_ids,
                           const std::string& reason);
  void mark_resident(const std::string& logical_id,
                     addr_type resident_addr,
                     uint64_t bytes);
  void pin(const std::string& logical_id);
  void unpin(const std::string& logical_id);
  void release(const std::string& logical_id);
  uint64_t used_bytes() const { return _used_bytes; }
  uint64_t capacity_bytes() const { return _capacity_bytes; }

  addr_type source_addr(const std::string& logical_id,
                        uint64_t bytes,
                        MemoryMedium medium);

 private:
  std::unordered_map<std::string, Entry> _entries;
  std::unordered_map<std::string, addr_type> _source_addrs;
  uint64_t _capacity_bytes = 0;
  uint64_t _used_bytes = 0;
  uint64_t _clock = 0;
  std::unordered_set<std::string> _blocked_reasons;

  bool capacity_enabled() const { return _capacity_bytes > 0; }
  bool evict_one(const std::set<std::string>& protected_ids);
};
