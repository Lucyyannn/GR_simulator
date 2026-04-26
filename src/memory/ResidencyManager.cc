#include "ResidencyManager.h"

bool ResidencyManager::is_resident(const std::string& logical_id) const {
  auto it = _entries.find(logical_id);
  return it != _entries.end() && it->second.resident;
}

addr_type ResidencyManager::resident_addr(const std::string& logical_id) const {
  auto it = _entries.find(logical_id);
  if (it == _entries.end()) return 0;
  return it->second.resident_addr;
}

addr_type ResidencyManager::reserve_destination(const std::string& logical_id,
                                                uint64_t bytes,
                                                MemoryMedium medium) {
  auto& entry = _entries[logical_id];
  if (entry.resident_addr == 0) {
    entry.resident_addr =
        allocate_address_in_medium(static_cast<uint32_t>(bytes), medium);
    entry.bytes = bytes;
  }
  return entry.resident_addr;
}

void ResidencyManager::mark_resident(const std::string& logical_id,
                                     addr_type resident_addr,
                                     uint64_t bytes) {
  auto& entry = _entries[logical_id];
  entry.resident_addr = resident_addr;
  entry.bytes = bytes;
  entry.resident = true;
}

addr_type ResidencyManager::source_addr(const std::string& logical_id,
                                        uint64_t bytes,
                                        MemoryMedium medium) {
  std::string key = logical_id + "@" + std::to_string(static_cast<int>(medium));
  auto it = _source_addrs.find(key);
  if (it != _source_addrs.end()) return it->second;
  addr_type addr = allocate_address_in_medium(static_cast<uint32_t>(bytes), medium);
  _source_addrs[key] = addr;
  return addr;
}
