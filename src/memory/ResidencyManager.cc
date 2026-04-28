#include "ResidencyManager.h"

#include <algorithm>
#include <limits>

void ResidencyManager::configure_capacity(uint64_t capacity_bytes) {
  _capacity_bytes = capacity_bytes;
  spdlog::info("[Residency] HBM residency capacity {} bytes ({:.3f} MiB)",
               _capacity_bytes,
               static_cast<double>(_capacity_bytes) / (1024.0 * 1024.0));
}

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
  entry.bytes = std::max(entry.bytes, bytes);
  entry.last_touch = ++_clock;
  if (entry.resident_addr == 0) {
    entry.resident_addr =
        allocate_address_in_medium(static_cast<uint32_t>(bytes), medium);
  }
  return entry.resident_addr;
}

void ResidencyManager::note_entry(const std::string& logical_id,
                                  uint64_t bytes,
                                  uint32_t user_id,
                                  int32_t layer_id,
                                  const std::string& role,
                                  int64_t next_use_rank) {
  auto& entry = _entries[logical_id];
  entry.bytes = std::max(entry.bytes, bytes);
  entry.user_id = user_id;
  entry.layer_id = layer_id;
  entry.role = role;
  entry.next_use_rank = next_use_rank;
  entry.last_touch = ++_clock;
}

bool ResidencyManager::evict_one(const std::set<std::string>& protected_ids) {
  auto victim = _entries.end();
  for (auto it = _entries.begin(); it != _entries.end(); ++it) {
    const Entry& entry = it->second;
    if (!entry.resident || entry.loading || entry.pin_count > 0) continue;
    if (protected_ids.count(it->first)) continue;
    if (victim == _entries.end()) {
      victim = it;
      continue;
    }
    const Entry& best = victim->second;
    if (entry.next_use_rank > best.next_use_rank ||
        (entry.next_use_rank == best.next_use_rank &&
         entry.last_touch < best.last_touch)) {
      victim = it;
    }
  }
  if (victim == _entries.end()) return false;

  Entry& entry = victim->second;
  entry.resident = false;
  if (_used_bytes >= entry.bytes) _used_bytes -= entry.bytes;
  else _used_bytes = 0;
  spdlog::info("[Residency] evict {} role={} layer={} bytes={} used={}/{}",
               victim->first, entry.role, entry.layer_id, entry.bytes,
               _used_bytes, _capacity_bytes);
  return true;
}

bool ResidencyManager::ensure_capacity_for(
    const std::vector<StageLoad>& loads,
    const std::set<std::string>& protected_ids,
    const std::string& reason) {
  std::vector<StageLoad> unique_loads;
  std::set<std::string> seen;
  uint64_t additional_bytes = 0;
  for (const auto& load : loads) {
    if (load.logical_id.empty() || !seen.insert(load.logical_id).second) continue;
    auto& entry = _entries[load.logical_id];
    note_entry(load.logical_id, load.bytes, load.user_id, load.layer_id,
               load.role, load.next_use_rank);
    if (entry.resident || entry.loading) continue;
    unique_loads.push_back(load);
    additional_bytes += entry.bytes;
  }

  if (!capacity_enabled()) {
    for (const auto& load : unique_loads) {
      Entry& entry = _entries[load.logical_id];
      if (!entry.loading && !entry.resident) {
        entry.loading = true;
        _used_bytes += entry.bytes;
      }
    }
    _blocked_reasons.erase(reason);
    return true;
  }

  uint64_t request_bytes = additional_bytes;
  if (request_bytes > _capacity_bytes) {
    spdlog::error("[Residency] stage {} requires {} bytes, larger than HBM "
                  "residency capacity {} bytes",
                  reason, request_bytes, _capacity_bytes);
    return false;
  }

  while (_used_bytes + additional_bytes > _capacity_bytes) {
    if (!evict_one(protected_ids)) {
      if (_blocked_reasons.insert(reason).second) {
        spdlog::info("[Residency] capacity blocked stage {} need={} used={}/{}",
                     reason, additional_bytes, _used_bytes, _capacity_bytes);
      }
      return false;
    }
  }

  for (const auto& load : unique_loads) {
    Entry& entry = _entries[load.logical_id];
    if (!entry.loading && !entry.resident) {
      entry.loading = true;
      _used_bytes += entry.bytes;
      spdlog::info("[Residency] reserve-load {} role={} layer={} bytes={} "
                   "used={}/{}",
                   load.logical_id, entry.role, entry.layer_id, entry.bytes,
                   _used_bytes, _capacity_bytes);
    }
  }
  _blocked_reasons.erase(reason);
  return true;
}

void ResidencyManager::mark_resident(const std::string& logical_id,
                                     addr_type resident_addr,
                                     uint64_t bytes) {
  auto& entry = _entries[logical_id];
  entry.resident_addr = resident_addr;
  entry.bytes = std::max(entry.bytes, bytes);
  if (!entry.resident && !entry.loading) _used_bytes += entry.bytes;
  entry.resident = true;
  entry.loading = false;
  entry.last_touch = ++_clock;
  spdlog::info("[Residency] resident {} role={} layer={} bytes={} used={}/{}",
               logical_id, entry.role, entry.layer_id, entry.bytes,
               _used_bytes, _capacity_bytes);
}

void ResidencyManager::pin(const std::string& logical_id) {
  auto it = _entries.find(logical_id);
  if (it == _entries.end()) return;
  it->second.pin_count++;
  it->second.last_touch = ++_clock;
}

void ResidencyManager::unpin(const std::string& logical_id) {
  auto it = _entries.find(logical_id);
  if (it == _entries.end() || it->second.pin_count == 0) return;
  it->second.pin_count--;
  it->second.last_touch = ++_clock;
}

void ResidencyManager::release(const std::string& logical_id) {
  auto it = _entries.find(logical_id);
  if (it == _entries.end()) return;
  it->second.pin_count = 0;
  it->second.last_touch = ++_clock;
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
