#include "StorageController.h"

#include <algorithm>
#include <limits>

namespace {

uint64_t migration_total_bytes(const MigrationRequest& request) {
  if (request.segments.empty()) return request.bytes;
  uint64_t total = 0;
  for (const auto& segment : request.segments) total += segment.bytes;
  return total;
}

std::vector<MigrationSegment> normalized_segments(
    const MigrationRequest& request) {
  if (!request.segments.empty()) return request.segments;
  if (request.bytes == 0) return {};
  return std::vector<MigrationSegment>{MigrationSegment{
      .src_addr = request.src_addr,
      .dst_addr = request.dst_addr,
      .bytes = request.bytes,
  }};
}

}  // namespace

StorageController::StorageController(SimulationConfig config, Dram* hbm,
                                     Dram* ddr, Ssd* ssd)
    : _config(config), _hbm(hbm), _ddr(ddr), _ssd(ssd) {
  _runtime_stats.hbm.name = "HBM";
  _runtime_stats.ddr.name = "DDR";
  _runtime_stats.ssd.name = "SSD";
}

void StorageController::advance_to(uint64_t now_ps) {
  _last_advanced_ps = std::max(_last_advanced_ps, now_ps);
  if (_hbm) _hbm->advance_to(now_ps);
  if (_ddr) _ddr->advance_to(now_ps);
  if (_ssd) _ssd->advance_to(now_ps);
  drain_device_responses(now_ps, _hbm, MemoryMedium::HBM, _config.hbm.channels);
  drain_device_responses(now_ps, _ddr, MemoryMedium::DDR, _config.ddr.channels);
  drain_ssd_responses(now_ps);
  service_migrations(now_ps);
  flush_pending_ssd_writes(now_ps, false);
}

bool StorageController::dispatch_request(uint32_t preferred_port,
                                         MemoryAccess* request,
                                         uint64_t now_ps) {
  if (request == nullptr) return false;

  MemoryMedium medium = MemoryMedium::HBM;
  if (_ssd && _ssd->owns_address(request->dram_address)) {
    medium = MemoryMedium::SSD;
  } else if (_ddr && _ddr->owns_address(request->dram_address)) {
    medium = MemoryMedium::DDR;
  }
  return route_to_device(preferred_port, request, medium, now_ps);
}

MemoryAccess* StorageController::top_ready_response() {
  assert(!_ready_responses.empty());
  return _ready_responses.front();
}

void StorageController::pop_ready_response() {
  assert(!_ready_responses.empty());
  _ready_responses.pop_front();
}

bool StorageController::has_pending() const {
  return !_ready_responses.empty() || !_retry_queue.empty() ||
         !_active_migrations.empty() || !_pending_ssd_writes.empty() ||
         !_ssd_inflight_read_pages.empty() || (_hbm && _hbm->running()) ||
         (_ddr && _ddr->running()) || (_ssd && _ssd->running());
}

uint64_t StorageController::next_event_time_ps() const {
  if (!_ready_responses.empty()) return _last_advanced_ps;
  uint64_t next_ps = std::numeric_limits<uint64_t>::max();
  if (_hbm) next_ps = std::min(next_ps, _hbm->next_event_time_ps());
  if (_ddr) next_ps = std::min(next_ps, _ddr->next_event_time_ps());
  if (_ssd) next_ps = std::min(next_ps, _ssd->next_event_time_ps());
  if (!_pending_ssd_writes.empty()) {
    uint64_t timeout_ps = ssd_write_idle_timeout_ps();
    for (const auto& [_, pending] : _pending_ssd_writes) {
      next_ps = std::min(next_ps, pending.last_update_ps + timeout_ps);
    }
  }
  if (!_retry_queue.empty() || !_active_migrations.empty()) {
    next_ps = std::min(next_ps, _last_advanced_ps);
  }
  return next_ps;
}

uint64_t StorageController::submit_migration_request(
    const MigrationRequest& request, uint64_t now_ps) {
  auto ids = submit_migration_requests(std::vector<MigrationRequest>{request},
                                       now_ps);
  return ids.empty() ? 0 : ids.front();
}

std::vector<uint64_t> StorageController::submit_migration_requests(
    const std::vector<MigrationRequest>& requests, uint64_t now_ps) {
  std::vector<uint64_t> ids;
  ids.reserve(requests.size());
  for (const auto& request : requests) {
    if (migration_total_bytes(request) == 0) continue;
    MigrationRequest req = request;
    if (req.id == 0) req.id = _next_migration_id++;
    req.submitted_time_ps = now_ps;
    req.segments = normalized_segments(req);
    uint64_t total_bytes = migration_total_bytes(req);
    _active_migrations[req.id] = ActiveMigration{
        .request = req,
        .total_bytes = total_bytes,
    };
    ids.push_back(req.id);
  }
  service_migrations(now_ps);
  return ids;
}

bool StorageController::movement_done(uint64_t movement_id) const {
  return _completed_migrations.find(movement_id) != _completed_migrations.end();
}

bool StorageController::movements_done(
    const std::vector<uint64_t>& movement_ids) const {
  for (uint64_t id : movement_ids) {
    if (!movement_done(id)) return false;
  }
  return true;
}

Dram* StorageController::device_for_medium(MemoryMedium medium) const {
  switch (medium) {
    case MemoryMedium::HBM:
      return _hbm;
    case MemoryMedium::DDR:
      return _ddr;
    case MemoryMedium::SSD:
    case MemoryMedium::UNKNOWN:
    default:
      return nullptr;
  }
}

MemoryTierRuntimeStats& StorageController::tier_stats_for_medium(
    MemoryMedium medium) {
  switch (medium) {
    case MemoryMedium::HBM:
      return _runtime_stats.hbm;
    case MemoryMedium::DDR:
      return _runtime_stats.ddr;
    case MemoryMedium::SSD:
      return _runtime_stats.ssd;
    case MemoryMedium::UNKNOWN:
    default:
      return _runtime_stats.hbm;
  }
}

void StorageController::record_dispatch_blocked(MemoryMedium medium) {
  tier_stats_for_medium(medium).dispatch_blocked++;
}

void StorageController::record_completed_access(const MemoryAccess* access) {
  if (access == nullptr) return;
  MemoryTierRuntimeStats& stats = tier_stats_for_medium(access->target_medium);
  const uint64_t bytes = access->logical_size_bytes == 0
                             ? access->size
                             : access->logical_size_bytes;
  if (access->write) {
    stats.write_requests++;
    stats.write_bytes += bytes;
    if (access->controller_generated)
      stats.controller_write_bytes += bytes;
    else
      stats.core_write_bytes += bytes;
  } else {
    stats.read_requests++;
    stats.read_bytes += bytes;
    if (access->controller_generated)
      stats.controller_read_bytes += bytes;
    else
      stats.core_read_bytes += bytes;
  }
  if (access->mem_finish_time_ps >= access->mem_enter_time_ps) {
    uint64_t latency = access->mem_finish_time_ps - access->mem_enter_time_ps;
    stats.total_latency_ps += latency;
    stats.max_latency_ps = std::max(stats.max_latency_ps, latency);
  }
}

bool StorageController::route_to_device(uint32_t preferred_port,
                                        MemoryAccess* request,
                                        MemoryMedium medium,
                                        uint64_t now_ps) {
  request->target_medium = medium;
  request->mem_enter_time_ps = now_ps;
  if (request->issue_time_ps == 0) request->issue_time_ps = now_ps;
  if (request->logical_size_bytes == 0) request->logical_size_bytes = request->size;

  if (medium == MemoryMedium::SSD) {
    return route_to_ssd(request, now_ps);
  }

  Dram* device = device_for_medium(medium);
  if (device == nullptr) return false;
  uint32_t channel_count =
      medium == MemoryMedium::HBM ? _config.hbm.channels : _config.ddr.channels;
  uint32_t channel =
      preferred_port < channel_count ? preferred_port : device->get_channel_id(request);
  if (device->is_full(channel, request)) {
    record_dispatch_blocked(medium);
    return false;
  }
  device->push(channel, request);
  return true;
}

bool StorageController::route_to_ssd(MemoryAccess* request, uint64_t now_ps) {
  if (_ssd == nullptr) return false;
  if (request->ssd_host_request) {
    if (_ssd->is_full(request)) {
      record_dispatch_blocked(MemoryMedium::SSD);
      return false;
    }
    _ssd->set_current_time_ps(now_ps);
    _ssd->push(request);
    return true;
  }

  if (request->write) {
    return handle_ssd_write(request, now_ps);
  }
  return handle_ssd_read(request, now_ps);
}

bool StorageController::handle_ssd_read(MemoryAccess* request, uint64_t now_ps) {
  addr_type page_addr = ssd_page_addr(request->dram_address);
  auto inflight_it = _ssd_inflight_read_pages.find(page_addr);
  if (inflight_it != _ssd_inflight_read_pages.end()) {
    _ssd_read_aggregates[inflight_it->second].waiters.push_back(request);
    return true;
  }

  auto* aggregate = new MemoryAccess();
  aggregate->id = generate_mem_access_id();
  aggregate->dram_address = page_addr;
  aggregate->spad_address = request->spad_address;
  aggregate->size = ssd_page_bytes();
  aggregate->logical_size_bytes = ssd_page_bytes();
  aggregate->write = false;
  aggregate->request = true;
  aggregate->core_id = request->core_id;
  aggregate->buffer_id = request->buffer_id;
  aggregate->issue_time_ps = request->issue_time_ps;
  aggregate->mem_enter_time_ps = now_ps;
  aggregate->ssd_host_request = true;
  aggregate->target_medium = MemoryMedium::SSD;

  if (!dispatch_ssd_aggregate(aggregate, now_ps)) {
    delete aggregate;
    return false;
  }

  _ssd_inflight_read_pages[page_addr] = aggregate->id;
  _ssd_read_aggregates[aggregate->id].waiters.push_back(request);
  return true;
}

bool StorageController::handle_ssd_write(MemoryAccess* request, uint64_t now_ps) {
  SsdWriteStreamKey key = make_ssd_write_stream_key(request);
  auto it = _pending_ssd_writes.find(key);

  if (it == _pending_ssd_writes.end()) {
    size_t active_pages = 0;
    bool have_oldest = false;
    SsdWriteStreamKey oldest_key;
    uint64_t oldest_created_ps = std::numeric_limits<uint64_t>::max();
    for (const auto& [existing_key, pending] : _pending_ssd_writes) {
      if (!same_ssd_write_stream(existing_key, key)) continue;
      active_pages++;
      if (!have_oldest || pending.created_time_ps < oldest_created_ps) {
        have_oldest = true;
        oldest_key = existing_key;
        oldest_created_ps = pending.created_time_ps;
      }
    }
    if (active_pages >= 4 && have_oldest) {
      if (!flush_pending_ssd_write_key(oldest_key, now_ps)) return false;
    }

    PendingSsdWrite pending;
    pending.page_addr = key.page_addr;
    pending.created_time_ps = now_ps;
    pending.last_update_ps = now_ps;
    pending.waiters.push_back(request);
    _pending_ssd_writes.emplace(key, std::move(pending));
  } else {
    it->second.last_update_ps = now_ps;
    it->second.waiters.push_back(request);
  }

  if (_pending_ssd_writes[key].waiters.size() * _config.hbm.req_size >=
      ssd_page_bytes()) {
    flush_pending_ssd_writes(now_ps, false);
  }
  return true;
}

void StorageController::flush_pending_ssd_writes(uint64_t now_ps, bool force) {
  if (_ssd == nullptr || _pending_ssd_writes.empty()) return;

  std::vector<SsdWriteStreamKey> flush_keys;
  uint64_t timeout_ps = ssd_write_idle_timeout_ps();
  for (const auto& [key, pending] : _pending_ssd_writes) {
    if (pending.waiters.empty()) {
      flush_keys.push_back(key);
      continue;
    }
    bool timed_out = pending.last_update_ps + timeout_ps <= now_ps;
    bool page_full =
        pending.waiters.size() * _config.hbm.req_size >= ssd_page_bytes();
    if (force || timed_out || page_full) flush_keys.push_back(key);
  }

  for (const auto& key : flush_keys) {
    flush_pending_ssd_write_key(key, now_ps);
  }
}

bool StorageController::flush_pending_ssd_write_key(const SsdWriteStreamKey& key,
                                                    uint64_t now_ps) {
  auto it = _pending_ssd_writes.find(key);
  if (it == _pending_ssd_writes.end()) return true;
  if (it->second.waiters.empty()) {
    _pending_ssd_writes.erase(it);
    return true;
  }

  auto* aggregate = new MemoryAccess();
  aggregate->id = generate_mem_access_id();
  aggregate->dram_address = it->second.page_addr;
  aggregate->spad_address = it->second.waiters.front()->spad_address;
  aggregate->size = ssd_page_bytes();
  aggregate->logical_size_bytes = ssd_page_bytes();
  aggregate->write = true;
  aggregate->request = true;
  aggregate->core_id = it->second.waiters.front()->core_id;
  aggregate->buffer_id = 0;
  aggregate->issue_time_ps = it->second.waiters.front()->issue_time_ps;
  aggregate->mem_enter_time_ps = now_ps;
  aggregate->ssd_host_request = true;
  aggregate->target_medium = MemoryMedium::SSD;

  if (!dispatch_ssd_aggregate(aggregate, now_ps)) {
    delete aggregate;
    return false;
  }

  SsdAggregateContext context;
  context.waiters = std::move(it->second.waiters);
  _ssd_write_aggregates.emplace(aggregate->id, std::move(context));
  _pending_ssd_writes.erase(it);
  return true;
}

bool StorageController::dispatch_ssd_aggregate(MemoryAccess* aggregate,
                                               uint64_t now_ps) {
  if (_ssd == nullptr) return false;
  if (_ssd->is_full(aggregate)) {
    record_dispatch_blocked(MemoryMedium::SSD);
    return false;
  }
  _ssd->set_current_time_ps(now_ps);
  _ssd->push(aggregate);
  return true;
}

void StorageController::complete_ssd_aggregate(uint64_t now_ps,
                                               MemoryAccess* response) {
  auto write_it = _ssd_write_aggregates.find(response->id);
  if (write_it != _ssd_write_aggregates.end()) {
    auto waiters = std::move(write_it->second.waiters);
    _ssd_write_aggregates.erase(write_it);
    for (MemoryAccess* waiter : waiters) {
      waiter->request = false;
      waiter->ssd_host_request = false;
      waiter->target_medium = MemoryMedium::SSD;
      waiter->mem_finish_time_ps = response->mem_finish_time_ps;
      waiter->dram_finish_cycle = response->dram_finish_cycle;
      handle_completed_access(now_ps, waiter);
    }
    delete response;
    return;
  }

  addr_type page_addr = ssd_page_addr(response->dram_address);
  auto read_page_it = _ssd_inflight_read_pages.find(page_addr);
  if (read_page_it == _ssd_inflight_read_pages.end() ||
      read_page_it->second != response->id) {
    return;
  }

  auto read_ctx_it = _ssd_read_aggregates.find(response->id);
  if (read_ctx_it == _ssd_read_aggregates.end()) return;
  auto waiters = std::move(read_ctx_it->second.waiters);
  _ssd_read_aggregates.erase(read_ctx_it);
  _ssd_inflight_read_pages.erase(read_page_it);
  for (MemoryAccess* waiter : waiters) {
    waiter->request = false;
    waiter->ssd_host_request = false;
    waiter->target_medium = MemoryMedium::SSD;
    waiter->mem_finish_time_ps = response->mem_finish_time_ps;
    waiter->dram_finish_cycle = response->dram_finish_cycle;
    handle_completed_access(now_ps, waiter);
  }
  delete response;
}

uint64_t StorageController::ssd_page_bytes() const {
  return static_cast<uint64_t>(_config.ssd.secsz) * _config.ssd.secs_per_pg;
}

addr_type StorageController::ssd_page_addr(addr_type addr) const {
  uint64_t page_bytes = ssd_page_bytes();
  if (page_bytes == 0 || !_ssd || !_ssd->owns_address(addr)) return addr;
  uint64_t offset = addr - _config.ssd.address_base;
  return _config.ssd.address_base + (offset / page_bytes) * page_bytes;
}

uint64_t StorageController::ssd_write_idle_timeout_ps() const {
  uint64_t icnt_period = _config.icnt_freq > 0 ? 1000000ULL / _config.icnt_freq : 1000ULL;
  uint64_t req_size = std::max<uint64_t>(_config.hbm.req_size, 1);
  uint64_t page_segments = std::max<uint64_t>(1, ssd_page_bytes() / req_size);
  return std::max<uint64_t>(1, page_segments) * icnt_period;
}

StorageController::SsdWriteStreamKey StorageController::make_ssd_write_stream_key(
    const MemoryAccess* request) const {
  SsdWriteStreamKey key;
  key.controller_generated = request->controller_generated;
  key.core_id = request->core_id;
  key.macro_request_id = request->macro_request_id;
  key.page_addr = ssd_page_addr(request->dram_address);
  return key;
}

bool StorageController::same_ssd_write_stream(
    const SsdWriteStreamKey& lhs, const SsdWriteStreamKey& rhs) const {
  return lhs.controller_generated == rhs.controller_generated &&
         lhs.core_id == rhs.core_id &&
         lhs.macro_request_id == rhs.macro_request_id;
}

void StorageController::drain_device_responses(uint64_t now_ps, Dram* device,
                                               MemoryMedium medium,
                                               uint32_t channel_count) {
  if (device == nullptr) return;
  for (uint32_t ch = 0; ch < channel_count; ch++) {
    while (!device->is_empty(ch)) {
      MemoryAccess* response = device->top(ch);
      device->pop(ch);
      response->target_medium = medium;
      if (response->mem_finish_time_ps == 0)
        response->mem_finish_time_ps = now_ps;
      handle_completed_access(now_ps, response);
    }
  }
}

void StorageController::drain_ssd_responses(uint64_t now_ps) {
  if (_ssd == nullptr) return;
  while (!_ssd->is_empty()) {
    MemoryAccess* response = _ssd->top();
    _ssd->pop();
    response->target_medium = MemoryMedium::SSD;
    if (response->mem_finish_time_ps == 0)
      response->mem_finish_time_ps = now_ps;
    if (_ssd_write_aggregates.find(response->id) != _ssd_write_aggregates.end() ||
        _ssd_read_aggregates.find(response->id) != _ssd_read_aggregates.end()) {
      record_completed_access(response);
      complete_ssd_aggregate(now_ps, response);
      continue;
    }
    handle_completed_access(now_ps, response);
  }
}

void StorageController::handle_completed_access(uint64_t now_ps,
                                                MemoryAccess* response) {
  if (response->target_medium == MemoryMedium::HBM ||
      response->target_medium == MemoryMedium::DDR ||
      (response->target_medium == MemoryMedium::SSD &&
       response->ssd_host_request)) {
    record_completed_access(response);
  }

  if (!response->controller_generated) {
    _ready_responses.push_back(response);
    return;
  }

  auto migration_it = _active_migrations.find(response->macro_request_id);
  if (migration_it == _active_migrations.end()) {
    _ready_responses.push_back(response);
    return;
  }

  auto& migration = migration_it->second;
  if (!response->write) {
    if (migration.inflight_reads > 0) migration.inflight_reads--;

    auto* write_request = new MemoryAccess();
    write_request->id = generate_mem_access_id();
    write_request->dram_address = response->aux_address;
    write_request->size = response->size;
    write_request->logical_size_bytes = response->logical_size_bytes;
    write_request->write = true;
    write_request->request = true;
    write_request->macro_request_id = response->macro_request_id;
    write_request->source_medium = response->source_medium;
    write_request->destination_medium = response->destination_medium;
    write_request->controller_generated = true;

    uint32_t preferred_port = std::numeric_limits<uint32_t>::max();
    Dram* dst_device = device_for_medium(response->destination_medium);
    if (dst_device != nullptr) preferred_port = dst_device->get_channel_id(write_request);

    if (route_to_device(preferred_port, write_request,
                        response->destination_medium, now_ps)) {
      migration.inflight_writes++;
    } else {
      _retry_queue.push(write_request);
    }
  } else {
    if (migration.inflight_writes > 0) migration.inflight_writes--;
    migration.bytes_written += response->logical_size_bytes;
    _completed_migration_ranges.push_back(CompletedMigrationRange{
        .movement_id = response->macro_request_id,
        .dst_addr = response->dram_address,
        .bytes = response->logical_size_bytes,
    });
  }

  delete response;

  if (migration.bytes_written >= migration.total_bytes &&
      migration.inflight_reads == 0 && migration.inflight_writes == 0 &&
      migration.segment_index >= migration.request.segments.size()) {
    _completed_migrations.insert(migration_it->first);
    _active_migrations.erase(migration_it);
  }
}

void StorageController::service_migrations(uint64_t now_ps) {
  size_t retry_count = _retry_queue.size();
  while (retry_count-- > 0 && !_retry_queue.empty()) {
    MemoryAccess* retry = _retry_queue.front();
    uint32_t preferred_port = std::numeric_limits<uint32_t>::max();
    Dram* target_device = device_for_medium(retry->target_medium);
    if (target_device != nullptr) preferred_port = target_device->get_channel_id(retry);
    if (!route_to_device(preferred_port, retry, retry->target_medium, now_ps))
      break;

    auto migration_it = _active_migrations.find(retry->macro_request_id);
    if (migration_it != _active_migrations.end()) {
      if (retry->write)
        migration_it->second.inflight_writes++;
      else
        migration_it->second.inflight_reads++;
    }
    _retry_queue.pop();
  }

  for (auto& [migration_id, migration] : _active_migrations) {
    while (migration.segment_index < migration.request.segments.size()) {
      const auto& segment = migration.request.segments[migration.segment_index];
      if (segment.bytes == 0) {
        migration.segment_index++;
        migration.segment_offset = 0;
        continue;
      }
      uint64_t remaining = segment.bytes - migration.segment_offset;
      uint64_t chunk = std::min<uint64_t>(_config.hbm.req_size, remaining);
      auto* read_request = new MemoryAccess();
      read_request->id = generate_mem_access_id();
      read_request->dram_address = segment.src_addr + migration.segment_offset;
      read_request->aux_address = segment.dst_addr + migration.segment_offset;
      read_request->size = chunk;
      read_request->logical_size_bytes = chunk;
      read_request->write = false;
      read_request->request = true;
      read_request->macro_request_id = migration_id;
      read_request->source_medium = migration.request.src_medium;
      read_request->destination_medium = migration.request.dst_medium;
      read_request->controller_generated = true;

      uint32_t preferred_port = std::numeric_limits<uint32_t>::max();
      Dram* src_device = device_for_medium(migration.request.src_medium);
      if (src_device != nullptr) preferred_port = src_device->get_channel_id(read_request);
      if (!route_to_device(preferred_port, read_request,
                           migration.request.src_medium, now_ps)) {
        delete read_request;
        break;
      }

      migration.inflight_reads++;
      migration.segment_offset += chunk;
      if (migration.segment_offset >= segment.bytes) {
        migration.segment_index++;
        migration.segment_offset = 0;
      }
    }
  }
}
