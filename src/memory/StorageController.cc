#include "StorageController.h"

#include <algorithm>
#include <limits>

StorageController::StorageController(SimulationConfig config, Dram* dram,
                                     Ssd* ssd)
    : _config(config), _dram(dram), _ssd(ssd) {}

void StorageController::advance_to(uint64_t now_ps) {
  _last_advanced_ps = std::max(_last_advanced_ps, now_ps);
  if (_dram) _dram->advance_to(now_ps);
  if (_ssd) _ssd->advance_to(now_ps);
  drain_dram_responses(now_ps);
  drain_ssd_responses(now_ps);
  service_migrations(now_ps);
}

bool StorageController::dispatch_request(uint32_t preferred_port,
                                         MemoryAccess* request,
                                         uint64_t now_ps) {
  if (request == nullptr) return false;
  MemoryMedium medium = MemoryMedium::DRAM;
  if (_ssd && _ssd->owns_address(request->dram_address)) {
    medium = MemoryMedium::SSD;
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
         !_active_migrations.empty() || (_dram && _dram->running()) ||
         (_ssd && _ssd->running());
}

uint64_t StorageController::next_event_time_ps() const {
  if (!_ready_responses.empty()) return _last_advanced_ps;
  uint64_t next_ps = std::numeric_limits<uint64_t>::max();
  if (_dram) next_ps = std::min(next_ps, _dram->next_event_time_ps());
  if (_ssd) next_ps = std::min(next_ps, _ssd->next_event_time_ps());
  if (!_retry_queue.empty() || !_active_migrations.empty()) {
    next_ps = std::min(next_ps, _last_advanced_ps);
  }
  return next_ps;
}

uint64_t StorageController::submit_migration_request(
    const MigrationRequest& request, uint64_t now_ps) {
  MigrationRequest req = request;
  if (req.id == 0) req.id = _next_migration_id++;
  req.submitted_time_ps = now_ps;
  _active_migrations[req.id] = ActiveMigration{.request = req};
  service_migrations(now_ps);
  return req.id;
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
    if (_ssd == nullptr || _ssd->is_full(request)) return false;
    _ssd->set_current_time_ps(now_ps);
    _ssd->push(request);
    return true;
  }

  if (_dram == nullptr) return false;
  uint32_t channel =
      preferred_port < _config.dram_channels ? preferred_port
                                             : _dram->get_channel_id(request);
  if (_dram->is_full(channel, request)) return false;
  _dram->push(channel, request);
  return true;
}

void StorageController::drain_dram_responses(uint64_t now_ps) {
  if (_dram == nullptr) return;
  for (uint32_t ch = 0; ch < _config.dram_channels; ch++) {
    while (!_dram->is_empty(ch)) {
      MemoryAccess* response = _dram->top(ch);
      _dram->pop(ch);
      response->target_medium = MemoryMedium::DRAM;
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
    handle_completed_access(now_ps, response);
  }
}

void StorageController::handle_completed_access(uint64_t now_ps,
                                                MemoryAccess* response) {
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

    uint32_t preferred_port = _config.dram_channels;
    if (response->destination_medium == MemoryMedium::DRAM && _dram) {
      preferred_port = _dram->get_channel_id(write_request);
    }
    if (route_to_device(preferred_port, write_request,
                        response->destination_medium, now_ps)) {
      migration.inflight_writes++;
    } else {
      _retry_queue.push(write_request);
    }
  } else {
    if (migration.inflight_writes > 0) migration.inflight_writes--;
    migration.bytes_written += response->logical_size_bytes;
  }

  delete response;

  if (migration.bytes_written >= migration.request.bytes &&
      migration.inflight_reads == 0 && migration.inflight_writes == 0 &&
      migration.next_offset >= migration.request.bytes) {
    _active_migrations.erase(migration_it);
  }
}

void StorageController::service_migrations(uint64_t now_ps) {
  size_t retry_count = _retry_queue.size();
  while (retry_count-- > 0 && !_retry_queue.empty()) {
    MemoryAccess* retry = _retry_queue.front();
    uint32_t preferred_port = _config.dram_channels;
    if (retry->target_medium == MemoryMedium::DRAM && _dram) {
      preferred_port = _dram->get_channel_id(retry);
    }
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
    while (migration.next_offset < migration.request.bytes) {
      uint64_t chunk =
          std::min<uint64_t>(_config.dram_req_size,
                             migration.request.bytes - migration.next_offset);
      auto* read_request = new MemoryAccess();
      read_request->id = generate_mem_access_id();
      read_request->dram_address = migration.request.src_addr + migration.next_offset;
      read_request->aux_address = migration.request.dst_addr + migration.next_offset;
      read_request->size = chunk;
      read_request->logical_size_bytes = chunk;
      read_request->write = false;
      read_request->request = true;
      read_request->macro_request_id = migration_id;
      read_request->source_medium = migration.request.src_medium;
      read_request->destination_medium = migration.request.dst_medium;
      read_request->controller_generated = true;

      uint32_t preferred_port = _config.dram_channels;
      if (migration.request.src_medium == MemoryMedium::DRAM && _dram) {
        preferred_port = _dram->get_channel_id(read_request);
      }
      if (!route_to_device(preferred_port, read_request,
                           migration.request.src_medium, now_ps)) {
        delete read_request;
        break;
      }

      migration.inflight_reads++;
      migration.next_offset += chunk;
    }
  }
}
