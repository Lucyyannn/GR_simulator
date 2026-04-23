#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "../Common.h"
#include "../Dram.h"
#include "../Ssd.h"

struct MigrationRequest {
  uint64_t id = 0;
  MemoryMedium src_medium = MemoryMedium::UNKNOWN;
  MemoryMedium dst_medium = MemoryMedium::UNKNOWN;
  addr_type src_addr = 0;
  addr_type dst_addr = 0;
  uint64_t bytes = 0;
  uint64_t submitted_time_ps = 0;
};

class StorageController {
 public:
  StorageController(SimulationConfig config, Dram* dram, Ssd* ssd);

  void advance_to(uint64_t now_ps);
  bool dispatch_request(uint32_t preferred_port, MemoryAccess* request,
                        uint64_t now_ps);

  bool has_ready_response() const { return !_ready_responses.empty(); }
  MemoryAccess* top_ready_response();
  void pop_ready_response();

  bool has_pending() const;
  uint64_t next_event_time_ps() const;

  uint64_t submit_migration_request(const MigrationRequest& request,
                                    uint64_t now_ps);

 private:
  struct SsdWriteStreamKey {
    bool controller_generated = false;
    uint32_t core_id = 0;
    uint64_t macro_request_id = 0;
    addr_type page_addr = 0;

    bool operator<(const SsdWriteStreamKey& other) const {
      return std::tie(controller_generated, core_id,
                      macro_request_id, page_addr) <
             std::tie(other.controller_generated, other.core_id,
                      other.macro_request_id, other.page_addr);
    }
  };

  struct PendingSsdWrite {
    addr_type page_addr = 0;
    uint64_t created_time_ps = 0;
    uint64_t last_update_ps = 0;
    std::vector<MemoryAccess*> waiters;
  };

  struct SsdAggregateContext {
    std::vector<MemoryAccess*> waiters;
  };

  struct ActiveMigration {
    MigrationRequest request;
    uint64_t next_offset = 0;
    uint64_t bytes_written = 0;
    uint64_t inflight_reads = 0;
    uint64_t inflight_writes = 0;
  };

  bool route_to_device(uint32_t preferred_port, MemoryAccess* request,
                       MemoryMedium medium, uint64_t now_ps);
  bool route_to_ssd(MemoryAccess* request, uint64_t now_ps);
  bool handle_ssd_read(MemoryAccess* request, uint64_t now_ps);
  bool handle_ssd_write(MemoryAccess* request, uint64_t now_ps);
  void flush_pending_ssd_writes(uint64_t now_ps, bool force);
  bool flush_pending_ssd_write_key(const SsdWriteStreamKey& key, uint64_t now_ps);
  bool dispatch_ssd_aggregate(MemoryAccess* aggregate, uint64_t now_ps);
  void complete_ssd_aggregate(uint64_t now_ps, MemoryAccess* response);
  uint64_t ssd_page_bytes() const;
  addr_type ssd_page_addr(addr_type addr) const;
  uint64_t ssd_write_idle_timeout_ps() const;
  SsdWriteStreamKey make_ssd_write_stream_key(const MemoryAccess* request) const;
  bool same_ssd_write_stream(const SsdWriteStreamKey& lhs,
                             const SsdWriteStreamKey& rhs) const;
  void drain_dram_responses(uint64_t now_ps);
  void drain_ssd_responses(uint64_t now_ps);
  void handle_completed_access(uint64_t now_ps, MemoryAccess* response);
  void service_migrations(uint64_t now_ps);

  SimulationConfig _config;
  Dram* _dram = nullptr;
  Ssd* _ssd = nullptr;
  uint64_t _last_advanced_ps = 0;
  uint64_t _next_migration_id = 1;
  std::deque<MemoryAccess*> _ready_responses;
  std::queue<MemoryAccess*> _retry_queue;
  std::map<uint64_t, ActiveMigration> _active_migrations;
  std::map<SsdWriteStreamKey, PendingSsdWrite> _pending_ssd_writes;
  std::map<uint64_t, SsdAggregateContext> _ssd_write_aggregates;
  std::map<addr_type, uint64_t> _ssd_inflight_read_pages;
  std::map<uint64_t, SsdAggregateContext> _ssd_read_aggregates;
};
