#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <queue>
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
  struct ActiveSsdPageRequest {
    addr_type page_addr = 0;
    bool write = false;
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
  bool dispatch_ssd_page_request(MemoryAccess* request, uint64_t now_ps);
  uint64_t ssd_page_bytes() const;
  addr_type align_ssd_page_address(addr_type addr) const;
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
  std::map<std::pair<addr_type, bool>, uint64_t> _active_ssd_page_lookup;
  std::map<uint64_t, ActiveSsdPageRequest> _active_ssd_page_requests;
  std::map<uint64_t, ActiveMigration> _active_migrations;
};
