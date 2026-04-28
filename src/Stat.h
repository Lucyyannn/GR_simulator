#pragma once

#include <stdint.h>

#include <string>
#include <vector>

typedef struct {
  uint64_t start_cycle;
  uint64_t cycles;
  uint64_t compute_cycles;
  uint64_t memory_stall;
  uint64_t dependency_stall;
  uint64_t sram_reads;
  uint64_t sram_writes;
} TileStat;

typedef struct {
  uint64_t op_cycles;
  std::vector<TileStat> tile_stats;
} OpStat;

typedef struct {
  uint64_t total_cycles;
  std::vector<OpStat> op_stats;
} ModelStat;

struct CoreRuntimeStats {
  uint32_t core_id = 0;
  uint64_t total_cycles = 0;
  uint64_t systolic_active_cycles = 0;
  uint64_t vector_active_cycles = 0;
  uint64_t idle_cycles = 0;
  uint64_t memory_wait_cycles = 0;
  uint64_t dependency_wait_cycles = 0;
  uint64_t request_injection_wait_cycles = 0;
  double matmul_pe_cycles = 0.0;
};

struct MemoryTierRuntimeStats {
  std::string name;
  uint64_t read_requests = 0;
  uint64_t write_requests = 0;
  uint64_t read_bytes = 0;
  uint64_t write_bytes = 0;
  uint64_t core_read_bytes = 0;
  uint64_t core_write_bytes = 0;
  uint64_t controller_read_bytes = 0;
  uint64_t controller_write_bytes = 0;
  uint64_t dispatch_blocked = 0;
  uint64_t total_latency_ps = 0;
  uint64_t max_latency_ps = 0;
};

struct StorageRuntimeStats {
  MemoryTierRuntimeStats hbm;
  MemoryTierRuntimeStats ddr;
  MemoryTierRuntimeStats ssd;
};
