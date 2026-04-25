#ifndef DRAM_H
#define DRAM_H
#include <algorithm>
#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <utility>

#include "Common.h"
#include "ramulator/Ramulator.hpp"
#include "ramulator2.hh"


class Dram {
 public:
  virtual ~Dram() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual void advance_to(uint64_t now_ps);
  virtual bool is_full(uint32_t cid, MemoryAccess* request) = 0;
  virtual void push(uint32_t cid, MemoryAccess* request) = 0;
  virtual bool is_empty(uint32_t cid) = 0;
  virtual MemoryAccess* top(uint32_t cid) = 0;
  virtual void pop(uint32_t cid) = 0;
  virtual bool owns_address(addr_type addr) const {
    return addr >= _address_base &&
           addr < _address_base + _capacity_bytes;
  }
  uint32_t get_channel_id(MemoryAccess* request);
  uint64_t next_event_time_ps() {
    return running() ? (_time_ps + std::max<uint64_t>(_period_ps, 1))
                     : UINT64_MAX;
  }
  uint64_t current_time_ps() const { return _time_ps; }
  virtual void print_stat() {}

 protected:
  SimulationConfig _config;
  uint32_t _n_ch;
  cycle_type _cycles;
  uint64_t _period_ps = 0;
  uint64_t _time_ps = 0;
  uint64_t _inflight_requests = 0;
  uint64_t _issued_requests = 0;
  uint64_t _address_base = 0;
  uint64_t _capacity_bytes = 0;
  uint32_t _address_req_size = 32;
};

class SimpleDram : public Dram {
 public:
  SimpleDram(SimulationConfig config);
  ~SimpleDram() override;
  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;

 private:
  uint32_t _latency;
  double _bandwidth;

  uint64_t _last_finish_cycle;
  std::vector<std::queue<std::pair<addr_type, MemoryAccess*>>> _waiting_queue;
  std::vector<std::queue<MemoryAccess*>> _response_queue;
};

class DramRamulator : public Dram {
 public:
  DramRamulator(SimulationConfig config);
  ~DramRamulator() override;
  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::unique_ptr<ram::Ramulator> _mem;
  robin_hood::unordered_flat_map<uint64_t, MemoryAccess*> _waiting_mem_access;
  std::queue<MemoryAccess*> _responses;

  std::vector<uint64_t> _total_processed_requests;
  std::vector<uint64_t> _processed_requests;
};

class Ramulator2Memory : public Dram {
 public:
  Ramulator2Memory(const SimulationConfig& config,
                   const TieredMemoryConfig& tier_config,
                   std::string device_name);
  ~Ramulator2Memory() override;
  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 protected:
  TieredMemoryConfig _tier_config;
  std::string _device_name;
  std::vector<std::unique_ptr<NDPSim::Ramulator2>> _mem;
  int _tx_ch_log2;
  int _tx_log2;
  int _req_size;
};

class Ddr : public Ramulator2Memory {
 public:
  explicit Ddr(const SimulationConfig& config);
};
#endif
