#ifndef SSD_H
#define SSD_H

#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <vector>
#include <memory>

#include "Common.h"

/**
 * @brief FEMU BlackBox-SSD inspired storage latency model.
 *
 * This module reproduces the core timing behavior of FEMU's bbssd
 * `ssd_advance_status()` (extern/femu/bbssd/ftl.c), including the
 * per-LUN `next_lun_avail_time` and per-channel `next_ch_avail_time`
 * bookkeeping that captures channel/LUN-level concurrency. The full
 * FTL (page mapping, GC, FDP, rte_ring, pqueue ...) is intentionally
 * elided to avoid pulling in QEMU dependencies; LBA -> (ch,lun) is a
 * simple stripe mapping, sufficient for NPU-centric workloads that
 * page DNN weights/KV-cache between SSD and DRAM.
 *
 * Interface is deliberately aligned with Dram (see Dram.h) so that
 * Simulator integrates it in the same push/top/pop style.
 */

/* NAND op type mirroring bbssd/ftl.h */
enum class SsdCmd {
    NAND_READ = 0,
    NAND_WRITE = 1,
    NAND_ERASE = 2,
};

/* Knobs inherited from FEMU `BbCtrlParams` (extern/femu/nvme.h) */
struct SsdConfig {
    /* Address space (global NPU-visible) */
    uint64_t address_base = 0x800000000ULL;    // where SSD region starts
    uint64_t capacity_bytes = (1ULL << 40);    // 1TB default

    /* Geometry */
    int secsz           = 512;
    int secs_per_pg     = 8;       // -> 4KB page
    int pgs_per_blk     = 256;
    int blks_per_pl     = 256;
    int pls_per_lun     = 1;
    int luns_per_ch     = 8;
    int nchs            = 8;

    /* Latencies in ns (FEMU defaults from ftl.h) */
    int pg_rd_lat       = 40000;    // 40us
    int pg_wr_lat       = 200000;   // 200us
    int blk_er_lat      = 2000000;  // 2ms
    int ch_xfer_lat     = 0;        // channel transfer
};

/**
 * @brief Minimal subset of FEMU ssd structures for latency modelling.
 *
 * We keep only `next_lun_avail_time` and `next_ch_avail_time` -- these are
 * the state used by ssd_advance_status() in bbssd/ftl.c.
 */
struct SsdLunState {
    uint64_t next_lun_avail_time = 0;
};

struct SsdChannelState {
    uint64_t next_ch_avail_time = 0;
    std::vector<SsdLunState> luns;
};

class Ssd {
 public:
  explicit Ssd(const SsdConfig& cfg, uint32_t tick_freq_mhz);
  ~Ssd();

  /* Lifecycle (mirrors Dram) */
  bool running();
  void cycle();                        // legacy tick wrapper around advance_to()
  void advance_to(uint64_t now_ps);
  uint64_t next_event_time_ps() const;
  void print_stat();

  void set_current_time_ps(uint64_t ps) { _now_ps = ps; }

  /* Address routing */
  bool owns_address(addr_type addr) const {
    return addr >= _cfg.address_base &&
           addr <  _cfg.address_base + _cfg.capacity_bytes;
  }

  /* Request path (analogous to Dram::push / top / pop, but the Simulator
     keeps SSD requests in a single virtual queue because SSD only has one
     "port" from the NoC-side perspective). */
  bool is_full(MemoryAccess* request);
  void push(MemoryAccess* request);

  bool is_empty();
  MemoryAccess* top();
  void pop();

  /* Stats */
  uint64_t total_reads()  const { return _stat_reads; }
  uint64_t total_writes() const { return _stat_writes; }

 private:
  /* === Core algorithm: direct port of FEMU bbssd/ftl.c
         ssd_advance_status() (NAND_READ / NAND_WRITE paths) === */
  uint64_t ssd_advance_status(uint32_t ch, uint32_t lun,
                              SsdCmd cmd, uint64_t cmd_stime_ns);

  /* LBA -> (ch, lun) stripe mapping (no full FTL) */
  void address_to_ch_lun(addr_type addr, uint32_t& ch, uint32_t& lun) const;

  /* ns <-> NPU cycle conversion */
  uint64_t ps_to_tick_cycles(uint64_t ps) const {
    return _tick_period_ps == 0 ? 0 : (ps / _tick_period_ps);
  }

  SsdConfig _cfg;
  uint32_t  _tick_freq_mhz;
  uint64_t  _tick_period_ps;
  uint64_t  _sim_time_ps = 0;
  /* Simulator-injected wall-time (picoseconds). UINT64_MAX = "not set",
   * in which case push() falls back to the internal SSD wall time. */
  uint64_t   _now_ps = UINT64_MAX;

  std::vector<SsdChannelState> _channels;

  /* Pending requests keyed by completion cycle (monotone counter) */
  struct PendingReq {
    uint64_t finish_time_ps;
    MemoryAccess* access;
    bool operator>(const PendingReq& o) const {
      return finish_time_ps > o.finish_time_ps;
    }
  };
  std::priority_queue<PendingReq, std::vector<PendingReq>,
                      std::greater<PendingReq>> _pending;
  std::queue<MemoryAccess*> _finished;

  /* Capacity backpressure */
  uint32_t _max_inflight = 4096;

  /* Stats */
  uint64_t _stat_reads = 0;
  uint64_t _stat_writes = 0;
  uint64_t _stat_total_read_lat_ns = 0;
  uint64_t _stat_total_write_lat_ns = 0;
  uint64_t _stat_max_lat_ns = 0;
  uint64_t _stat_min_lat_ns = UINT64_MAX;
  uint64_t _stat_max_read_lat_ns = 0;
  uint64_t _stat_max_write_lat_ns = 0;
  std::vector<uint64_t> _stat_ch_reads;
  std::vector<uint64_t> _stat_ch_writes;
};

#endif  // SSD_H
