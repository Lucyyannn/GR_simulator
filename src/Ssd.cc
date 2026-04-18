#include "Ssd.h"

#include <algorithm>

/*
 * The algorithm below is a line-by-line C++ port of
 *   extern/femu/bbssd/ftl.c :: ssd_advance_status()
 * for the NAND_READ / NAND_WRITE / NAND_ERASE cases.
 *
 * The original code in FEMU:
 *
 *   case NAND_READ:
 *     nand_stime = max(lun->next_lun_avail_time, cmd_stime);
 *     lun->next_lun_avail_time = nand_stime + spp->pg_rd_lat;
 *     lat = lun->next_lun_avail_time - cmd_stime;
 *     // optional channel xfer block guarded by #if 0 in FEMU source
 *
 *   case NAND_WRITE:
 *     nand_stime = max(lun->next_lun_avail_time, cmd_stime);
 *     lun->next_lun_avail_time = nand_stime + spp->pg_wr_lat;
 *     lat = lun->next_lun_avail_time - cmd_stime;
 *
 *   case NAND_ERASE: similar with blk_er_lat.
 *
 * If ch_xfer_lat > 0, we additionally charge the channel bus time, so
 * that this model can also approximate transfer-bound workloads. This
 * matches the (#if 0'd but conceptually correct) FEMU channel path.
 */

Ssd::Ssd(const SsdConfig& cfg, uint32_t core_freq_mhz)
    : _cfg(cfg),
      _core_freq_mhz(core_freq_mhz > 0 ? core_freq_mhz : 1000),
      _ns_per_cycle(1000.0 / (core_freq_mhz > 0 ? core_freq_mhz : 1000)),
      _cycles(0) {
  _channels.resize(_cfg.nchs);
  for (int c = 0; c < _cfg.nchs; c++) {
    _channels[c].luns.resize(_cfg.luns_per_ch);
  }
  spdlog::info(
      "[SSD] Initialized FEMU-bbssd model: {} channels x {} LUNs, "
      "rd={}ns wr={}ns er={}ns xfer={}ns, addr_base=0x{:x} cap={}GB",
      _cfg.nchs, _cfg.luns_per_ch,
      _cfg.pg_rd_lat, _cfg.pg_wr_lat, _cfg.blk_er_lat, _cfg.ch_xfer_lat,
      _cfg.address_base, _cfg.capacity_bytes / (1ULL << 30));
}

bool Ssd::running() {
  return !_pending.empty() || !_finished.empty();
}

void Ssd::cycle() {
  _cycles++;
  // Move any pending requests whose finish_cycle has passed into the
  // completion queue in time order.
  while (!_pending.empty() && _pending.top().finish_cycle <= _cycles) {
    MemoryAccess* a = _pending.top().access;
    _pending.pop();
    a->request = false;                // mark as response (see Dram.cc)
    a->dram_finish_cycle = _cycles;    // reuse dram_finish_cycle field
    _finished.push(a);
  }
}

void Ssd::address_to_ch_lun(addr_type addr, uint32_t& ch, uint32_t& lun) const {
  uint64_t offset = addr - _cfg.address_base;
  uint64_t page_bytes = (uint64_t)_cfg.secsz * _cfg.secs_per_pg;
  uint64_t page_id = offset / (page_bytes == 0 ? 4096 : page_bytes);
  ch  = (uint32_t)(page_id % (uint32_t)_cfg.nchs);
  lun = (uint32_t)((page_id / _cfg.nchs) % (uint32_t)_cfg.luns_per_ch);
}

uint64_t Ssd::ssd_advance_status(uint32_t ch_idx, uint32_t lun_idx,
                                 SsdCmd cmd, uint64_t cmd_stime_ns) {
  SsdChannelState& ch  = _channels[ch_idx];
  SsdLunState&     lun = ch.luns[lun_idx];

  uint64_t nand_stime;
  uint64_t lat = 0;

  switch (cmd) {
    case SsdCmd::NAND_READ: {
      nand_stime = std::max(lun.next_lun_avail_time, cmd_stime_ns);
      lun.next_lun_avail_time = nand_stime + (uint64_t)_cfg.pg_rd_lat;
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      if (_cfg.ch_xfer_lat > 0) {
        uint64_t chnl_stime = std::max(ch.next_ch_avail_time,
                                       lun.next_lun_avail_time);
        ch.next_ch_avail_time = chnl_stime + (uint64_t)_cfg.ch_xfer_lat;
        lat = ch.next_ch_avail_time - cmd_stime_ns;
      }
      break;
    }
    case SsdCmd::NAND_WRITE: {
      nand_stime = std::max(lun.next_lun_avail_time, cmd_stime_ns);
      lun.next_lun_avail_time = nand_stime + (uint64_t)_cfg.pg_wr_lat;
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      if (_cfg.ch_xfer_lat > 0) {
        uint64_t chnl_stime = std::max(ch.next_ch_avail_time, cmd_stime_ns);
        ch.next_ch_avail_time = chnl_stime + (uint64_t)_cfg.ch_xfer_lat;
        nand_stime = std::max(lun.next_lun_avail_time, ch.next_ch_avail_time);
        lun.next_lun_avail_time = nand_stime + (uint64_t)_cfg.pg_wr_lat;
        lat = lun.next_lun_avail_time - cmd_stime_ns;
      }
      break;
    }
    case SsdCmd::NAND_ERASE: {
      nand_stime = std::max(lun.next_lun_avail_time, cmd_stime_ns);
      lun.next_lun_avail_time = nand_stime + (uint64_t)_cfg.blk_er_lat;
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      break;
    }
  }
  return lat;
}

bool Ssd::is_full(MemoryAccess* request) {
  return _pending.size() >= _max_inflight;
}

void Ssd::push(MemoryAccess* req) {
  uint32_t ch = 0, lun = 0;
  address_to_ch_lun(req->dram_address, ch, lun);

  uint64_t stime_ns = cycles_to_ns(_cycles);
  SsdCmd cmd = req->write ? SsdCmd::NAND_WRITE : SsdCmd::NAND_READ;
  uint64_t lat_ns = ssd_advance_status(ch, lun, cmd, stime_ns);

  cycle_type finish = _cycles + ns_to_cycles(lat_ns);
  if (finish <= _cycles) finish = _cycles + 1;

  _pending.push({finish, req});

  if (req->write) {
    _stat_writes++;
    _stat_total_write_lat_ns += lat_ns;
  } else {
    _stat_reads++;
    _stat_total_read_lat_ns += lat_ns;
  }
  if (lat_ns > _stat_max_lat_ns) _stat_max_lat_ns = lat_ns;

  spdlog::trace(
      "[SSD] push addr=0x{:x} wr={} ch={} lun={} stime={}ns lat={}ns "
      "finish_cycle={}",
      req->dram_address, req->write, ch, lun, stime_ns, lat_ns, finish);
}

bool Ssd::is_empty() { return _finished.empty(); }

MemoryAccess* Ssd::top() {
  assert(!_finished.empty());
  return _finished.front();
}

void Ssd::pop() {
  assert(!_finished.empty());
  _finished.pop();
}

void Ssd::print_stat() {
  uint64_t total = _stat_reads + _stat_writes;
  double avg_rd = _stat_reads  ? (double)_stat_total_read_lat_ns  / _stat_reads  : 0.0;
  double avg_wr = _stat_writes ? (double)_stat_total_write_lat_ns / _stat_writes : 0.0;
  spdlog::info("[SSD] Total IOs: {} (read={}, write={})", total,
               _stat_reads, _stat_writes);
  spdlog::info("[SSD] Avg read  latency: {:.2f} us", avg_rd / 1000.0);
  spdlog::info("[SSD] Avg write latency: {:.2f} us", avg_wr / 1000.0);
  spdlog::info("[SSD] Max latency:      {:.2f} us",
               (double)_stat_max_lat_ns / 1000.0);
}
