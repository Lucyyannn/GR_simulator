#include "Ssd.h"

#include <algorithm>

Ssd::Ssd(const SsdConfig& cfg, uint32_t tick_freq_mhz)
    : _cfg(cfg),
      _tick_freq_mhz(tick_freq_mhz > 0 ? tick_freq_mhz : 1000),
      _ns_per_cycle(1000.0 / (tick_freq_mhz > 0 ? tick_freq_mhz : 1000)),
      _cycles(0),
      _stat_ch_reads(cfg.nchs, 0),
      _stat_ch_writes(cfg.nchs, 0) {
  _channels.resize(_cfg.nchs);
  for (int c = 0; c < _cfg.nchs; c++) {
    _channels[c].luns.resize(_cfg.luns_per_ch);
  }
  spdlog::info(
      "[SSD] Initialized FEMU-bbssd model: {} channels x {} LUNs, "
      "rd={}ns wr={}ns er={}ns xfer={}ns, addr_base=0x{:x} cap={}GB, tick_freq={}MHz",
      _cfg.nchs, _cfg.luns_per_ch,
      _cfg.pg_rd_lat, _cfg.pg_wr_lat, _cfg.blk_er_lat, _cfg.ch_xfer_lat,
      _cfg.address_base, _cfg.capacity_bytes / (1ULL << 30), _tick_freq_mhz);
}

Ssd::~Ssd() {
  while (!_pending.empty()) {
    delete _pending.top().access;
    _pending.pop();
  }
  while (!_finished.empty()) {
    delete _finished.front();
    _finished.pop();
  }
}

bool Ssd::running() {
  return !_pending.empty() || !_finished.empty();
}

void Ssd::cycle() {
  _cycles++;
  while (!_pending.empty() && _pending.top().finish_cycle <= _cycles) {
    MemoryAccess* a = _pending.top().access;
    _pending.pop();
    a->request = false;
    a->dram_finish_cycle = _cycles;
    _finished.push(a);
    spdlog::debug("[SSD] complete addr=0x{:x} wr={} finish_cycle={} pending={}",
                  a->dram_address, a->write, _cycles, _pending.size());
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
    if (lat_ns > _stat_max_write_lat_ns) _stat_max_write_lat_ns = lat_ns;
    if (ch < _stat_ch_writes.size()) _stat_ch_writes[ch]++;
  } else {
    _stat_reads++;
    _stat_total_read_lat_ns += lat_ns;
    if (lat_ns > _stat_max_read_lat_ns) _stat_max_read_lat_ns = lat_ns;
    if (ch < _stat_ch_reads.size()) _stat_ch_reads[ch]++;
  }
  if (lat_ns > _stat_max_lat_ns) _stat_max_lat_ns = lat_ns;
  if (lat_ns < _stat_min_lat_ns) _stat_min_lat_ns = lat_ns;

  spdlog::debug(
      "[SSD] push addr=0x{:x} wr={} ch={} lun={} core={} spad=0x{:x} "
      "stime={:.1f}ns lat={:.1f}ns finish_cycle={} pending={}",
      req->dram_address, req->write, ch, lun, req->core_id,
      req->spad_address, (double)stime_ns, (double)lat_ns,
      finish, _pending.size());
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
  double min_lat = _stat_min_lat_ns < UINT64_MAX ? (double)_stat_min_lat_ns : 0.0;
  spdlog::info("[SSD] Total IOs: {} (read={}, write={})", total,
               _stat_reads, _stat_writes);
  spdlog::info("[SSD] Avg read  latency: {:.2f} us", avg_rd / 1000.0);
  spdlog::info("[SSD] Avg write latency: {:.2f} us", avg_wr / 1000.0);
  spdlog::info("[SSD] Min latency:      {:.2f} us", min_lat / 1000.0);
  spdlog::info("[SSD] Max latency:       {:.2f} us",
               (double)_stat_max_lat_ns / 1000.0);
  spdlog::info("[SSD] Max read  latency: {:.2f} us",
               (double)_stat_max_read_lat_ns / 1000.0);
  spdlog::info("[SSD] Max write latency: {:.2f} us",
               (double)_stat_max_write_lat_ns / 1000.0);
  for (int c = 0; c < _cfg.nchs; c++) {
    uint64_t ch_rd = c < (int)_stat_ch_reads.size() ? _stat_ch_reads[c] : 0;
    uint64_t ch_wr = c < (int)_stat_ch_writes.size() ? _stat_ch_writes[c] : 0;
    spdlog::info("[SSD] CH{}: reads={} writes={}", c, ch_rd, ch_wr);
  }
}
