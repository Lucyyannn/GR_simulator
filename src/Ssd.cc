#include "Ssd.h"

#include <algorithm>
#include <limits>

namespace {

constexpr uint64_t kInvalidLpn = std::numeric_limits<uint64_t>::max();

}  // namespace

Ssd::Ssd(const SsdConfig& cfg, uint32_t tick_freq_mhz)
    : _cfg(cfg),
      _tick_freq_mhz(tick_freq_mhz > 0 ? tick_freq_mhz : 1000),
      _tick_period_ps((uint64_t)(1000000.0 /
                                 (tick_freq_mhz > 0 ? tick_freq_mhz : 1000))),
      _stat_ch_reads(cfg.nchs, 0),
      _stat_ch_writes(cfg.nchs, 0) {
  init_geometry();
  spdlog::info(
      "[SSD] Initialized FEMU-bbssd-compatible model: {} channels x {} LUNs, "
      "page={}B block={} pages gc={}/{}%",
      _cfg.nchs, _cfg.luns_per_ch, page_bytes(), _cfg.pgs_per_blk,
      _cfg.gc_thres_pcent, _cfg.gc_thres_pcent_high);
}

Ssd::~Ssd() {
  for (auto& [_, merge] : _frontend_merges) {
    for (MemoryAccess* waiter : merge.waiters) delete waiter;
  }
  while (!_to_ftl.empty()) {
    for (MemoryAccess* waiter : _to_ftl.front()->waiters) delete waiter;
    _to_ftl.pop_front();
  }
  while (!_pending_completions.empty()) {
    for (MemoryAccess* waiter : _pending_completions.top().request->waiters) {
      delete waiter;
    }
    _pending_completions.pop();
  }
  while (!_finished.empty()) {
    delete _finished.front();
    _finished.pop();
  }
}

void Ssd::init_geometry() {
  _secs_per_blk = _cfg.secs_per_pg * _cfg.pgs_per_blk;
  _secs_per_pl = _secs_per_blk * _cfg.blks_per_pl;
  _secs_per_lun = _secs_per_pl * _cfg.pls_per_lun;
  _secs_per_ch = _secs_per_lun * _cfg.luns_per_ch;
  _tt_secs = _secs_per_ch * _cfg.nchs;

  _pgs_per_pl = _cfg.pgs_per_blk * _cfg.blks_per_pl;
  _pgs_per_lun = _pgs_per_pl * _cfg.pls_per_lun;
  _pgs_per_ch = _pgs_per_lun * _cfg.luns_per_ch;
  _tt_pgs = _pgs_per_ch * _cfg.nchs;

  _blks_per_lun = _cfg.blks_per_pl * _cfg.pls_per_lun;
  _blks_per_ch = _blks_per_lun * _cfg.luns_per_ch;
  _tt_blks = _blks_per_ch * _cfg.nchs;

  _tt_luns = _cfg.luns_per_ch * _cfg.nchs;
  _blks_per_line = _tt_luns;
  _pgs_per_line = _blks_per_line * _cfg.pgs_per_blk;
  _secs_per_line = _pgs_per_line * _cfg.secs_per_pg;
  _tt_lines = _blks_per_lun;

  _gc_thres_lines =
      static_cast<int>((1.0 - _cfg.gc_thres_pcent / 100.0) * _tt_lines);
  _gc_thres_lines_high =
      static_cast<int>((1.0 - _cfg.gc_thres_pcent_high / 100.0) * _tt_lines);

  _channels.resize(_cfg.nchs);
  for (int ch = 0; ch < _cfg.nchs; ch++) {
    _channels[ch].luns.resize(_cfg.luns_per_ch);
    for (int lun = 0; lun < _cfg.luns_per_ch; lun++) {
      _channels[ch].luns[lun].planes.resize(_cfg.pls_per_lun);
      for (int pl = 0; pl < _cfg.pls_per_lun; pl++) {
        _channels[ch].luns[lun].planes[pl].blocks.resize(_cfg.blks_per_pl);
        for (int blk = 0; blk < _cfg.blks_per_pl; blk++) {
          NandBlock& block = _channels[ch].luns[lun].planes[pl].blocks[blk];
          block.pages.resize(_cfg.pgs_per_blk);
        }
      }
    }
  }

  _maptbl.assign(_tt_pgs, Ppa{});
  _rmap.assign(_tt_pgs, kInvalidLpn);
  init_lines();
  init_write_pointer();
}

void Ssd::init_lines() {
  _lines.resize(_tt_lines);
  _free_lines.clear();
  for (int i = 0; i < _tt_lines; i++) {
    _lines[i].id = i;
    _lines[i].ipc = 0;
    _lines[i].vpc = 0;
    _lines[i].state = Line::State::FREE;
    _free_lines.push_back(i);
  }
}

Ssd::Line* Ssd::get_next_free_line() {
  if (_free_lines.empty()) return nullptr;
  int id = _free_lines.front();
  _free_lines.pop_front();
  _lines[id].state = Line::State::ACTIVE;
  return &_lines[id];
}

void Ssd::init_write_pointer() {
  Line* line = get_next_free_line();
  if (line == nullptr) return;
  _wp.line_id = line->id;
  _wp.ch = 0;
  _wp.lun = 0;
  _wp.pg = 0;
  _wp.blk = line->id;
  _wp.pl = 0;
}

uint64_t Ssd::sector_bytes() const { return static_cast<uint64_t>(_cfg.secsz); }

uint64_t Ssd::page_bytes() const {
  return static_cast<uint64_t>(_cfg.secsz) * _cfg.secs_per_pg;
}

uint64_t Ssd::page_key(addr_type page_addr, bool write) const {
  uint64_t page_id = (page_addr - _cfg.address_base) / std::max<uint64_t>(page_bytes(), 1);
  return (page_id << 1) | (write ? 1ULL : 0ULL);
}

addr_type Ssd::align_page_address(addr_type addr) const {
  uint64_t bytes = page_bytes();
  if (bytes == 0 || addr < _cfg.address_base) return addr;
  uint64_t offset = addr - _cfg.address_base;
  return _cfg.address_base + (offset / bytes) * bytes;
}

uint64_t Ssd::current_time_ps() const {
  return _now_ps != UINT64_MAX ? _now_ps : _sim_time_ps;
}

uint64_t Ssd::current_time_ns() const { return current_time_ps() / 1000ULL; }

uint64_t Ssd::ps_to_tick_cycles(uint64_t ps) const {
  return _tick_period_ps == 0 ? 0 : (ps / _tick_period_ps);
}

bool Ssd::running() {
  return !_frontend_merges.empty() || !_to_ftl.empty() ||
         !_pending_completions.empty() || !_finished.empty();
}

void Ssd::cycle() {
  _sim_time_ps += _tick_period_ps;
  advance_to(_sim_time_ps);
}

bool Ssd::should_route_through_host_frontend(
    const MemoryAccess* request) const {
  return request != nullptr && !request->ssd_host_request &&
         request->size < page_bytes();
}

std::shared_ptr<Ssd::HostRequest> Ssd::make_host_request(addr_type base_addr,
                                                         uint64_t size_bytes,
                                                         bool write,
                                                         uint64_t issue_time_ps) {
  auto request = std::make_shared<HostRequest>();
  request->id = _next_host_request_id++;
  request->write = write;
  request->issue_time_ps = issue_time_ps;
  request->base_addr = base_addr;
  request->size_bytes = size_bytes;

  uint64_t offset = base_addr - _cfg.address_base;
  uint64_t first_sector = offset / std::max<uint64_t>(sector_bytes(), 1);
  uint64_t end_offset = offset + size_bytes;
  uint64_t last_sector = (end_offset + sector_bytes() - 1) /
                         std::max<uint64_t>(sector_bytes(), 1);
  request->slba = first_sector;
  request->nlb = static_cast<uint32_t>(
      std::max<uint64_t>(1, last_sector > first_sector ? last_sector - first_sector : 1));
  return request;
}

void Ssd::enqueue_host_request(const std::shared_ptr<HostRequest>& request) {
  _to_ftl.push_back(request);
}

bool Ssd::is_full(MemoryAccess* /*request*/) {
  return _frontend_merges.size() + _to_ftl.size() + _pending_completions.size() >=
         _max_inflight;
}

void Ssd::push(MemoryAccess* request) {
  request->target_medium = MemoryMedium::SSD;
  request->mem_enter_time_ps = current_time_ps();
  if (request->issue_time_ps == 0) request->issue_time_ps = request->mem_enter_time_ps;
  if (request->logical_size_bytes == 0) request->logical_size_bytes = request->size;

  if (should_route_through_host_frontend(request)) {
    addr_type page_addr = align_page_address(request->dram_address);
    uint64_t key = page_key(page_addr, request->write);
    auto it = _frontend_merges.find(key);
    if (it == _frontend_merges.end()) {
      FrontendMerge merge;
      merge.issue_time_ps = request->issue_time_ps;
      merge.page_addr = page_addr;
      merge.write = request->write;
      merge.waiters.push_back(request);
      _frontend_merges.emplace(key, std::move(merge));
    } else {
      it->second.waiters.push_back(request);
      it->second.issue_time_ps = std::min(it->second.issue_time_ps, request->issue_time_ps);
    }
    return;
  }

  auto host_request = make_host_request(request->dram_address, request->size,
                                        request->write, request->issue_time_ps);
  host_request->waiters.push_back(request);
  enqueue_host_request(host_request);
}

void Ssd::prefill_range(addr_type base_addr, uint64_t size_bytes) {
  if (size_bytes == 0 || !owns_address(base_addr)) return;

  auto request = make_host_request(base_addr, size_bytes, true, 0);
  uint64_t start_lpn = request->slba / std::max<uint64_t>(_cfg.secs_per_pg, 1);
  uint64_t end_lpn =
      (request->slba + std::max<uint32_t>(request->nlb, 1U) - 1) /
      std::max<uint64_t>(_cfg.secs_per_pg, 1);

  for (uint64_t lpn = start_lpn; lpn <= end_lpn; lpn++) {
    if (!valid_lpn(lpn)) break;

    while (_wp.line_id < 0) {
      Line* next = get_next_free_line();
      if (next == nullptr) return;
      _wp.line_id = next->id;
      _wp.blk = next->id;
      _wp.ch = _wp.lun = _wp.pg = _wp.pl = 0;
    }

    Ppa old_ppa = get_maptbl_ent(lpn);
    if (mapped_ppa(old_ppa) && valid_ppa(old_ppa)) {
      mark_page_invalid(old_ppa);
      set_rmap_ent(kInvalidLpn, old_ppa);
    }

    Ppa new_ppa = get_new_page();
    set_maptbl_ent(lpn, new_ppa);
    set_rmap_ent(lpn, new_ppa);
    mark_page_valid(new_ppa);
    advance_write_pointer();
  }
}

void Ssd::flush_frontend_merges(uint64_t now_ps) {
  std::vector<uint64_t> ready_keys;
  ready_keys.reserve(_frontend_merges.size());
  for (const auto& [key, merge] : _frontend_merges) {
    if (merge.issue_time_ps <= now_ps) ready_keys.push_back(key);
  }

  for (uint64_t key : ready_keys) {
    auto it = _frontend_merges.find(key);
    if (it == _frontend_merges.end()) continue;
    auto host_request =
        make_host_request(it->second.page_addr, page_bytes(), it->second.write,
                          it->second.issue_time_ps);
    host_request->waiters = std::move(it->second.waiters);
    enqueue_host_request(host_request);
    _frontend_merges.erase(it);
  }
}

bool Ssd::valid_lpn(uint64_t lpn) const { return lpn < static_cast<uint64_t>(_tt_pgs); }

bool Ssd::valid_ppa(const Ppa& ppa) const {
  return ppa.mapped && ppa.ch >= 0 && ppa.ch < _cfg.nchs && ppa.lun >= 0 &&
         ppa.lun < _cfg.luns_per_ch && ppa.pl >= 0 && ppa.pl < _cfg.pls_per_lun &&
         ppa.blk >= 0 && ppa.blk < _cfg.blks_per_pl && ppa.pg >= 0 &&
         ppa.pg < _cfg.pgs_per_blk && ppa.sec >= 0 && ppa.sec < _cfg.secs_per_pg;
}

bool Ssd::mapped_ppa(const Ppa& ppa) const { return ppa.mapped; }

uint64_t Ssd::ppa_to_pgidx(const Ppa& ppa) const {
  uint64_t pgidx = static_cast<uint64_t>(ppa.ch) * _pgs_per_ch +
                   static_cast<uint64_t>(ppa.lun) * _pgs_per_lun +
                   static_cast<uint64_t>(ppa.pl) * _pgs_per_pl +
                   static_cast<uint64_t>(ppa.blk) * _cfg.pgs_per_blk +
                   static_cast<uint64_t>(ppa.pg);
  assert(pgidx < static_cast<uint64_t>(_tt_pgs));
  return pgidx;
}

Ssd::Ppa Ssd::get_maptbl_ent(uint64_t lpn) const {
  assert(lpn < _maptbl.size());
  return _maptbl[lpn];
}

void Ssd::set_maptbl_ent(uint64_t lpn, const Ppa& ppa) {
  assert(lpn < _maptbl.size());
  _maptbl[lpn] = ppa;
}

uint64_t Ssd::get_rmap_ent(const Ppa& ppa) const { return _rmap[ppa_to_pgidx(ppa)]; }

void Ssd::set_rmap_ent(uint64_t lpn, const Ppa& ppa) {
  _rmap[ppa_to_pgidx(ppa)] = lpn;
}

Ssd::NandLun& Ssd::get_lun(const Ppa& ppa) {
  return _channels[ppa.ch].luns[ppa.lun];
}

const Ssd::NandLun& Ssd::get_lun(const Ppa& ppa) const {
  return _channels[ppa.ch].luns[ppa.lun];
}

Ssd::NandBlock& Ssd::get_blk(const Ppa& ppa) {
  return _channels[ppa.ch].luns[ppa.lun].planes[ppa.pl].blocks[ppa.blk];
}

Ssd::NandPage& Ssd::get_pg(const Ppa& ppa) { return get_blk(ppa).pages[ppa.pg]; }

Ssd::Line& Ssd::get_line(const Ppa& ppa) { return _lines[ppa.blk]; }

uint64_t Ssd::ssd_advance_status(const Ppa& ppa, SsdCmd cmd,
                                 uint64_t cmd_stime_ns) {
  NandLun& lun = get_lun(ppa);
  SsdChannelState& ch = _channels[ppa.ch];
  uint64_t nand_stime = std::max(lun.next_lun_avail_time, cmd_stime_ns);
  uint64_t lat = 0;

  switch (cmd) {
    case SsdCmd::NAND_READ:
      lun.next_lun_avail_time = nand_stime + static_cast<uint64_t>(_cfg.pg_rd_lat);
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      if (ppa.ch < static_cast<int>(_stat_ch_reads.size())) _stat_ch_reads[ppa.ch]++;
      break;
    case SsdCmd::NAND_WRITE:
      lun.next_lun_avail_time = nand_stime + static_cast<uint64_t>(_cfg.pg_wr_lat);
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      if (ppa.ch < static_cast<int>(_stat_ch_writes.size())) _stat_ch_writes[ppa.ch]++;
      break;
    case SsdCmd::NAND_ERASE:
      lun.next_lun_avail_time =
          nand_stime + static_cast<uint64_t>(_cfg.blk_er_lat);
      lat = lun.next_lun_avail_time - cmd_stime_ns;
      break;
  }

  if (_cfg.ch_xfer_lat > 0 && cmd != SsdCmd::NAND_ERASE) {
    if (cmd == SsdCmd::NAND_READ) {
      uint64_t chnl_stime = std::max(ch.next_ch_avail_time, lun.next_lun_avail_time);
      ch.next_ch_avail_time = chnl_stime + static_cast<uint64_t>(_cfg.ch_xfer_lat);
      lat = ch.next_ch_avail_time - cmd_stime_ns;
    } else {
      uint64_t chnl_stime = std::max(ch.next_ch_avail_time, cmd_stime_ns);
      ch.next_ch_avail_time = chnl_stime + static_cast<uint64_t>(_cfg.ch_xfer_lat);
      nand_stime = std::max(lun.next_lun_avail_time, ch.next_ch_avail_time);
      lun.next_lun_avail_time = nand_stime + static_cast<uint64_t>(_cfg.pg_wr_lat);
      lat = lun.next_lun_avail_time - cmd_stime_ns;
    }
  }
  return lat;
}

void Ssd::mark_page_valid(const Ppa& ppa) {
  NandPage& page = get_pg(ppa);
  assert(page.status == PG_FREE);
  page.status = PG_VALID;

  NandBlock& block = get_blk(ppa);
  block.vpc++;

  Line& line = get_line(ppa);
  line.vpc++;
}

void Ssd::mark_page_invalid(const Ppa& ppa) {
  NandPage& page = get_pg(ppa);
  if (page.status != PG_VALID) return;
  page.status = PG_INVALID;

  NandBlock& block = get_blk(ppa);
  block.ipc++;
  block.vpc--;

  Line& line = get_line(ppa);
  bool was_full = (line.vpc == _pgs_per_line);
  line.ipc++;
  line.vpc--;
  if (was_full) {
    line.state = Line::State::VICTIM;
  }
}

void Ssd::mark_block_free(const Ppa& ppa) {
  NandBlock& block = get_blk(ppa);
  for (NandPage& page : block.pages) page.status = PG_FREE;
  block.ipc = 0;
  block.vpc = 0;
  block.erase_cnt++;
  block.wp = 0;
}

Ssd::Ppa Ssd::get_new_page() const {
  Ppa ppa;
  ppa.ch = _wp.ch;
  ppa.lun = _wp.lun;
  ppa.pl = _wp.pl;
  ppa.blk = _wp.blk;
  ppa.pg = _wp.pg;
  ppa.sec = 0;
  ppa.mapped = true;
  return ppa;
}

void Ssd::advance_write_pointer() {
  if (_wp.line_id < 0) return;

  _wp.ch++;
  if (_wp.ch != _cfg.nchs) return;
  _wp.ch = 0;

  _wp.lun++;
  if (_wp.lun != _cfg.luns_per_ch) return;
  _wp.lun = 0;

  _wp.pg++;
  if (_wp.pg != _cfg.pgs_per_blk) return;
  _wp.pg = 0;

  Line& line = _lines[_wp.line_id];
  if (line.vpc == _pgs_per_line) {
    line.state = Line::State::FULL;
  } else {
    line.state = Line::State::VICTIM;
  }

  _wp.line_id = -1;
  Line* next = get_next_free_line();
  if (next == nullptr) {
    _wp.blk = -1;
    return;
  }
  _wp.line_id = next->id;
  _wp.blk = next->id;
  _wp.pl = 0;
}

bool Ssd::should_gc() const {
  return static_cast<int>(_free_lines.size()) <= _gc_thres_lines;
}

bool Ssd::should_gc_high() const {
  return static_cast<int>(_free_lines.size()) <= _gc_thres_lines_high;
}

Ssd::Line* Ssd::select_victim_line(bool force) {
  Line* victim = nullptr;
  for (Line& line : _lines) {
    if (line.state != Line::State::VICTIM) continue;
    if (victim == nullptr || line.vpc < victim->vpc) victim = &line;
  }
  if (victim == nullptr) return nullptr;
  if (!force && victim->ipc < _pgs_per_line / 8) return nullptr;
  victim->state = Line::State::ACTIVE;
  return victim;
}

void Ssd::gc_read_page(const Ppa& ppa, uint64_t now_ns) {
  (void)ssd_advance_status(ppa, SsdCmd::NAND_READ, now_ns);
}

void Ssd::gc_write_page(const Ppa& old_ppa, uint64_t now_ns) {
  uint64_t lpn = get_rmap_ent(old_ppa);
  if (!valid_lpn(lpn)) return;
  while (_wp.line_id < 0) {
    if (do_gc(true, now_ns) == -1) return;
    if (_wp.line_id < 0) {
      Line* next = get_next_free_line();
      if (next == nullptr) return;
      _wp.line_id = next->id;
      _wp.blk = next->id;
      _wp.ch = _wp.lun = _wp.pg = _wp.pl = 0;
    }
  }

  Ppa new_ppa = get_new_page();
  set_maptbl_ent(lpn, new_ppa);
  set_rmap_ent(lpn, new_ppa);
  mark_page_valid(new_ppa);
  advance_write_pointer();
  (void)ssd_advance_status(new_ppa, SsdCmd::NAND_WRITE, now_ns);
  get_lun(new_ppa).gc_endtime = get_lun(new_ppa).next_lun_avail_time;
}

void Ssd::clean_one_block(Ppa ppa, uint64_t now_ns) {
  for (int pg = 0; pg < _cfg.pgs_per_blk; pg++) {
    ppa.pg = pg;
    NandPage& page = get_pg(ppa);
    if (page.status == PG_VALID) {
      gc_read_page(ppa, now_ns);
      gc_write_page(ppa, now_ns);
    }
  }
}

void Ssd::mark_line_free(const Ppa& ppa) {
  Line& line = get_line(ppa);
  line.ipc = 0;
  line.vpc = 0;
  line.state = Line::State::FREE;
  _free_lines.push_back(line.id);
}

int Ssd::do_gc(bool force, uint64_t now_ns) {
  Line* victim = select_victim_line(force);
  if (victim == nullptr) return -1;

  Ppa ppa;
  ppa.blk = victim->id;
  ppa.pl = 0;
  ppa.mapped = true;

  for (int ch = 0; ch < _cfg.nchs; ch++) {
    for (int lun = 0; lun < _cfg.luns_per_ch; lun++) {
      ppa.ch = ch;
      ppa.lun = lun;
      clean_one_block(ppa, now_ns);
      mark_block_free(ppa);
      (void)ssd_advance_status(ppa, SsdCmd::NAND_ERASE, now_ns);
      get_lun(ppa).gc_endtime = get_lun(ppa).next_lun_avail_time;
    }
  }

  mark_line_free(ppa);
  _stat_gc_runs++;
  return 0;
}

uint64_t Ssd::process_read(const HostRequest& request, uint64_t issue_time_ns) {
  uint64_t start_lpn = request.slba / std::max<uint64_t>(_cfg.secs_per_pg, 1);
  uint64_t end_lpn =
      (request.slba + std::max<uint32_t>(request.nlb, 1U) - 1) /
      std::max<uint64_t>(_cfg.secs_per_pg, 1);

  uint64_t maxlat = 0;
  for (uint64_t lpn = start_lpn; lpn <= end_lpn; lpn++) {
    if (!valid_lpn(lpn)) continue;
    Ppa ppa = get_maptbl_ent(lpn);
    if (!mapped_ppa(ppa) || !valid_ppa(ppa) || get_pg(ppa).status != PG_VALID) continue;
    uint64_t sublat = ssd_advance_status(ppa, SsdCmd::NAND_READ, issue_time_ns);
    maxlat = std::max(maxlat, sublat);
  }
  return maxlat;
}

uint64_t Ssd::process_write(const HostRequest& request, uint64_t issue_time_ns) {
  uint64_t start_lpn = request.slba / std::max<uint64_t>(_cfg.secs_per_pg, 1);
  uint64_t end_lpn =
      (request.slba + std::max<uint32_t>(request.nlb, 1U) - 1) /
      std::max<uint64_t>(_cfg.secs_per_pg, 1);

  while (should_gc_high()) {
    if (do_gc(true, issue_time_ns) == -1) break;
  }

  uint64_t maxlat = 0;
  for (uint64_t lpn = start_lpn; lpn <= end_lpn; lpn++) {
    if (!valid_lpn(lpn)) break;

    while (_wp.line_id < 0) {
      if (do_gc(true, issue_time_ns) == -1) return maxlat;
      if (_wp.line_id < 0) {
        Line* next = get_next_free_line();
        if (next == nullptr) return maxlat;
        _wp.line_id = next->id;
        _wp.blk = next->id;
        _wp.ch = _wp.lun = _wp.pg = _wp.pl = 0;
      }
    }

    Ppa old_ppa = get_maptbl_ent(lpn);
    if (mapped_ppa(old_ppa) && valid_ppa(old_ppa)) {
      mark_page_invalid(old_ppa);
      set_rmap_ent(kInvalidLpn, old_ppa);
    }

    Ppa new_ppa = get_new_page();
    set_maptbl_ent(lpn, new_ppa);
    set_rmap_ent(lpn, new_ppa);
    mark_page_valid(new_ppa);
    advance_write_pointer();

    uint64_t curlat = ssd_advance_status(new_ppa, SsdCmd::NAND_WRITE, issue_time_ns);
    maxlat = std::max(maxlat, curlat);
  }
  return maxlat;
}

uint64_t Ssd::process_trim(const HostRequest& request, uint64_t /*issue_time_ns*/) {
  uint64_t start_lpn = request.slba / std::max<uint64_t>(_cfg.secs_per_pg, 1);
  uint64_t end_lpn =
      (request.slba + std::max<uint32_t>(request.nlb, 1U) - 1) /
      std::max<uint64_t>(_cfg.secs_per_pg, 1);

  for (uint64_t lpn = start_lpn; lpn <= end_lpn; lpn++) {
    if (!valid_lpn(lpn)) continue;
    Ppa ppa = get_maptbl_ent(lpn);
    if (!mapped_ppa(ppa) || !valid_ppa(ppa)) continue;
    mark_page_invalid(ppa);
    set_rmap_ent(kInvalidLpn, ppa);
    set_maptbl_ent(lpn, Ppa{});
  }
  return 0;
}

void Ssd::finalize_host_request(const std::shared_ptr<HostRequest>& request,
                                uint64_t finish_time_ps) {
  for (MemoryAccess* waiter : request->waiters) {
    waiter->request = false;
    waiter->target_medium = MemoryMedium::SSD;
    waiter->mem_finish_time_ps = finish_time_ps;
    waiter->dram_finish_cycle = ps_to_tick_cycles(finish_time_ps);
    _finished.push(waiter);
  }
}

void Ssd::process_queued_host_requests(uint64_t now_ps) {
  while (!_to_ftl.empty() && _to_ftl.front()->issue_time_ps <= now_ps) {
    std::shared_ptr<HostRequest> request = _to_ftl.front();
    _to_ftl.pop_front();

    uint64_t issue_time_ns = request->issue_time_ps / 1000ULL;
    uint64_t lat_ns = 0;
    if (request->trim) {
      lat_ns = process_trim(*request, issue_time_ns);
    } else if (request->write) {
      lat_ns = process_write(*request, issue_time_ns);
      _stat_writes++;
      _stat_total_write_lat_ns += lat_ns;
      _stat_max_write_lat_ns = std::max(_stat_max_write_lat_ns, lat_ns);
    } else {
      lat_ns = process_read(*request, issue_time_ns);
      _stat_reads++;
      _stat_total_read_lat_ns += lat_ns;
      _stat_max_read_lat_ns = std::max(_stat_max_read_lat_ns, lat_ns);
    }

    _stat_max_lat_ns = std::max(_stat_max_lat_ns, lat_ns);
    _stat_min_lat_ns = std::min(_stat_min_lat_ns, lat_ns);

    uint64_t finish_time_ps = request->issue_time_ps + lat_ns * 1000ULL;
    if (finish_time_ps <= request->issue_time_ps) finish_time_ps = request->issue_time_ps + 1;
    _pending_completions.push({finish_time_ps, request});
  }
}

void Ssd::run_background_gc(uint64_t now_ps) {
  uint64_t now_ns = now_ps / 1000ULL;
  int gc_iters = 0;
  int max_gc_iters = std::max(_tt_lines, 1);
  while (should_gc() && gc_iters < max_gc_iters) {
    if (do_gc(false, now_ns) == -1) break;
    gc_iters++;
  }
}

void Ssd::complete_finished_requests(uint64_t now_ps) {
  while (!_pending_completions.empty() &&
         _pending_completions.top().finish_time_ps <= now_ps) {
    Completion completion = _pending_completions.top();
    _pending_completions.pop();
    finalize_host_request(completion.request, completion.finish_time_ps);
  }
}

void Ssd::advance_to(uint64_t now_ps) {
  _sim_time_ps = std::max(_sim_time_ps, now_ps);
  flush_frontend_merges(now_ps);
  process_queued_host_requests(now_ps);
  run_background_gc(now_ps);
  complete_finished_requests(now_ps);
}

uint64_t Ssd::next_event_time_ps() const {
  uint64_t next_ps = std::numeric_limits<uint64_t>::max();
  if (!_pending_completions.empty()) {
    next_ps = std::min(next_ps, _pending_completions.top().finish_time_ps);
  }
  if (!_to_ftl.empty()) next_ps = std::min(next_ps, _to_ftl.front()->issue_time_ps);
  for (const auto& [_, merge] : _frontend_merges) {
    next_ps = std::min(next_ps, merge.issue_time_ps);
  }
  if (!_finished.empty()) next_ps = std::min(next_ps, current_time_ps());
  return next_ps;
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
  double avg_rd =
      _stat_reads ? static_cast<double>(_stat_total_read_lat_ns) / _stat_reads : 0.0;
  double avg_wr = _stat_writes
                      ? static_cast<double>(_stat_total_write_lat_ns) / _stat_writes
                      : 0.0;
  double min_lat = _stat_min_lat_ns < UINT64_MAX ? static_cast<double>(_stat_min_lat_ns)
                                                 : 0.0;
  spdlog::info("[SSD] Host IOs: {} (read={}, write={})", total, _stat_reads,
               _stat_writes);
  spdlog::info("[SSD] Avg read  latency: {:.2f} us", avg_rd / 1000.0);
  spdlog::info("[SSD] Avg write latency: {:.2f} us", avg_wr / 1000.0);
  spdlog::info("[SSD] Min latency:       {:.2f} us", min_lat / 1000.0);
  spdlog::info("[SSD] Max latency:       {:.2f} us",
               static_cast<double>(_stat_max_lat_ns) / 1000.0);
  spdlog::info("[SSD] Max read latency:  {:.2f} us",
               static_cast<double>(_stat_max_read_lat_ns) / 1000.0);
  spdlog::info("[SSD] Max write latency: {:.2f} us",
               static_cast<double>(_stat_max_write_lat_ns) / 1000.0);
  spdlog::info("[SSD] GC runs: {}", _stat_gc_runs);
  for (int ch = 0; ch < _cfg.nchs; ch++) {
    uint64_t ch_rd = ch < static_cast<int>(_stat_ch_reads.size()) ? _stat_ch_reads[ch] : 0;
    uint64_t ch_wr =
        ch < static_cast<int>(_stat_ch_writes.size()) ? _stat_ch_writes[ch] : 0;
    spdlog::info("[SSD] CH{}: reads={} writes={}", ch, ch_rd, ch_wr);
  }
}
