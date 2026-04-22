#ifndef SSD_H
#define SSD_H

#include <robin_hood.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <queue>
#include <vector>

#include "Common.h"

enum class SsdCmd {
  NAND_READ = 0,
  NAND_WRITE = 1,
  NAND_ERASE = 2,
};

struct SsdConfig {
  uint64_t address_base = 0x800000000ULL;
  uint64_t capacity_bytes = (1ULL << 40);

  int secsz = 512;
  int secs_per_pg = 8;
  int pgs_per_blk = 256;
  int blks_per_pl = 256;
  int pls_per_lun = 1;
  int luns_per_ch = 8;
  int nchs = 8;

  int pg_rd_lat = 40000;
  int pg_wr_lat = 200000;
  int blk_er_lat = 2000000;
  int ch_xfer_lat = 0;

  int gc_thres_pcent = 75;
  int gc_thres_pcent_high = 95;
};

class Ssd {
 public:
  explicit Ssd(const SsdConfig& cfg, uint32_t tick_freq_mhz);
  ~Ssd();

  bool running();
  void cycle();
  void advance_to(uint64_t now_ps);
  uint64_t next_event_time_ps() const;
  void print_stat();

  void set_current_time_ps(uint64_t ps) { _now_ps = ps; }

  bool owns_address(addr_type addr) const {
    return addr >= _cfg.address_base &&
           addr < _cfg.address_base + _cfg.capacity_bytes;
  }

  bool is_full(MemoryAccess* request);
  void push(MemoryAccess* request);

  bool is_empty();
  MemoryAccess* top();
  void pop();

  uint64_t total_reads() const { return _stat_reads; }
  uint64_t total_writes() const { return _stat_writes; }

 private:
  enum NandStatus : uint8_t {
    SEC_FREE = 0,
    SEC_INVALID = 1,
    SEC_VALID = 2,
    PG_FREE = 0,
    PG_INVALID = 1,
    PG_VALID = 2,
  };

  struct Ppa {
    int ch = -1;
    int lun = -1;
    int pl = -1;
    int blk = -1;
    int pg = -1;
    int sec = 0;
    bool mapped = false;
  };

  struct NandPage {
    uint8_t status = PG_FREE;
  };

  struct NandBlock {
    std::vector<NandPage> pages;
    int ipc = 0;
    int vpc = 0;
    int erase_cnt = 0;
    int wp = 0;
  };

  struct NandPlane {
    std::vector<NandBlock> blocks;
  };

  struct NandLun {
    std::vector<NandPlane> planes;
    uint64_t next_lun_avail_time = 0;
    bool busy = false;
    uint64_t gc_endtime = 0;
  };

  struct SsdChannelState {
    std::vector<NandLun> luns;
    uint64_t next_ch_avail_time = 0;
    bool busy = false;
    uint64_t gc_endtime = 0;
  };

  struct Line {
    enum class State : uint8_t {
      FREE = 0,
      ACTIVE = 1,
      VICTIM = 2,
      FULL = 3,
    };

    int id = 0;
    int ipc = 0;
    int vpc = 0;
    State state = State::FREE;
  };

  struct WritePointer {
    int line_id = -1;
    int ch = 0;
    int lun = 0;
    int pg = 0;
    int blk = 0;
    int pl = 0;
  };

  struct HostRequest {
    uint64_t id = 0;
    bool write = false;
    bool trim = false;
    uint64_t issue_time_ps = 0;
    addr_type base_addr = 0;
    uint64_t size_bytes = 0;
    uint64_t slba = 0;
    uint32_t nlb = 0;
    std::vector<MemoryAccess*> waiters;
  };

  struct FrontendMerge {
    uint64_t issue_time_ps = 0;
    addr_type page_addr = 0;
    bool write = false;
    std::vector<MemoryAccess*> waiters;
  };

  struct Completion {
    uint64_t finish_time_ps = 0;
    std::shared_ptr<HostRequest> request;

    bool operator>(const Completion& other) const {
      return finish_time_ps > other.finish_time_ps;
    }
  };

  void init_geometry();
  void init_lines();
  void init_write_pointer();

  uint64_t sector_bytes() const;
  uint64_t page_bytes() const;
  uint64_t page_key(addr_type page_addr, bool write) const;
  addr_type align_page_address(addr_type addr) const;
  uint64_t current_time_ps() const;
  uint64_t current_time_ns() const;
  uint64_t ps_to_tick_cycles(uint64_t ps) const;

  bool should_route_through_host_frontend(const MemoryAccess* request) const;
  std::shared_ptr<HostRequest> make_host_request(addr_type base_addr,
                                                 uint64_t size_bytes,
                                                 bool write,
                                                 uint64_t issue_time_ps);
  void enqueue_host_request(const std::shared_ptr<HostRequest>& request);
  void flush_frontend_merges(uint64_t now_ps);
  void process_queued_host_requests(uint64_t now_ps);
  void complete_finished_requests(uint64_t now_ps);
  void run_background_gc(uint64_t now_ps);

  bool valid_lpn(uint64_t lpn) const;
  bool valid_ppa(const Ppa& ppa) const;
  bool mapped_ppa(const Ppa& ppa) const;
  uint64_t ppa_to_pgidx(const Ppa& ppa) const;
  Ppa get_maptbl_ent(uint64_t lpn) const;
  void set_maptbl_ent(uint64_t lpn, const Ppa& ppa);
  uint64_t get_rmap_ent(const Ppa& ppa) const;
  void set_rmap_ent(uint64_t lpn, const Ppa& ppa);

  NandLun& get_lun(const Ppa& ppa);
  const NandLun& get_lun(const Ppa& ppa) const;
  NandBlock& get_blk(const Ppa& ppa);
  NandPage& get_pg(const Ppa& ppa);
  Line& get_line(const Ppa& ppa);

  uint64_t ssd_advance_status(const Ppa& ppa, SsdCmd cmd, uint64_t cmd_stime_ns);

  void mark_page_valid(const Ppa& ppa);
  void mark_page_invalid(const Ppa& ppa);
  void mark_block_free(const Ppa& ppa);

  Ppa get_new_page() const;
  Line* get_next_free_line();
  void advance_write_pointer();

  bool should_gc() const;
  bool should_gc_high() const;
  Line* select_victim_line(bool force);
  void gc_read_page(const Ppa& ppa, uint64_t now_ns);
  void gc_write_page(const Ppa& old_ppa, uint64_t now_ns);
  void clean_one_block(Ppa ppa, uint64_t now_ns);
  void mark_line_free(const Ppa& ppa);
  int do_gc(bool force, uint64_t now_ns);

  uint64_t process_read(const HostRequest& request, uint64_t issue_time_ns);
  uint64_t process_write(const HostRequest& request, uint64_t issue_time_ns);
  uint64_t process_trim(const HostRequest& request, uint64_t issue_time_ns);
  void finalize_host_request(const std::shared_ptr<HostRequest>& request,
                             uint64_t finish_time_ps);

  SsdConfig _cfg;
  uint32_t _tick_freq_mhz = 0;
  uint64_t _tick_period_ps = 0;
  uint64_t _sim_time_ps = 0;
  uint64_t _now_ps = UINT64_MAX;
  uint64_t _next_host_request_id = 1;
  uint32_t _max_inflight = 4096;

  int _secs_per_blk = 0;
  int _secs_per_pl = 0;
  int _secs_per_lun = 0;
  int _secs_per_ch = 0;
  int _tt_secs = 0;
  int _pgs_per_pl = 0;
  int _pgs_per_lun = 0;
  int _pgs_per_ch = 0;
  int _tt_pgs = 0;
  int _blks_per_lun = 0;
  int _blks_per_ch = 0;
  int _tt_blks = 0;
  int _blks_per_line = 0;
  int _pgs_per_line = 0;
  int _secs_per_line = 0;
  int _tt_lines = 0;
  int _tt_luns = 0;
  int _gc_thres_lines = 0;
  int _gc_thres_lines_high = 0;

  std::vector<SsdChannelState> _channels;
  std::vector<Ppa> _maptbl;
  std::vector<uint64_t> _rmap;
  std::vector<Line> _lines;
  std::deque<int> _free_lines;
  WritePointer _wp;

  robin_hood::unordered_flat_map<uint64_t, FrontendMerge> _frontend_merges;
  std::deque<std::shared_ptr<HostRequest>> _to_ftl;
  std::priority_queue<Completion, std::vector<Completion>, std::greater<Completion>>
      _pending_completions;
  std::queue<MemoryAccess*> _finished;

  uint64_t _stat_reads = 0;
  uint64_t _stat_writes = 0;
  uint64_t _stat_total_read_lat_ns = 0;
  uint64_t _stat_total_write_lat_ns = 0;
  uint64_t _stat_max_lat_ns = 0;
  uint64_t _stat_min_lat_ns = UINT64_MAX;
  uint64_t _stat_max_read_lat_ns = 0;
  uint64_t _stat_max_write_lat_ns = 0;
  uint64_t _stat_gc_runs = 0;
  std::vector<uint64_t> _stat_ch_reads;
  std::vector<uint64_t> _stat_ch_writes;
};

#endif  // SSD_H
