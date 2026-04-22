#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../Common.h"
#include "../Dram.h"
#include "../Ssd.h"
#include "../SimulationConfig.h"
#include "../memory/StorageController.h"

enum class BenchmarkMedium {
  DRAM = 0,
  SSD = 1,
};

enum class BenchmarkIssueMode {
  BACK_TO_BACK = 0,
  SERIALIZED = 1,
};

struct MemBenchmarkCase {
  BenchmarkMedium medium = BenchmarkMedium::DRAM;
  bool write = false;
  uint64_t access_size_bytes = 512;
  uint32_t burst_count = 1;
  BenchmarkIssueMode issue_mode = BenchmarkIssueMode::BACK_TO_BACK;
  std::string address_pattern = "contiguous";
};

class MemBenchmarkRunner {
 public:
  MemBenchmarkRunner(const SimulationConfig& config, const json& bench_config,
                     const std::string& output_dir);

  void run();

 private:
  struct MacroStats {
    uint64_t issue_time_ps = 0;
    uint64_t first_return_ps = UINT64_MAX;
    uint64_t last_return_ps = 0;
    uint64_t completed_subrequests = 0;
    uint64_t total_subrequests = 0;
  };

  struct ScheduledAccess {
    uint64_t issue_time_ps = 0;
    MemoryAccess* request = nullptr;
  };

  std::vector<MemBenchmarkCase> expand_cases() const;
  void run_case(uint64_t case_id, const MemBenchmarkCase& bench_case,
                std::ofstream& summary_csv, std::ofstream& detail_csv);
  void seed_macro_requests(const MemBenchmarkCase& bench_case,
                           uint32_t macro_request_id, uint64_t issue_time_ps,
                           std::deque<ScheduledAccess>& pending_issues,
                           std::vector<MacroStats>& macro_stats) const;
  void write_summary_header(std::ofstream& summary_csv) const;
  void write_detail_header(std::ofstream& detail_csv) const;

  std::unique_ptr<Dram> create_dram() const;
  std::unique_ptr<Ssd> create_ssd() const;

  uint64_t round_up(uint64_t value, uint64_t alignment) const;
  uint64_t next_address(BenchmarkMedium medium, uint32_t macro_request_id,
                        uint32_t subrequest_id, uint64_t access_size_bytes,
                        const std::string& address_pattern) const;
  std::string medium_to_string(BenchmarkMedium medium) const;
  std::string issue_mode_to_string(BenchmarkIssueMode mode) const;
  std::string rw_to_string(bool write) const;

  SimulationConfig _config;
  json _bench_config;
  std::string _output_dir;
};
