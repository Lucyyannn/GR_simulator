#include "MemBenchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <numeric>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace {

std::vector<uint64_t> parse_uint64_list(const json& config, const std::string& key,
                                        const std::vector<uint64_t>& defaults) {
  if (!config.contains(key)) return defaults;
  std::vector<uint64_t> values;
  for (const auto& entry : config.at(key)) {
    values.push_back(entry.get<uint64_t>());
  }
  return values.empty() ? defaults : values;
}

std::vector<std::string> parse_string_list(const json& config,
                                           const std::string& key,
                                           const std::vector<std::string>& defaults) {
  if (!config.contains(key)) return defaults;
  std::vector<std::string> values;
  for (const auto& entry : config.at(key)) {
    values.push_back(entry.get<std::string>());
  }
  return values.empty() ? defaults : values;
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  double rank = pct * static_cast<double>(values.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(rank));
  size_t upper = static_cast<size_t>(std::ceil(rank));
  if (lower == upper) return values[lower];
  double weight = rank - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

}  // namespace

MemBenchmarkRunner::MemBenchmarkRunner(const SimulationConfig& config,
                                       const json& bench_config,
                                       const std::string& output_dir)
    : _config(config), _bench_config(bench_config), _output_dir(output_dir) {}

void MemBenchmarkRunner::run() {
  fs::create_directories(_output_dir);
  fs::path summary_path = fs::path(_output_dir) / "summary.csv";
  fs::path detail_path = fs::path(_output_dir) / "detail.csv";
  std::ofstream summary_csv(summary_path);
  std::ofstream detail_csv(detail_path);
  if (!summary_csv || !detail_csv) {
    throw std::runtime_error(
        fmt::format("Failed to open benchmark output files under {}", _output_dir));
  }

  write_summary_header(summary_csv);
  write_detail_header(detail_csv);

  std::vector<MemBenchmarkCase> cases = expand_cases();
  spdlog::info("[mem_bench] Running {} benchmark cases, output={}",
               cases.size(), _output_dir);
  for (uint64_t case_id = 0; case_id < cases.size(); case_id++) {
    run_case(case_id, cases[case_id], summary_csv, detail_csv);
  }
}

std::vector<MemBenchmarkCase> MemBenchmarkRunner::expand_cases() const {
  std::vector<uint64_t> size_values = parse_uint64_list(
      _bench_config, "sizes_bytes",
      {512, 1024, 2048, 4096, 8192, 16384});
  std::vector<uint64_t> burst_values = parse_uint64_list(
      _bench_config, "burst_counts", {1, 2, 4, 8, 16, 32, 64, 128});
  std::vector<std::string> medium_values = parse_string_list(
      _bench_config, "media", {"dram", "ssd"});
  std::vector<std::string> access_values = parse_string_list(
      _bench_config, "access_types", {"read", "write"});
  std::vector<std::string> issue_mode_values = parse_string_list(
      _bench_config, "issue_modes", {"back_to_back"});
  std::string address_pattern =
      _bench_config.value("address_pattern", std::string("contiguous"));

  std::vector<MemBenchmarkCase> cases;
  for (const std::string& medium_name : medium_values) {
    BenchmarkMedium medium;
    if (medium_name == "dram") {
      medium = BenchmarkMedium::DRAM;
    } else if (medium_name == "ssd") {
      if (!_config.ssd.enabled) {
        spdlog::warn("[mem_bench] Skip SSD cases because SSD is disabled");
        continue;
      }
      medium = BenchmarkMedium::SSD;
    } else {
      throw std::runtime_error(fmt::format("Unsupported medium {}", medium_name));
    }

    for (const std::string& access_name : access_values) {
      bool write;
      if (access_name == "read") {
        write = false;
      } else if (access_name == "write") {
        write = true;
      } else {
        throw std::runtime_error(
            fmt::format("Unsupported access type {}", access_name));
      }

      for (const std::string& issue_mode_name : issue_mode_values) {
        BenchmarkIssueMode issue_mode;
        if (issue_mode_name == "back_to_back") {
          issue_mode = BenchmarkIssueMode::BACK_TO_BACK;
        } else if (issue_mode_name == "serialized") {
          issue_mode = BenchmarkIssueMode::SERIALIZED;
        } else {
          throw std::runtime_error(
              fmt::format("Unsupported issue mode {}", issue_mode_name));
        }

        for (uint64_t size_bytes : size_values) {
          for (uint64_t burst_count : burst_values) {
            MemBenchmarkCase bench_case;
            bench_case.medium = medium;
            bench_case.write = write;
            bench_case.access_size_bytes = size_bytes;
            bench_case.burst_count = static_cast<uint32_t>(burst_count);
            bench_case.issue_mode = issue_mode;
            bench_case.address_pattern = address_pattern;
            cases.push_back(bench_case);
          }
        }
      }
    }
  }
  return cases;
}

void MemBenchmarkRunner::run_case(uint64_t case_id,
                                  const MemBenchmarkCase& bench_case,
                                  std::ofstream& summary_csv,
                                  std::ofstream& detail_csv) {
  auto dram = create_dram();
  auto ssd = create_ssd();
  StorageController controller(_config, dram.get(), ssd.get());

  std::vector<MacroStats> macro_stats(bench_case.burst_count);
  std::deque<ScheduledAccess> pending_issues;
  if (bench_case.issue_mode == BenchmarkIssueMode::BACK_TO_BACK) {
    for (uint32_t macro_id = 0; macro_id < bench_case.burst_count; macro_id++) {
      seed_macro_requests(bench_case, macro_id, 0, pending_issues, macro_stats);
    }
  } else {
    seed_macro_requests(bench_case, 0, 0, pending_issues, macro_stats);
  }

  uint32_t next_serialized_macro = 1;
  uint64_t completed_macros = 0;
  uint64_t subrequests_per_macro =
      round_up(bench_case.access_size_bytes, _config.dram_req_size) /
      _config.dram_req_size;
  uint64_t total_subrequests =
      subrequests_per_macro * static_cast<uint64_t>(bench_case.burst_count);

  uint64_t now_ps = 0;
  while (!pending_issues.empty() || controller.has_pending() ||
         controller.has_ready_response()) {
    controller.advance_to(now_ps);

    while (controller.has_ready_response()) {
      MemoryAccess* response = controller.top_ready_response();
      response->return_time_ps = now_ps;
      MacroStats& macro = macro_stats.at(response->macro_request_id);
      macro.first_return_ps = std::min(macro.first_return_ps, now_ps);
      macro.last_return_ps = std::max(macro.last_return_ps, now_ps);
      macro.completed_subrequests++;
      if (macro.completed_subrequests == macro.total_subrequests) {
        completed_macros++;
        if (bench_case.issue_mode == BenchmarkIssueMode::SERIALIZED &&
            next_serialized_macro < bench_case.burst_count &&
            completed_macros == next_serialized_macro) {
          seed_macro_requests(bench_case, next_serialized_macro, now_ps,
                              pending_issues, macro_stats);
          next_serialized_macro++;
        }
      }

      detail_csv << case_id << ','
                 << medium_to_string(bench_case.medium) << ','
                 << rw_to_string(bench_case.write) << ','
                 << bench_case.access_size_bytes << ','
                 << bench_case.burst_count << ','
                 << issue_mode_to_string(bench_case.issue_mode) << ','
                 << bench_case.address_pattern << ','
                 << response->macro_request_id << ','
                 << response->buffer_id << ','
                 << response->dram_address << ','
                 << response->size << ','
                 << response->issue_time_ps << ','
                 << response->mem_enter_time_ps << ','
                 << response->mem_finish_time_ps << ','
                 << response->return_time_ps << ','
                 << (response->mem_finish_time_ps - response->mem_enter_time_ps) / 1000.0
                 << ','
                 << (response->return_time_ps - response->issue_time_ps) / 1000.0
                 << '\n';

      controller.pop_ready_response();
      delete response;
    }

    while (!pending_issues.empty() && pending_issues.front().issue_time_ps <= now_ps) {
      ScheduledAccess scheduled = pending_issues.front();
      if (!controller.dispatch_request(0, scheduled.request, now_ps)) break;
      pending_issues.pop_front();
    }

    if (pending_issues.empty() && !controller.has_pending() &&
        !controller.has_ready_response()) {
      break;
    }

    uint64_t next_issue_ps = std::numeric_limits<uint64_t>::max();
    if (!pending_issues.empty()) next_issue_ps = pending_issues.front().issue_time_ps;
    uint64_t next_event_ps = controller.next_event_time_ps();
    uint64_t next_ps = std::min(next_issue_ps, next_event_ps);
    if (next_ps == std::numeric_limits<uint64_t>::max()) break;
    if (next_ps <= now_ps) next_ps = now_ps + 1;
    now_ps = next_ps;
  }

  std::vector<double> macro_latencies_ns;
  macro_latencies_ns.reserve(macro_stats.size());
  uint64_t final_return_ps = 0;
  for (const auto& macro : macro_stats) {
    final_return_ps = std::max(final_return_ps, macro.last_return_ps);
    if (macro.total_subrequests > 0 &&
        macro.completed_subrequests == macro.total_subrequests &&
        macro.last_return_ps >= macro.issue_time_ps) {
      macro_latencies_ns.push_back(
          static_cast<double>(macro.last_return_ps - macro.issue_time_ps) / 1000.0);
    }
  }

  double avg_latency_ns = macro_latencies_ns.empty()
                              ? 0.0
                              : std::accumulate(macro_latencies_ns.begin(),
                                                macro_latencies_ns.end(), 0.0) /
                                    static_cast<double>(macro_latencies_ns.size());
  double min_latency_ns = macro_latencies_ns.empty()
                              ? 0.0
                              : *std::min_element(macro_latencies_ns.begin(),
                                                  macro_latencies_ns.end());
  double max_latency_ns = macro_latencies_ns.empty()
                              ? 0.0
                              : *std::max_element(macro_latencies_ns.begin(),
                                                  macro_latencies_ns.end());
  double p50_latency_ns = percentile(macro_latencies_ns, 0.50);
  double p95_latency_ns = percentile(macro_latencies_ns, 0.95);
  double p99_latency_ns = percentile(macro_latencies_ns, 0.99);
  double total_time_ns = static_cast<double>(final_return_ps) / 1000.0;
  double total_bytes = static_cast<double>(bench_case.access_size_bytes) *
                       static_cast<double>(bench_case.burst_count);
  double bandwidth_gbps =
      total_time_ns > 0.0 ? total_bytes / total_time_ns : 0.0;

  summary_csv << case_id << ','
              << medium_to_string(bench_case.medium) << ','
              << rw_to_string(bench_case.write) << ','
              << bench_case.access_size_bytes << ','
              << bench_case.burst_count << ','
              << issue_mode_to_string(bench_case.issue_mode) << ','
              << bench_case.address_pattern << ','
              << total_subrequests << ','
              << min_latency_ns << ','
              << avg_latency_ns << ','
              << p50_latency_ns << ','
              << p95_latency_ns << ','
              << p99_latency_ns << ','
              << max_latency_ns << ','
              << total_time_ns << ','
              << bandwidth_gbps << '\n';

  spdlog::info(
      "[mem_bench] case={} medium={} rw={} size={}B burst={} mode={} avg={:.3f}ns max={:.3f}ns",
      case_id, medium_to_string(bench_case.medium), rw_to_string(bench_case.write),
      bench_case.access_size_bytes, bench_case.burst_count,
      issue_mode_to_string(bench_case.issue_mode), avg_latency_ns, max_latency_ns);
}

void MemBenchmarkRunner::seed_macro_requests(
    const MemBenchmarkCase& bench_case, uint32_t macro_request_id,
    uint64_t issue_time_ps, std::deque<ScheduledAccess>& pending_issues,
    std::vector<MacroStats>& macro_stats) const {
  MacroStats& macro = macro_stats.at(macro_request_id);
  macro.issue_time_ps = issue_time_ps;
  uint64_t subrequest_count =
      round_up(bench_case.access_size_bytes, _config.dram_req_size) /
      _config.dram_req_size;
  macro.total_subrequests = subrequest_count;
  for (uint32_t subrequest_id = 0; subrequest_id < subrequest_count; subrequest_id++) {
    auto* access = new MemoryAccess();
    access->id = generate_mem_access_id();
    access->dram_address = next_address(bench_case.medium, macro_request_id,
                                        subrequest_id, bench_case.access_size_bytes,
                                        bench_case.address_pattern);
    access->spad_address = 0;
    access->size = _config.dram_req_size;
    access->write = bench_case.write;
    access->request = true;
    access->core_id = 0;
    access->buffer_id = subrequest_id;
    access->issue_time_ps = issue_time_ps;
    access->logical_size_bytes = bench_case.access_size_bytes;
    access->macro_request_id = macro_request_id;
    access->source_medium = MemoryMedium::UNKNOWN;
    access->target_medium = bench_case.medium == BenchmarkMedium::SSD
                                ? MemoryMedium::SSD
                                : MemoryMedium::DRAM;
    pending_issues.push_back({issue_time_ps, access});
  }
}

void MemBenchmarkRunner::write_summary_header(std::ofstream& summary_csv) const {
  summary_csv
      << "case_id,medium,rw,access_size_bytes,burst_count,issue_mode,"
      << "address_pattern,total_subrequests,macro_min_latency_ns,"
      << "macro_avg_latency_ns,macro_p50_latency_ns,macro_p95_latency_ns,"
      << "macro_p99_latency_ns,macro_max_latency_ns,total_time_ns,"
      << "bandwidth_GBps\n";
}

void MemBenchmarkRunner::write_detail_header(std::ofstream& detail_csv) const {
  detail_csv
      << "case_id,medium,rw,access_size_bytes,burst_count,issue_mode,"
      << "address_pattern,macro_request_id,subrequest_id,address,size_bytes,"
      << "issue_time_ps,mem_enter_time_ps,mem_finish_time_ps,return_time_ps,"
      << "device_latency_ns,end_to_end_latency_ns\n";
}

std::unique_ptr<Dram> MemBenchmarkRunner::create_dram() const {
  SimulationConfig config = _config;
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path =
      onnxim_path_env != nullptr ? std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    return std::make_unique<SimpleDram>(config);
  }
  if (config.dram_type == DramType::RAMULATOR1) {
    config.dram_config_path = fs::path(onnxim_path)
                                  .append("configs")
                                  .append(config.dram_config_path)
                                  .string();
    return std::make_unique<DramRamulator>(config);
  }
  if (config.dram_type == DramType::RAMULATOR2) {
    config.dram_config_path = fs::path(onnxim_path)
                                  .append("configs")
                                  .append(config.dram_config_path)
                                  .string();
    return std::make_unique<DramRamulator2>(config);
  }
  throw std::runtime_error("Unsupported dram type for mem_bench");
}

std::unique_ptr<Ssd> MemBenchmarkRunner::create_ssd() const {
  if (!_config.ssd.enabled) return nullptr;
  SsdConfig ssd_config;
  ssd_config.address_base = _config.ssd.address_base;
  ssd_config.capacity_bytes = _config.ssd.capacity_bytes;
  ssd_config.secsz = _config.ssd.secsz;
  ssd_config.secs_per_pg = _config.ssd.secs_per_pg;
  ssd_config.pgs_per_blk = _config.ssd.pgs_per_blk;
  ssd_config.blks_per_pl = _config.ssd.blks_per_pl;
  ssd_config.pls_per_lun = _config.ssd.pls_per_lun;
  ssd_config.luns_per_ch = _config.ssd.luns_per_ch;
  ssd_config.nchs = _config.ssd.nchs;
  ssd_config.pg_rd_lat = _config.ssd.pg_rd_lat;
  ssd_config.pg_wr_lat = _config.ssd.pg_wr_lat;
  ssd_config.blk_er_lat = _config.ssd.blk_er_lat;
  ssd_config.ch_xfer_lat = _config.ssd.ch_xfer_lat;
  return std::make_unique<Ssd>(ssd_config, _config.dram_freq);
}

uint64_t MemBenchmarkRunner::round_up(uint64_t value, uint64_t alignment) const {
  if (alignment == 0) return value;
  uint64_t remainder = value % alignment;
  return remainder == 0 ? value : (value + alignment - remainder);
}

uint64_t MemBenchmarkRunner::next_address(BenchmarkMedium medium,
                                          uint32_t macro_request_id,
                                          uint32_t subrequest_id,
                                          uint64_t access_size_bytes,
                                          const std::string& address_pattern) const {
  uint64_t base_addr = medium == BenchmarkMedium::SSD ? _config.ssd.address_base : 0;
  uint64_t macro_stride = round_up(access_size_bytes, _config.dram_req_size);
  if (address_pattern != "contiguous") {
    throw std::runtime_error(
        fmt::format("Unsupported address pattern {}", address_pattern));
  }
  return base_addr + macro_request_id * macro_stride +
         subrequest_id * _config.dram_req_size;
}

std::string MemBenchmarkRunner::medium_to_string(BenchmarkMedium medium) const {
  return medium == BenchmarkMedium::SSD ? "ssd" : "dram";
}

std::string MemBenchmarkRunner::issue_mode_to_string(
    BenchmarkIssueMode mode) const {
  return mode == BenchmarkIssueMode::SERIALIZED ? "serialized"
                                                : "back_to_back";
}

std::string MemBenchmarkRunner::rw_to_string(bool write) const {
  return write ? "write" : "read";
}
