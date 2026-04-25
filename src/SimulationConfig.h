#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { SIMPLE, RAMULATOR1, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class TensorPlacementPolicy {
  HBM = 0,
  DDR = 1,
  SSD = 2,
  SIZE_THRESHOLD = 3,
};

struct CoreConfig {
  CoreType core_type;
  uint32_t core_width;
  uint32_t core_height;

  /* Vector config*/
  uint32_t vector_process_bit;
  uint32_t layernorm_latency = 1;
  uint32_t softmax_latency = 1;
  uint32_t add_latency = 1;
  uint32_t mul_latency = 1;
  uint32_t mac_latency = 1;
  uint32_t div_latency = 1;
  uint32_t exp_latency = 1;
  uint32_t gelu_latency = 1;
  uint32_t add_tree_latency = 1;
  uint32_t scalar_sqrt_latency = 1;
  uint32_t scalar_add_latency = 1;
  uint32_t scalar_mul_latency = 1;

  /* SRAM config */
  uint32_t sram_width;
  uint32_t spad_size;
  uint32_t accum_spad_size;
};

struct SsdSimConfig {
  bool enabled = false;
  uint64_t address_base = 0x800000000ULL;   // 32GB
  uint64_t capacity_bytes = (1ULL << 40);   // 1TB
  int secsz = 512;
  int secs_per_pg = 8;
  int pgs_per_blk = 256;
  int blks_per_pl = 256;
  int pls_per_lun = 1;
  int luns_per_ch = 8;
  int nchs = 8;
  int pg_rd_lat  = 40000;
  int pg_wr_lat  = 200000;
  int blk_er_lat = 2000000;
  int ch_xfer_lat = 0;
};

struct TieredMemoryConfig {
  bool enabled = false;
  DramType type = DramType::RAMULATOR2;
  uint32_t freq = 0;
  uint32_t channels = 0;
  uint32_t req_size = 32;
  uint32_t latency = 0;
  uint32_t size_gb = 0;
  uint32_t nbl = 1;
  uint32_t print_interval = 0;
  std::string config_path;
  uint64_t address_base = 0;
  uint64_t capacity_bytes = 0;
};

struct PlacementConfig {
  TensorPlacementPolicy policy = TensorPlacementPolicy::SIZE_THRESHOLD;
  uint64_t ssd_threshold_bytes = 0;
};

struct SimulationConfig {
  /* Core config */
  uint32_t num_cores;
  uint32_t core_freq;
  uint32_t core_print_interval;
  struct CoreConfig *core_config;

  /* Tensor placement policy */
  PlacementConfig placement;

  /* HBM / DDR / SSD hierarchy */
  TieredMemoryConfig hbm;
  TieredMemoryConfig ddr;

  /* SSD config (FEMU-inspired black-box SSD) */
  SsdSimConfig ssd;

  /* Legacy HBM aliases preserved for compatibility with existing code paths. */
  DramType dram_type;
  uint32_t dram_freq;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  uint32_t dram_size; // in GB
  uint32_t dram_nbl = 1; // busrt length in clock cycles (bust_length 8 in DDR -> 4 nbl)
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq;
  uint32_t icnt_latency;
  uint32_t icnt_print_interval=0;

  /* Sheduler config */
  std::string scheduler_type;

  /* Other configs */
  uint32_t precision;
  uint32_t full_precision = 4;
  std::string layout;
  bool enable_fast_forward = false;

  /*
   * This map stores the partition information: <partition_id, core_id>
   *
   * Note: Each core belongs to one partition. Through these partition IDs,
   * it is possible to assign a specific DNN model to a particular group of cores.
   */
  std::map<uint32_t, std::vector<uint32_t>> partiton_map;

  uint64_t req_size_for(TensorPlacementPolicy policy) const {
    if (policy == TensorPlacementPolicy::DDR && ddr.req_size > 0)
      return ddr.req_size;
    return hbm.req_size > 0 ? hbm.req_size : dram_req_size;
  }

  uint64_t align_address(uint64_t addr) {
    const uint64_t req_size = hbm.req_size > 0 ? hbm.req_size : dram_req_size;
    return req_size == 0 ? addr : (addr - (addr % req_size));
  }

  uint64_t align_address(uint64_t addr, uint64_t req_size) const {
    return req_size == 0 ? addr : (addr - (addr % req_size));
  }

  float max_systolic_flops(uint32_t id) {
    return core_config[id].core_width * core_config[id].core_height * core_freq * 2 * num_cores / 1000; // GFLOPS
  }

  float max_vector_flops(uint32_t id) {
    return (core_config[id].vector_process_bit >> 3) / precision * 2 * core_freq / 1000; // GFLOPS
  }

  float max_hbm_bandwidth() const {
    return hbm.freq * hbm.channels * hbm.req_size / std::max(hbm.nbl, 1u) / 1000.0f;
  }

  float max_ddr_bandwidth() const {
    return ddr.freq * ddr.channels * ddr.req_size / std::max(ddr.nbl, 1u) / 1000.0f;
  }

  float max_dram_bandwidth() {
    return max_hbm_bandwidth();
  }

};
