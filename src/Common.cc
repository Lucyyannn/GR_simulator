#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}
uint32_t generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

addr_type allocate_address(uint32_t size) {
  static addr_type base_addr{0};
  addr_type result = base_addr;
  int offset = 0;
  if (result % 256 != 0) {
    offset = 256 - (result % 256);
  }
  result += offset;
  assert(result % 256 == 0);
  base_addr += (size + offset);
  base_addr += (256 - base_addr % 256);
  return result;
}

static TensorPlacementPolicy g_tensor_placement_policy =
    TensorPlacementPolicy::SIZE_THRESHOLD;
static uint64_t g_ssd_threshold_bytes = 0;
static uint64_t g_hbm_base_addr = 0;
static uint64_t g_hbm_capacity_bytes = 0;
static uint64_t g_ddr_base_addr = 0x400000000ULL;
static uint64_t g_ddr_capacity_bytes = 0;
static uint64_t g_ssd_base_addr = 0x800000000ULL;
static uint64_t g_ssd_capacity_bytes = (1ULL << 40);
static addr_type g_hbm_cursor = 0;
static addr_type g_ddr_cursor = 0;
static addr_type g_ssd_cursor = 0;

namespace {

uint64_t parse_u64_entry(const json& value, uint64_t default_value = 0) {
  if (value.is_null()) return default_value;
  if (value.is_string()) {
    return std::stoull(value.get<std::string>(), nullptr, 0);
  }
  if (value.is_number_unsigned()) return value.get<uint64_t>();
  if (value.is_number_integer()) return static_cast<uint64_t>(value.get<int64_t>());
  return default_value;
}

uint64_t read_u64(const json& config, const std::string& key, uint64_t default_value) {
  if (!config.contains(key)) return default_value;
  return parse_u64_entry(config.at(key), default_value);
}

addr_type allocate_from_region(uint32_t size, uint64_t base_addr, addr_type& cursor) {
  addr_type result = base_addr + cursor;
  int offset = 0;
  if (result % 256 != 0) offset = 256 - (result % 256);
  result += offset;
  cursor += size + offset;
  cursor += (256 - cursor % 256);
  return result;
}

TensorPlacementPolicy parse_placement_policy(const std::string& policy_name) {
  if (policy_name == "hbm") return TensorPlacementPolicy::HBM;
  if (policy_name == "ddr") return TensorPlacementPolicy::DDR;
  if (policy_name == "ssd") return TensorPlacementPolicy::SSD;
  if (policy_name == "size_threshold") return TensorPlacementPolicy::SIZE_THRESHOLD;
  throw std::runtime_error(fmt::format("Unsupported placement policy {}", policy_name));
}

void populate_tier_config(TieredMemoryConfig& tier, const json& source,
                          uint64_t default_base, uint64_t default_capacity,
                          const TieredMemoryConfig* fallback = nullptr) {
  if (fallback != nullptr) tier = *fallback;
  tier.enabled = source.value("enabled", true);
  if (source.contains("type")) {
    const std::string type_name = source["type"].get<std::string>();
    if (type_name == "simple")
      tier.type = DramType::SIMPLE;
    else if (type_name == "ramulator")
      tier.type = DramType::RAMULATOR1;
    else if (type_name == "ramulator2")
      tier.type = DramType::RAMULATOR2;
    else
      throw std::runtime_error(fmt::format("Not implemented dram type {} ", type_name));
  }
  tier.freq = source.value("freq", tier.freq);
  tier.channels = source.value("channels", tier.channels);
  tier.req_size = source.value("req_size", tier.req_size);
  tier.latency = source.value("latency", tier.latency);
  tier.size_gb = source.value("size_gb", tier.size_gb);
  tier.nbl = source.value("nbl", tier.nbl);
  tier.print_interval = source.value("print_interval", tier.print_interval);
  if (source.contains("config_path"))
    tier.config_path = source["config_path"].get<std::string>();
  tier.address_base = read_u64(source, "address_base", default_base);
  tier.capacity_bytes = read_u64(source, "capacity_bytes", default_capacity);
  if (tier.capacity_bytes == 0 && tier.size_gb > 0)
    tier.capacity_bytes = static_cast<uint64_t>(tier.size_gb) << 30;
}

}  // namespace

void configure_tensor_placement_policy(const SimulationConfig& config) {
  g_tensor_placement_policy = config.placement.policy;
  g_ssd_threshold_bytes = config.placement.ssd_threshold_bytes;
  g_hbm_base_addr = config.hbm.address_base;
  g_hbm_capacity_bytes = config.hbm.capacity_bytes;
  g_ddr_base_addr = config.ddr.address_base;
  g_ddr_capacity_bytes = config.ddr.capacity_bytes;
  g_ssd_base_addr = config.ssd.address_base;
  g_ssd_capacity_bytes = config.ssd.capacity_bytes;
  g_hbm_cursor = 0;
  g_ddr_cursor = 0;
  g_ssd_cursor = 0;
}

MemoryMedium default_tensor_medium(uint32_t size) {
  switch (g_tensor_placement_policy) {
    case TensorPlacementPolicy::HBM:
      return MemoryMedium::HBM;
    case TensorPlacementPolicy::DDR:
      return MemoryMedium::DDR;
    case TensorPlacementPolicy::SSD:
      return MemoryMedium::SSD;
    case TensorPlacementPolicy::SIZE_THRESHOLD:
    default:
      return (g_ssd_threshold_bytes > 0 && size >= g_ssd_threshold_bytes)
                 ? MemoryMedium::SSD
                 : MemoryMedium::HBM;
  }
}

uint64_t get_medium_base(MemoryMedium medium) {
  switch (medium) {
    case MemoryMedium::HBM:
      return g_hbm_base_addr;
    case MemoryMedium::DDR:
      return g_ddr_base_addr;
    case MemoryMedium::SSD:
      return g_ssd_base_addr;
    case MemoryMedium::UNKNOWN:
    default:
      return 0;
  }
}

uint64_t get_medium_capacity(MemoryMedium medium) {
  switch (medium) {
    case MemoryMedium::HBM:
      return g_hbm_capacity_bytes;
    case MemoryMedium::DDR:
      return g_ddr_capacity_bytes;
    case MemoryMedium::SSD:
      return g_ssd_capacity_bytes;
    case MemoryMedium::UNKNOWN:
    default:
      return 0;
  }
}

addr_type allocate_address_in_medium(uint32_t size, MemoryMedium medium) {
  switch (medium) {
    case MemoryMedium::HBM:
      return allocate_from_region(size, g_hbm_base_addr, g_hbm_cursor);
    case MemoryMedium::DDR:
      return allocate_from_region(size, g_ddr_base_addr, g_ddr_cursor);
    case MemoryMedium::SSD:
      return allocate_from_region(size, g_ssd_base_addr, g_ssd_cursor);
    case MemoryMedium::UNKNOWN:
    default:
      return allocate_address(size);
  }
}

void set_ssd_placement_policy(uint64_t threshold_bytes, uint64_t ssd_base, uint64_t capacity_bytes) {
  g_tensor_placement_policy = TensorPlacementPolicy::SIZE_THRESHOLD;
  g_ssd_threshold_bytes = threshold_bytes;
  g_ssd_base_addr = ssd_base;
  g_ssd_capacity_bytes = capacity_bytes;
  g_ssd_cursor = 0;
}

bool should_place_in_ssd(uint32_t size) {
  return default_tensor_medium(size) == MemoryMedium::SSD;
}

uint64_t get_ssd_base() { return g_ssd_base_addr; }

uint64_t get_ssd_capacity() { return g_ssd_capacity_bytes; }

addr_type allocate_address_placed(uint32_t size, bool place_in_ssd,
                                  uint64_t ssd_base) {
  if (!place_in_ssd) return allocate_address_in_medium(size, MemoryMedium::HBM);
  if (ssd_base != g_ssd_base_addr) {
    g_ssd_base_addr = ssd_base;
  }
  return allocate_address_in_medium(size, MemoryMedium::SSD);
}

template <typename T>
T get_config_value(json config, std::string key) {
  if (config.contains(key)) {
    return config[key];
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

const static std::map<std::string, CoreType> core_type_map = {
  {"systolic_os", CoreType::SYSTOLIC_OS},
  {"systolic_ws", CoreType::SYSTOLIC_WS}
};

const static std::map<std::string, DramType> dram_type_map = {
  {"simple", DramType::SIMPLE},
  {"ramulator", DramType::RAMULATOR1},
  {"ramulator2", DramType::RAMULATOR2}
};

const static std::map<std::string, IcntType> icnt_type_map = {
  {"simple", IcntType::SIMPLE},
  {"booksim2", IcntType::BOOKSIM2}
};

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  parsed_config.num_cores = get_config_value<uint32_t>(config, "num_cores");
  parsed_config.core_config = new struct CoreConfig[parsed_config.num_cores];
  parsed_config.core_freq = get_config_value<uint32_t>(config, "core_freq");
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_print_interval");

  for (int i=0; i<parsed_config.num_cores; i++) {
    std::string core_id = "core_" + std::to_string(i);
    auto core_config = config["core_config"][core_id];
    std::string core_type = core_config["core_type"];
    if (core_type_map.contains(core_type)) {
      parsed_config.core_config[i].core_type = core_type_map.at(core_type);
    } else {
      throw std::runtime_error(fmt::format("Not implemented core type {} ", core_type));
    }
    parsed_config.core_config[i].core_width = core_config["core_width"];
    parsed_config.core_config[i].core_height = core_config["core_height"];

    /* Vector configs */
    parsed_config.core_config[i].vector_process_bit = core_config["vector_process_bit"];
    parsed_config.core_config[i].add_latency = core_config["add_latency"];
    parsed_config.core_config[i].mul_latency = core_config["mul_latency"];
    parsed_config.core_config[i].exp_latency = core_config["exp_latency"];
    parsed_config.core_config[i].gelu_latency = core_config["gelu_latency"];
    parsed_config.core_config[i].add_tree_latency = core_config["add_tree_latency"];
    parsed_config.core_config[i].scalar_sqrt_latency = core_config["scalar_sqrt_latency"];
    parsed_config.core_config[i].scalar_add_latency = core_config["scalar_add_latency"];
    parsed_config.core_config[i].scalar_mul_latency = core_config["scalar_mul_latency"];
    parsed_config.core_config[i].mac_latency = core_config["mac_latency"];
    parsed_config.core_config[i].div_latency = core_config["div_latency"];

    /* SRAM configs */
    parsed_config.core_config[i].sram_width = core_config["sram_width"];
    parsed_config.core_config[i].spad_size = core_config["spad_size"];
    parsed_config.core_config[i].accum_spad_size = core_config["accum_spad_size"];
  }

  /* Legacy DRAM config -> HBM compatibility alias */
  std::string dram_type = get_config_value<std::string>(config, "dram_type");
  if (dram_type_map.contains(dram_type)) {
    parsed_config.dram_type = dram_type_map.at(dram_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented dram type {} ", dram_type));
  }

  parsed_config.dram_freq = get_config_value<uint32_t>(config, "dram_freq");
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  if (config.contains("dram_config_path"))
    parsed_config.dram_config_path = config["dram_config_path"];
  parsed_config.dram_channels = config["dram_channels"];
  if (config.contains("dram_req_size"))
    parsed_config.dram_req_size = config["dram_req_size"];
  if (config.contains("dram_print_interval"))
    parsed_config.dram_print_interval = config["dram_print_interval"];
  if(config.contains("dram_nbl"))
    parsed_config.dram_nbl = config["dram_nbl"];
  if (config.contains("dram_size"))
    parsed_config.dram_size = config["dram_size"];
  else
    parsed_config.dram_size = 0;

  parsed_config.hbm.enabled = true;
  parsed_config.hbm.type = parsed_config.dram_type;
  parsed_config.hbm.freq = parsed_config.dram_freq;
  parsed_config.hbm.channels = parsed_config.dram_channels;
  parsed_config.hbm.req_size = parsed_config.dram_req_size;
  parsed_config.hbm.latency = parsed_config.dram_latency;
  parsed_config.hbm.size_gb = parsed_config.dram_size;
  parsed_config.hbm.nbl = parsed_config.dram_nbl;
  parsed_config.hbm.print_interval = parsed_config.dram_print_interval;
  parsed_config.hbm.config_path = parsed_config.dram_config_path;
  parsed_config.hbm.address_base = read_u64(config, "hbm_address_base", 0);
  parsed_config.hbm.capacity_bytes =
      parsed_config.hbm.size_gb > 0 ? (static_cast<uint64_t>(parsed_config.hbm.size_gb) << 30)
                                    : 0;

  parsed_config.ddr.enabled = false;
  parsed_config.ddr.type = DramType::RAMULATOR2;
  parsed_config.ddr.address_base = parsed_config.hbm.capacity_bytes;
  parsed_config.ddr.size_gb = 16;
  parsed_config.ddr.capacity_bytes =
      static_cast<uint64_t>(parsed_config.ddr.size_gb) << 30;
  parsed_config.ddr.config_path = "../configs/ramulator2_configs/DDR4.yaml";
  parsed_config.ddr.freq = parsed_config.dram_freq;
  parsed_config.ddr.channels = parsed_config.dram_channels;
  parsed_config.ddr.req_size = parsed_config.dram_req_size;
  parsed_config.ddr.latency = parsed_config.dram_latency;
  parsed_config.ddr.nbl = parsed_config.dram_nbl;
  parsed_config.ddr.print_interval = parsed_config.dram_print_interval;

  if (config.contains("hbm")) {
    populate_tier_config(parsed_config.hbm, config["hbm"], 0,
                         parsed_config.hbm.capacity_bytes, &parsed_config.hbm);
  }
  if (config.contains("ddr")) {
    populate_tier_config(parsed_config.ddr, config["ddr"],
                         parsed_config.hbm.address_base + parsed_config.hbm.capacity_bytes,
                         parsed_config.ddr.capacity_bytes, &parsed_config.ddr);
  }

  /* Icnt config */
  std::string icnt_type = get_config_value<std::string>(config, "icnt_type");
  if (icnt_type_map.contains(icnt_type)) {
    parsed_config.icnt_type = icnt_type_map.at(icnt_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ", icnt_type));
  }

  parsed_config.icnt_freq = get_config_value<uint32_t>(config, "icnt_freq");
  if (config.contains("icnt_latency"))
    parsed_config.icnt_latency = config["icnt_latency"];
  if (config.contains("icnt_config_path"))
    parsed_config.icnt_config_path = config["icnt_config_path"];
  if (config.contains("icnt_print_interval"))
    parsed_config.icnt_print_interval = config["icnt_print_interval"];
  if (config.contains("icnt_injection_ports_per_core"))
    parsed_config.icnt_injection_ports_per_core = config["icnt_injection_ports_per_core"];

  /* SSD config (optional) */
  if (config.contains("ssd")) {
    auto& s = config["ssd"];
    parsed_config.ssd.enabled = s.value("enabled", true);
    parsed_config.ssd.address_base = read_u64(
        s, "address_base",
        parsed_config.ddr.address_base + parsed_config.ddr.capacity_bytes);
    parsed_config.ssd.capacity_bytes =
        read_u64(s, "capacity_bytes", parsed_config.ssd.capacity_bytes);
    parsed_config.ssd.secsz        = s.value("secsz",        parsed_config.ssd.secsz);
    parsed_config.ssd.secs_per_pg  = s.value("secs_per_pg",  parsed_config.ssd.secs_per_pg);
    parsed_config.ssd.pgs_per_blk  = s.value("pgs_per_blk",  parsed_config.ssd.pgs_per_blk);
    parsed_config.ssd.blks_per_pl  = s.value("blks_per_pl",  parsed_config.ssd.blks_per_pl);
    parsed_config.ssd.pls_per_lun  = s.value("pls_per_lun",  parsed_config.ssd.pls_per_lun);
    parsed_config.ssd.luns_per_ch  = s.value("luns_per_ch",  parsed_config.ssd.luns_per_ch);
    parsed_config.ssd.nchs         = s.value("nchs",         parsed_config.ssd.nchs);
    parsed_config.ssd.pg_rd_lat    = s.value("pg_rd_lat",    parsed_config.ssd.pg_rd_lat);
    parsed_config.ssd.pg_wr_lat    = s.value("pg_wr_lat",    parsed_config.ssd.pg_wr_lat);
    parsed_config.ssd.blk_er_lat   = s.value("blk_er_lat",   parsed_config.ssd.blk_er_lat);
    parsed_config.ssd.ch_xfer_lat  = s.value("ch_xfer_lat",  parsed_config.ssd.ch_xfer_lat);
    spdlog::info("[CONFIG] SSD enabled, base=0x{:x}, cap={}GB, chs={}, luns/ch={}",
                 parsed_config.ssd.address_base,
                 parsed_config.ssd.capacity_bytes / (1ULL<<30),
                 parsed_config.ssd.nchs, parsed_config.ssd.luns_per_ch);
  } else {
    parsed_config.ssd.enabled = false;
  }

  if (config.contains("placement")) {
    const auto& placement = config["placement"];
    parsed_config.placement.policy =
        parse_placement_policy(placement.value("policy", std::string("size_threshold")));
    parsed_config.placement.ssd_threshold_bytes =
        read_u64(placement, "ssd_threshold_bytes", 0);
  } else {
    parsed_config.placement.policy = TensorPlacementPolicy::SIZE_THRESHOLD;
    if (config.contains("ssd") && config["ssd"].contains("place_threshold_bytes"))
      parsed_config.placement.ssd_threshold_bytes =
          parse_u64_entry(config["ssd"]["place_threshold_bytes"], 0);
  }

  /* Refresh legacy aliases so existing code paths continue to see HBM values. */
  parsed_config.dram_type = parsed_config.hbm.type;
  parsed_config.dram_freq = parsed_config.hbm.freq;
  parsed_config.dram_channels = parsed_config.hbm.channels;
  parsed_config.dram_req_size = parsed_config.hbm.req_size;
  parsed_config.dram_latency = parsed_config.hbm.latency;
  parsed_config.dram_size = parsed_config.hbm.size_gb;
  parsed_config.dram_nbl = parsed_config.hbm.nbl;
  parsed_config.dram_print_interval = parsed_config.hbm.print_interval;
  parsed_config.dram_config_path = parsed_config.hbm.config_path;

  parsed_config.scheduler_type = get_config_value<std::string>(config, "scheduler");
  parsed_config.precision = get_config_value<uint32_t>(config, "precision");
  parsed_config.layout = get_config_value<std::string>(config, "layout");
  parsed_config.enable_fast_forward = config.value("enable_fast_forward", false);

  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[partition_id].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[0].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}

uint32_t ceil_div(uint32_t src, uint32_t div) { return (src+div-1)/div; }

std::vector<uint32_t> parse_dims(const std::string &str) {
  std::vector<uint32_t> dims;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, ',')) {
      dims.push_back(std::stoi(token));
  }
  return dims;
}

std::string dims_to_string(const std::vector<uint32_t> &dims){
  std::string str;
  for (int i=0; i<dims.size(); i++) {
    str += std::to_string(dims[i]);
    if (i != dims.size()-1) {
      str += ",";
    }
  }
  return str;
}
