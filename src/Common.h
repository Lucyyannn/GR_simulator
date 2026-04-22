#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Stat.h"
#include "helper/HelperFunctions.h"
#include "nlohmann/json.hpp"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBEGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096

using json = nlohmann::json;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

enum class MemoryMedium {
  UNKNOWN = 0,
  DRAM = 1,
  SSD = 2,
};

typedef struct {
  uint32_t id;
  addr_type dram_address;
  addr_type spad_address;
  uint64_t size;
  bool write;
  bool request;
  uint32_t core_id;
  cycle_type start_cycle;
  cycle_type dram_enter_cycle;
  cycle_type dram_finish_cycle;
  int buffer_id;
  addr_type aux_address = 0;
  uint64_t issue_time_ps = 0;
  uint64_t mem_enter_time_ps = 0;
  uint64_t mem_finish_time_ps = 0;
  uint64_t return_time_ps = 0;
  uint64_t logical_size_bytes = 0;
  uint64_t macro_request_id = 0;
  MemoryMedium source_medium = MemoryMedium::UNKNOWN;
  MemoryMedium target_medium = MemoryMedium::UNKNOWN;
  MemoryMedium destination_medium = MemoryMedium::UNKNOWN;
  bool controller_generated = false;
  bool ssd_host_request = false;
} MemoryAccess;

enum class Opcode {
  MOVIN,
  MOVOUT,
  MOVOUT_POOL,
  GEMM_PRELOAD,
  GEMM,
  GEMM_WRITE,
  COMP,
  IM2COL,
  SOFTMAX,
  LAYERNORM,
  ADD,
  MUL,
  MAC,
  DIV,
  ADDTREE,
  EXP,
  GELU,
  SWISH,
  BAR
};
struct Tile;
typedef struct {
  Opcode opcode;
  cycle_type start_cycle;
  cycle_type finish_cycle;
  std::string id;
  std::vector<std::string> dependent_ids;
  std::string dest_id;
  addr_type dest_addr;
  uint32_t size;          // Used for sram allocation. Multiple of _config.dram_req_size
  uint32_t compute_size;
  std::vector<addr_type> src_addrs;
  int spad_id;
  int accum_spad_id;
  uint32_t operand_id  = 0;
  addr_type base_addr;

  uint32_t tile_m;
  uint32_t tile_k;
  uint32_t tile_n;

  bool src_from_accum = false;
  bool zero_init = false;
  bool last_inst = false;
  Tile* my_tile;
  std::string to_string();
} Instruction;

struct Tile {
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    BAR,
    EMPTY,
  };
  Status status = Status::EMPTY;
  std::string optype;
  uint32_t layer_id;
  uint32_t fused_op_id; /* For fused operation */
  uint32_t batch;
  uint32_t Q;
  uint32_t P;
  uint32_t M;
  uint32_t C;
  uint32_t S;
  uint32_t R;

  TileStat stat;
  std::deque<std::unique_ptr<Instruction>> instructions;
  bool accum;
  bool skip;
  int spad_id;
  int accum_spad_id;
  int core_id = -1;
  bool inst_finished = false;
} ;

uint32_t generate_id();
uint32_t generate_mem_access_id();
addr_type allocate_address(uint32_t size);

/* Placement-aware allocator. When `place_in_ssd` is true, the returned
 * address lies in the SSD region [ssd_base, ssd_base + capacity). Uses an
 * internal monotonically-increasing cursor for each region. */
addr_type allocate_address_placed(uint32_t size, bool place_in_ssd,
                                  uint64_t ssd_base = 0x800000000ULL);

/* Global toggle set by Model loading based on SimulationConfig.ssd.
 * Tensors whose `size >= threshold` are routed to SSD region. */
void set_ssd_placement_policy(uint64_t threshold_bytes, uint64_t ssd_base, uint64_t capacity_bytes = (1ULL << 40));
uint64_t get_ssd_capacity();
bool should_place_in_ssd(uint32_t size);
uint64_t get_ssd_base();
SimulationConfig initialize_config(json config);
template <typename... Args>
std::string name_gen(Args... args) {
    std::vector<std::string> strs = {args...};
    assert(!strs.empty());
    std::string ret = "";
    for (auto &str : strs) {
        ret += str + ".";
    }
    ret.resize(ret.size() - 1);
    return ret;
}
uint32_t ceil_div(uint32_t src, uint32_t div);

std::vector<uint32_t> parse_dims(const std::string &str);

std::string dims_to_string(const std::vector<uint32_t> &dims);
