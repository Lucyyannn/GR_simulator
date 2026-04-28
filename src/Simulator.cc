#include "Simulator.h"

#include <fstream>
#include <filesystem>
#include <limits>
#include <string>

#include "SystolicOS.h"
#include "SystolicWS.h"

namespace fs = std::filesystem;

namespace {

double ps_to_us(uint64_t ps) {
  return static_cast<double>(ps) / 1e6;
}

double percent(uint64_t numerator, uint64_t denominator) {
  if (denominator == 0) return 0.0;
  return 100.0 * static_cast<double>(numerator) /
         static_cast<double>(denominator);
}

uint64_t tier_total_requests(const MemoryTierRuntimeStats& stats) {
  return stats.read_requests + stats.write_requests;
}

uint64_t tier_total_bytes(const MemoryTierRuntimeStats& stats) {
  return stats.read_bytes + stats.write_bytes;
}

double avg_latency_us(const MemoryTierRuntimeStats& stats) {
  uint64_t requests = tier_total_requests(stats);
  if (requests == 0) return 0.0;
  return ps_to_us(stats.total_latency_ps) / static_cast<double>(requests);
}

double effective_bandwidth_gbps(const MemoryTierRuntimeStats& stats,
                                uint64_t global_time_ps) {
  if (global_time_ps == 0) return 0.0;
  double seconds = static_cast<double>(global_time_ps) / 1e12;
  return static_cast<double>(tier_total_bytes(stats)) / seconds / 1e9;
}

}  // namespace

Simulator::Simulator(SimulationConfig config, bool language_mode)
    : _config(config), _core_cycles(0), _language_mode(language_mode) {
  // Create dram object
  spdlog::info("Simulator Configuration:");
  for (int i=0; i<config.num_cores;i++)
    spdlog::info("[Core {}] Systolic Array Throughput: {} GFLOPS, Spad size: {} KB, Accumulator size: {} KB",
      i, config.max_systolic_flops(i), config.core_config[i].spad_size, config.core_config[i].accum_spad_size);
  spdlog::info("HBM Bandwidth {} GB/s", config.max_hbm_bandwidth());
  if (config.ddr.enabled)
    spdlog::info("DDR Bandwidth {} GB/s", config.max_ddr_bandwidth());
  _core_period = 1000000 / (config.core_freq);
  _icnt_period = 1000000 / (config.icnt_freq);
  _mem_period = 1000000 / std::max<uint32_t>(config.hbm.freq, 1);
  _core_time = 0;
  _mem_time = 0;
  _icnt_time = 0;
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    _hbm = std::make_unique<SimpleDram>(config);
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _hbm = std::make_unique<DramRamulator>(config);
  } 
  else if (config.dram_type == DramType::RAMULATOR2) 
  {
    std::string hbm_config_path = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.hbm.config_path)
                                       .string();
    spdlog::info("HBM Ramulator2 config: {}", hbm_config_path);
    config.hbm.config_path = hbm_config_path;
    config.dram_config_path = hbm_config_path;
    _hbm = std::make_unique<Hbm>(config);
    if (config.ddr.enabled) {
      std::string ddr_config_path = fs::path(onnxim_path)
                                        .append("configs")
                                        .append(config.ddr.config_path)
                                        .string();
      spdlog::info("DDR Ramulator2 config: {}", ddr_config_path);
      config.ddr.config_path = ddr_config_path;
      _ddr = std::make_unique<Ddr>(config);
    }
  } 
  else {
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  }

  /* Optional SSD (FEMU bbssd-inspired) */
  if (config.ssd.enabled) {
    SsdConfig scfg;
    scfg.address_base   = config.ssd.address_base;
    scfg.capacity_bytes = config.ssd.capacity_bytes;
    scfg.secsz          = config.ssd.secsz;
    scfg.secs_per_pg    = config.ssd.secs_per_pg;
    scfg.pgs_per_blk    = config.ssd.pgs_per_blk;
    scfg.blks_per_pl    = config.ssd.blks_per_pl;
    scfg.pls_per_lun    = config.ssd.pls_per_lun;
    scfg.luns_per_ch    = config.ssd.luns_per_ch;
    scfg.nchs           = config.ssd.nchs;
    scfg.pg_rd_lat      = config.ssd.pg_rd_lat;
    scfg.pg_wr_lat      = config.ssd.pg_wr_lat;
    scfg.blk_er_lat     = config.ssd.blk_er_lat;
    scfg.ch_xfer_lat    = config.ssd.ch_xfer_lat;
    _ssd = std::make_unique<Ssd>(scfg, config.hbm.freq);
  }
  _storage_controller =
      std::make_unique<StorageController>(config, _hbm.get(), _ddr.get(), _ssd.get());
  _residency_manager = std::make_unique<ResidencyManager>();

  // Create interconnect object
  if (config.icnt_type == IcntType::SIMPLE) {
    _icnt = std::make_unique<SimpleInterconnect>(config);
  } else if (config.icnt_type == IcntType::BOOKSIM2) {
    _icnt = std::make_unique<Booksim2Interconnect>(config);
  } else {
    spdlog::error("[Configuration] {} Invalid interconnect type...!");
    exit(EXIT_FAILURE);
  }
  _icnt_interval = config.icnt_print_interval;

  // Create core objects
  _cores.resize(config.num_cores);
  _n_cores = config.num_cores;
  _n_memories = config.hbm.channels;
  _noc_node_per_core = config.icnt_injection_ports_per_core;
  _memory_req_size = config.hbm.req_size;
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    _cores[core_index] = Core::create(core_index, config);
  }

  //Configure Hardware Scheduler
  _scheduler = Scheduler::create(_config, &_core_cycles, &_core_time, this);
  spdlog::info("Pipeline admission max_preloading_models={}",
               _config.max_preloading_models);
  
  /* Create heap */
  std::make_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::run_simulator() {
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::handle_model() {
  if(_language_mode) {
    _lang_scheduler->cycle();
    if(_lang_scheduler->can_schedule_model()) {
      _models.push_back(_lang_scheduler->pop_model());
      std::push_heap(_models.begin(), _models.end(), CompareModel());
    }
  }
  size_t arrived_count = 0;
  while (!_models.empty() && _models.front()->get_request_time() <= _core_time) {
    std::unique_ptr<Model> launch_model = std::move(_models.front());
    std::pop_heap(_models.begin(), _models.end(), CompareModel());
    _models.pop_back();
    _waiting_to_preload_models.push_back(std::move(launch_model));
    arrived_count++;
  }
  if (arrived_count > 0) {
    spdlog::info("Queued {} model(s) waiting_to_preload at {:.6f} us "
                 "(total_waiting={})",
                 arrived_count, ps_to_us(_core_time),
                 _waiting_to_preload_models.size());
  }

  if (_scheduler && _config.enable_pipeline_preload) {
    _scheduler->refresh_pipeline_preload(_storage_controller.get());
  }

  for (auto it = _preloading_models.begin(); it != _preloading_models.end();) {
    if (_config.enable_pipeline_preload) {
      (*it)->refresh_pipeline_preload(_storage_controller.get());
    }
    if ((*it)->data_movements_ready(_storage_controller.get())) {
      (*it)->complete_data_movements(_storage_controller.get());
      spdlog::info("Model {} data ready at {:.6f} us",
                   (*it)->get_name(), ps_to_us(_core_time));
      _ready_to_compute_models.push_back(std::move(*it));
      it = _preloading_models.erase(it);
    } else {
      ++it;
    }
  }

  admit_preload_models();
  schedule_ready_models();
}

size_t Simulator::pipeline_preload_inflight_count() const {
  size_t count = _preloading_models.size();
  for (const auto& model : _ready_to_compute_models) {
    if (model && model->supports_pipeline_preload() &&
        !model->pipeline_preload_complete()) {
      count++;
    }
  }
  if (_scheduler) count += _scheduler->pipeline_preload_inflight_count();
  return count;
}

void Simulator::admit_preload_models() {
  const uint32_t max_preloading = _config.max_preloading_models;
  while (!_waiting_to_preload_models.empty() &&
         (max_preloading == 0 ||
          pipeline_preload_inflight_count() < max_preloading)) {
    std::unique_ptr<Model> launch_model =
        std::move(_waiting_to_preload_models.front());
    _waiting_to_preload_models.pop_front();

    const json model_config = launch_model->get_model_config();
    std::string weight_key = launch_model->get_name();
    if (model_config.contains("weight_key")) {
      weight_key = model_config["weight_key"].get<std::string>();
    }

    spdlog::info("Admit model {} to preload at {:.6f} us "
                 "(inflight_preloads={}/{}, waiting_remaining={})",
                 launch_model->get_name(), ps_to_us(_core_time),
                 pipeline_preload_inflight_count(), max_preloading,
                 _waiting_to_preload_models.size());
    launch_model->set_residency_manager(_residency_manager.get());
    launch_model->initialize_model(_weight_table[weight_key]);
    launch_model->prefill_ssd_tensors(_ssd.get());
    auto movement_ids =
        launch_model->submit_data_movements(_storage_controller.get(), _core_time);
    if (_config.enable_pipeline_preload &&
        launch_model->supports_pipeline_preload()) {
      spdlog::info("Model {} pipeline preload enabled; schedule before all "
                   "{} movements finish at {:.6f} us",
                   launch_model->get_name(), movement_ids.size(),
                   ps_to_us(_core_time));
      _ready_to_compute_models.push_back(std::move(launch_model));
      continue;
    }
    if (movement_ids.empty() ||
        launch_model->data_movements_ready(_storage_controller.get())) {
      launch_model->complete_data_movements(_storage_controller.get());
      spdlog::info("Model {} data ready at {:.6f} us",
                   launch_model->get_name(), ps_to_us(_core_time));
      _ready_to_compute_models.push_back(std::move(launch_model));
    } else {
      spdlog::info("Model {} preloading {} movements at {:.6f} us",
                   launch_model->get_name(), movement_ids.size(),
                   ps_to_us(_core_time));
      _preloading_models.push_back(std::move(launch_model));
    }
  }
}

void Simulator::schedule_ready_models() {
  while (!_ready_to_compute_models.empty()) {
    std::unique_ptr<Model> ready_model =
        std::move(_ready_to_compute_models.front());
    _ready_to_compute_models.pop_front();
    ready_model->set_request_time(_core_time);
    spdlog::info("Schedule model: {} at {:.6f} us",
                 ready_model->get_name(), ps_to_us(_core_time));
    _scheduler->schedule_model(std::move(ready_model), 1);
  }
}

void Simulator::cycle() {
  OpStat op_stat;
  ModelStat model_stat;
  uint32_t tile_count;
  bool is_accum_tile;
  while (running()) {
    int model_id = 0;

    uint64_t sim_time_ps = set_cycle_mask();
    _last_sim_time_ps = sim_time_ps;
    if (_storage_controller) _storage_controller->advance_to(sim_time_ps);
    // Core Cycle
    if (_cycle_mask & CORE_MASK) {
      /* Handle requested model */
      handle_model();

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        std::unique_ptr<Tile> finished_tile = _cores[core_id]->pop_finished_tile();
        if (finished_tile->status == Tile::Status::FINISH) {
          _scheduler->finish_tile(core_id, finished_tile->layer_id);
        }
        // Issue new tile to core
        if (!_scheduler->empty()) {
          is_accum_tile = _scheduler->is_accum_tile(core_id, 0);
          if (_cores[core_id]->can_issue(is_accum_tile)) {
            std::unique_ptr<Tile> tile = _scheduler->get_tile(core_id);
            if (tile->status == Tile::Status::INITIALIZED) {
              _cores[core_id]->issue(std::move(tile));
              _tile_timestamp.push_back(std::chrono::high_resolution_clock::now());
            }
          }
        }
        _cores[core_id]->cycle();
      }
      _core_cycles++;
    }

    // DRAM cycle
    if (_cycle_mask & DRAM_MASK) {
      _mem_cycles++;
    }
    // Interconnect cycle
    if (_cycle_mask & ICNT_MASK) {
      _icnt_cycle++;

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        for (int noc_id = 0; noc_id < _noc_node_per_core; noc_id++) {
          // PUHS core to ICNT. memory request
          int port_id = core_id * _noc_node_per_core + noc_id;
          if (_cores[core_id]->has_memory_request()) {
            MemoryAccess *front = _cores[core_id]->top_memory_request();
            front->core_id = core_id;
            if (!_icnt->is_full(port_id, front)) {
              _icnt->push(port_id, get_dest_node(front), front);
              _cores[core_id]->pop_memory_request();
              _nr_from_core++;
            }
          }
          // Push response from ICNT. to Core.
          if (!_icnt->is_empty(port_id)) {
            _cores[core_id]->push_memory_response(_icnt->top(port_id));
            _icnt->pop(port_id);
            _nr_to_core++;
          }
        }
      }

      for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
        // ICNT to memory (DRAM or SSD depending on target address)
        int core_offset = _n_cores * _noc_node_per_core;
        if (!_icnt->is_empty(core_offset + mem_id)) {
          MemoryAccess* mreq = _icnt->top(core_offset + mem_id);

          if (_storage_controller &&
              _storage_controller->dispatch_request(mem_id, mreq, sim_time_ps)) {
            _icnt->pop(core_offset + mem_id);
            _nr_to_mem++;
          }
        }
      }
      int max_controller_responses = _n_memories + (_ssd ? 1 : 0);
      for (int response_count = 0;
           response_count < max_controller_responses &&
           _storage_controller && _storage_controller->has_ready_response();
           response_count++) {
        int core_offset = _n_cores * _noc_node_per_core;
        MemoryAccess* response = _storage_controller->top_ready_response();
        response->return_time_ps = sim_time_ps;
        uint32_t source_port = 0;
        if (response->target_medium == MemoryMedium::HBM && _hbm) {
          source_port = _hbm->get_channel_id(response);
        }
        uint32_t dest_node = get_dest_node(response);
        if (!_icnt->is_full(core_offset + source_port, response)) {
          _icnt->push(core_offset + source_port, dest_node, response);
          _storage_controller->pop_ready_response();
          _nr_from_mem++;
        } else {
          break;
        }
      }
      if (_icnt_interval!=0 && _icnt_cycle % _icnt_interval == 0) {
        spdlog::info("[ICNT] Core->ICNT request {}GB/Sec", ((_memory_req_size*_nr_from_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec", ((_memory_req_size*_nr_to_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec", ((_memory_req_size*_nr_to_mem*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec", ((_memory_req_size*_nr_from_mem*(1000/_icnt_period)/_icnt_interval)));
        _nr_from_core=0;
        _nr_to_core=0;
        _nr_to_mem=0;
        _nr_from_mem=0;
      }
      _icnt->cycle();
    }
    try_fast_forward();
  }
  /* Print simulation stats */
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    _cores[core_id]->print_stats();
  }
  _icnt->print_stats();
  if (_hbm) _hbm->print_stat();
  if (_ddr) _ddr->print_stat();
  if (_ssd) _ssd->print_stat();
}

void Simulator::register_model(std::unique_ptr<Model> model) {
  const json model_config = model->get_model_config();
  std::string weight_key = model->get_name();
  if (model_config.contains("weight_key")) {
    weight_key = model_config["weight_key"].get<std::string>();
  }
  if(_weight_table.find(weight_key) == _weight_table.end()) {
    model->initialize_weight(_weight_table[weight_key]);
  } 
  _models.push_back(std::move(model));
  std::push_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::register_language_model(json info, std::unique_ptr<LanguageModel> model) {
  std::string name = info["name"];
  std::string trace_file = info["trace_file"];
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  trace_file = fs::path(onnxim_path).append("traces").append(trace_file).string();
  if(_weight_table.find(name) == _weight_table.end()) {
    model->initialize_weight(_weight_table[name]);
  }
  _lang_scheduler = LangScheduler::create(name, trace_file, std::move(model), _config, info);
}

void Simulator::finish_language_model(uint32_t model_id) {
  _lang_scheduler->finish_model(model_id);
}

bool Simulator::running() {
  bool running = false;
  running |= !_models.empty();
  running |= !_waiting_to_preload_models.empty();
  running |= !_preloading_models.empty();
  running |= !_ready_to_compute_models.empty();
  for (auto &core : _cores) {
    running = running || core->running();
  }
  running = running || _icnt->running();
  if (_storage_controller)
    running = running || _storage_controller->has_pending();
  else {
    if (_hbm) running = running || _hbm->running();
    if (_ddr) running = running || _ddr->running();
    if (_ssd) running = running || _ssd->running();
  }
  running = running || !_scheduler->empty();
  if(_language_mode) {
    running = running || _lang_scheduler->busy();
  }
  return running;
}

uint64_t Simulator::set_cycle_mask() {
  _cycle_mask = 0x0;
  uint64_t minimum_time = MIN3(_core_time, _mem_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_mem_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _mem_time += _mem_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
  return minimum_time;
}

uint64_t Simulator::count_ticks_before(uint64_t next_tick_ps,
                                       uint64_t period_ps,
                                       uint64_t target_ps) const {
  if (period_ps == 0 || next_tick_ps >= target_ps) return 0;
  return ((target_ps - 1 - next_tick_ps) / period_ps) + 1;
}

void Simulator::advance_idle_time_to(uint64_t target_ps) {
  if (target_ps <= _core_time && target_ps <= _icnt_time && target_ps <= _mem_time)
    return;

  uint64_t core_ticks = count_ticks_before(_core_time, _core_period, target_ps);
  uint64_t icnt_ticks = count_ticks_before(_icnt_time, _icnt_period, target_ps);
  uint64_t mem_ticks = count_ticks_before(_mem_time, _mem_period, target_ps);
  for (auto& core : _cores) {
    core->advance_stalled_cycles(core_ticks);
  }
  if (_icnt && icnt_ticks > 0) _icnt->advance_idle_cycles(icnt_ticks);
  _core_cycles += core_ticks;
  _icnt_cycle += icnt_ticks;
  _mem_cycles += mem_ticks;
  _core_time = std::max(_core_time, target_ps);
  _icnt_time = std::max(_icnt_time, target_ps);
  _mem_time = std::max(_mem_time, target_ps);
  _last_sim_time_ps = std::max(_last_sim_time_ps, target_ps);
}

bool Simulator::can_fast_forward_to(uint64_t target_ps) {
  if (!_config.enable_fast_forward) return false;
  if (_language_mode) return false;
  if (_icnt_interval != 0) return false;
  if (!_models.empty()) return false;
  if (!_waiting_to_preload_models.empty()) return false;
  if (!_ready_to_compute_models.empty()) return false;
  if (!_storage_controller || !_storage_controller->has_pending()) return false;
  if (_storage_controller->has_ready_response()) return false;
  if (!_icnt || !_icnt->supports_fast_forward() || _icnt->running()) return false;
  if (!_scheduler || !_scheduler->can_fast_forward_waiting()) return false;

  uint64_t next_tick = MIN3(_core_time, _mem_time, _icnt_time);
  if (target_ps <= next_tick) return false;

  for (auto& core : _cores) {
    if (!core->can_fast_forward_stalled()) return false;
    if (core->has_memory_request()) return false;
  }
  return true;
}

void Simulator::try_fast_forward() {
  if (!_config.enable_fast_forward || !_storage_controller) return;

  uint64_t target_ps = _storage_controller->next_event_time_ps();
  if (target_ps == std::numeric_limits<uint64_t>::max()) return;
  if (!can_fast_forward_to(target_ps)) return;

  uint64_t core_ticks = count_ticks_before(_core_time, _core_period, target_ps);
  uint64_t icnt_ticks = count_ticks_before(_icnt_time, _icnt_period, target_ps);
  uint64_t mem_ticks = count_ticks_before(_mem_time, _mem_period, target_ps);
  if (core_ticks == 0 && icnt_ticks == 0 && mem_ticks == 0) return;

  for (auto& core : _cores) {
    core->advance_stalled_cycles(core_ticks);
  }
  if (_icnt && icnt_ticks > 0) _icnt->advance_idle_cycles(icnt_ticks);

  _core_cycles += core_ticks;
  _icnt_cycle += icnt_ticks;
  _mem_cycles += mem_ticks;
  _core_time += core_ticks * _core_period;
  _icnt_time += icnt_ticks * _icnt_period;
  _mem_time += mem_ticks * _mem_period;
  _last_sim_time_ps = std::max(_last_sim_time_ps, target_ps);
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    uint32_t port = 0;
    if (_hbm && _hbm->owns_address(access->dram_address))
      port = _hbm->get_channel_id(access);
    return _config.num_cores * _config.icnt_injection_ports_per_core + port;
  } else {
    uint32_t source_port = 0;
    if (access->target_medium == MemoryMedium::HBM && _hbm)
      source_port = _hbm->get_channel_id(access);
    return access->core_id * _config.icnt_injection_ports_per_core +
           (source_port % _config.icnt_injection_ports_per_core);
  }
}

const double Simulator::get_tile_ops() {
  std::chrono::duration<double> duration = _tile_timestamp.back() - _tile_timestamp.front();
  if (_tile_timestamp.empty())
    return 0.0;
  else
    return _tile_timestamp.size() / duration.count();
}

void Simulator::print_simulation_time_summary(double wall_clock_seconds) const {
  const uint64_t core_time_ps = _core_cycles * _core_period;
  const uint64_t icnt_time_ps = _icnt_cycle * _icnt_period;
  const uint64_t dram_time_ps = _mem_cycles * _mem_period;
  const uint64_t global_time_ps =
      std::max({core_time_ps, icnt_time_ps, dram_time_ps, _last_sim_time_ps});

  spdlog::info(
      "simulation time : {:.6f} us | cycles: core={}, icnt={}, mem={}",
      ps_to_us(global_time_ps), 
       _core_cycles, _icnt_cycle, _mem_cycles);
  spdlog::info("wall-clock={:.6f} s",wall_clock_seconds);
}

void Simulator::print_memory_report(uint64_t global_time_ps) const {
  spdlog::info("========== Memory Stall Report ==========");
  uint64_t total_cycles = 0;
  uint64_t total_systolic = 0;
  uint64_t total_vector = 0;
  uint64_t total_idle = 0;
  uint64_t total_memory_wait = 0;
  uint64_t total_dependency_wait = 0;
  uint64_t total_injection_wait = 0;

  for (const auto& core : _cores) {
    CoreRuntimeStats stats = core->get_runtime_stats();
    total_cycles += stats.total_cycles;
    total_systolic += stats.systolic_active_cycles;
    total_vector += stats.vector_active_cycles;
    total_idle += stats.idle_cycles;
    total_memory_wait += stats.memory_wait_cycles;
    total_dependency_wait += stats.dependency_wait_cycles;
    total_injection_wait += stats.request_injection_wait_cycles;
    spdlog::info(
        "[MemoryReport][Core {}] total={} systolic_active={:.2f}% "
        "vector_active={:.2f}% memory_wait={:.2f}% dependency_wait={:.2f}% "
        "request_injection_wait={:.2f}% idle={:.2f}%",
        stats.core_id, stats.total_cycles,
        percent(stats.systolic_active_cycles, stats.total_cycles),
        percent(stats.vector_active_cycles, stats.total_cycles),
        percent(stats.memory_wait_cycles, stats.total_cycles),
        percent(stats.dependency_wait_cycles, stats.total_cycles),
        percent(stats.request_injection_wait_cycles, stats.total_cycles),
        percent(stats.idle_cycles, stats.total_cycles));
  }

  spdlog::info(
      "[MemoryReport][NPU] avg_systolic_active={:.2f}% avg_vector_active={:.2f}% "
      "avg_memory_wait={:.2f}% avg_dependency_wait={:.2f}% "
      "avg_request_injection_wait={:.2f}% avg_idle={:.2f}%",
      percent(total_systolic, total_cycles),
      percent(total_vector, total_cycles),
      percent(total_memory_wait, total_cycles),
      percent(total_dependency_wait, total_cycles),
      percent(total_injection_wait, total_cycles),
      percent(total_idle, total_cycles));

  auto print_tier = [&](const MemoryTierRuntimeStats& stats,
                        double peak_gbps) {
    double bw = effective_bandwidth_gbps(stats, global_time_ps);
    double util = peak_gbps > 0.0 ? 100.0 * bw / peak_gbps : 0.0;
    spdlog::info(
        "[MemoryReport][{}] reads={} writes={} read_bytes={} write_bytes={} "
        "effective_bw={:.3f}GB/s peak_bw={:.3f}GB/s util={:.2f}% "
        "avg_latency={:.6f}us max_latency={:.6f}us dispatch_blocked={}",
        stats.name, stats.read_requests, stats.write_requests, stats.read_bytes,
        stats.write_bytes, bw, peak_gbps, util, avg_latency_us(stats),
        ps_to_us(stats.max_latency_ps), stats.dispatch_blocked);
    spdlog::info(
        "[MemoryReport][{}] core_bytes={} controller_preload_bytes={}",
        stats.name, stats.core_read_bytes + stats.core_write_bytes,
        stats.controller_read_bytes + stats.controller_write_bytes);
  };

  if (_storage_controller) {
    const StorageRuntimeStats& storage = _storage_controller->runtime_stats();
    print_tier(storage.hbm, _config.max_hbm_bandwidth());
    if (_config.ddr.enabled) print_tier(storage.ddr, _config.max_ddr_bandwidth());
    if (_config.ssd.enabled) print_tier(storage.ssd, 0.0);

    double hbm_util = _config.max_hbm_bandwidth() > 0.0
                          ? 100.0 * effective_bandwidth_gbps(storage.hbm, global_time_ps) /
                                _config.max_hbm_bandwidth()
                          : 0.0;
    double ddr_util = (_config.ddr.enabled && _config.max_ddr_bandwidth() > 0.0)
                          ? 100.0 * effective_bandwidth_gbps(storage.ddr, global_time_ps) /
                                _config.max_ddr_bandwidth()
                          : 0.0;
    double avg_memory_wait = percent(total_memory_wait, total_cycles);
    if (avg_memory_wait > 20.0 && hbm_util < 30.0 &&
        (!_config.ddr.enabled || ddr_util < 30.0)) {
      spdlog::info(
          "[MemoryReport][Hint] NPU memory_wait is high while memory BW util is "
          "low; likely limited by request granularity, dependency serialization, "
          "or preload readiness rather than raw HBM/DDR bandwidth.");
    }
    if (storage.hbm.dispatch_blocked + storage.ddr.dispatch_blocked > 0) {
      spdlog::info(
          "[MemoryReport][Hint] Memory dispatch saw backpressure; controller or "
          "channel queues were full for some requests.");
    }
  }
  if (_scheduler) {
    spdlog::info("[MemoryReport][Scheduler] pipeline_data_wait_cycles={}",
                 _scheduler->pipeline_data_wait_cycles());
  }
  spdlog::info("=========================================");
}

void Simulator::write_memory_report_json(uint64_t global_time_ps) const {
  if (_config.memory_report_json.empty()) return;

  json report;
  report["simulation_time_us"] = ps_to_us(global_time_ps);
  report["cores"] = json::array();
  for (const auto& core : _cores) {
    CoreRuntimeStats stats = core->get_runtime_stats();
    report["cores"].push_back({
        {"core_id", stats.core_id},
        {"total_cycles", stats.total_cycles},
        {"systolic_active_cycles", stats.systolic_active_cycles},
        {"vector_active_cycles", stats.vector_active_cycles},
        {"idle_cycles", stats.idle_cycles},
        {"memory_wait_cycles", stats.memory_wait_cycles},
        {"dependency_wait_cycles", stats.dependency_wait_cycles},
        {"request_injection_wait_cycles", stats.request_injection_wait_cycles},
        {"matmul_pe_cycles", stats.matmul_pe_cycles},
    });
  }

  auto tier_json = [&](const MemoryTierRuntimeStats& stats, double peak_gbps) {
    double bw = effective_bandwidth_gbps(stats, global_time_ps);
    return json{
        {"read_requests", stats.read_requests},
        {"write_requests", stats.write_requests},
        {"read_bytes", stats.read_bytes},
        {"write_bytes", stats.write_bytes},
        {"core_read_bytes", stats.core_read_bytes},
        {"core_write_bytes", stats.core_write_bytes},
        {"controller_read_bytes", stats.controller_read_bytes},
        {"controller_write_bytes", stats.controller_write_bytes},
        {"dispatch_blocked", stats.dispatch_blocked},
        {"effective_bandwidth_gbps", bw},
        {"configured_peak_bandwidth_gbps", peak_gbps},
        {"bandwidth_util_percent",
         peak_gbps > 0.0 ? 100.0 * bw / peak_gbps : 0.0},
        {"avg_latency_us", avg_latency_us(stats)},
        {"max_latency_us", ps_to_us(stats.max_latency_ps)},
    };
  };

  if (_storage_controller) {
    const StorageRuntimeStats& storage = _storage_controller->runtime_stats();
    report["memory"]["hbm"] = tier_json(storage.hbm, _config.max_hbm_bandwidth());
    report["memory"]["ddr"] = tier_json(storage.ddr, _config.max_ddr_bandwidth());
    report["memory"]["ssd"] = tier_json(storage.ssd, 0.0);
  }
  if (_scheduler) {
    report["scheduler"]["pipeline_data_wait_cycles"] =
        _scheduler->pipeline_data_wait_cycles();
  }

  std::ofstream out(_config.memory_report_json);
  if (!out) {
    spdlog::error("Failed to open memory report JSON: {}",
                  _config.memory_report_json);
    return;
  }
  out << report.dump(2) << std::endl;
  spdlog::info("Memory report JSON written to {}", _config.memory_report_json);
}

void Simulator::print_final_summary(double wall_clock_seconds) const {
  const uint64_t core_time_ps = _core_cycles * _core_period;
  const uint64_t icnt_time_ps = _icnt_cycle * _icnt_period;
  const uint64_t dram_time_ps = _mem_cycles * _mem_period;
  const uint64_t global_time_ps =
      std::max({core_time_ps, icnt_time_ps, dram_time_ps, _last_sim_time_ps});
  print_memory_report(global_time_ps);
  write_memory_report_json(global_time_ps);
  print_simulation_time_summary(wall_clock_seconds);
}
