#include "Simulator.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "SystolicOS.h"
#include "SystolicWS.h"

namespace fs = std::filesystem;

namespace {

double ps_to_us(uint64_t ps) {
  return static_cast<double>(ps) / 1e6;
}

std::string csv_value(double value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string csv_value(uint64_t value) {
  return std::to_string(value);
}

void write_csv_row(std::ostream& out, const std::vector<std::string>& columns) {
  for (size_t i = 0; i < columns.size(); i++) {
    if (i > 0) out << ',';
    out << columns[i];
  }
  out << '\n';
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
  const uint32_t npu_count = std::max<uint32_t>(config.npu_count, 1);
  auto make_hbm = [&](const SimulationConfig& hbm_config) {
    if (hbm_config.dram_type == DramType::SIMPLE) {
      return std::unique_ptr<Dram>(std::make_unique<SimpleDram>(hbm_config));
    }
    if (hbm_config.dram_type == DramType::RAMULATOR1) {
      return std::unique_ptr<Dram>(std::make_unique<DramRamulator>(hbm_config));
    }
    if (hbm_config.dram_type == DramType::RAMULATOR2) {
      return std::unique_ptr<Dram>(std::make_unique<Hbm>(hbm_config));
    }
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  };

  if (config.dram_type == DramType::SIMPLE) {
    for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id)
      _hbms.push_back(make_hbm(config));
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id)
      _hbms.push_back(make_hbm(config));
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
    for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id)
      _hbms.push_back(make_hbm(config));
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
  std::vector<Dram*> hbm_ptrs;
  hbm_ptrs.reserve(_hbms.size());
  for (auto& hbm : _hbms) hbm_ptrs.push_back(hbm.get());
  _storage_controller =
      std::make_unique<StorageController>(config, hbm_ptrs, _ddr.get(), _ssd.get());
  _residency_managers.resize(npu_count);
  for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id) {
    _residency_managers[npu_id] = std::make_unique<ResidencyManager>();
    _residency_managers[npu_id]->configure_npu(npu_id);
    _residency_managers[npu_id]->configure_capacity(
        _config.hbm_residency_capacity_bytes == 0
            ? _config.hbm.capacity_bytes
            : _config.hbm_residency_capacity_bytes);
  }

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
  _n_memories = config.total_hbm_channels();
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

  for (auto it = _preloading_models.begin(); it != _preloading_models.end();) {
    Model* model = it->get();
    if (model->uses_layer_preload()) {
      model->complete_ready_data_movement_stages(_storage_controller.get(),
                                                 _core_time);
      if (model->has_data_movement_stage_to_submit()) {
        model->submit_next_data_movement_stage(_storage_controller.get(),
                                               _core_time);
      }
    }
    const bool ready = model->uses_layer_preload()
                           ? model->initial_data_movement_stage_ready(
                                 _storage_controller.get())
                           : model->data_movements_ready(
                                 _storage_controller.get());
    if (ready) {
      if (model->uses_layer_preload()) {
        model->complete_ready_data_movement_stages(_storage_controller.get(),
                                                   _core_time);
        if (model->has_data_movement_stage_to_submit())
          model->submit_next_data_movement_stage(_storage_controller.get(),
                                                 _core_time);
      } else {
        model->complete_data_movements(_storage_controller.get());
      }
      spdlog::info("Model {} data ready at {:.6f} us",
                   (*it)->get_name(), ps_to_us(_core_time));
      _ready_to_compute_models.push_back(std::move(*it));
      it = _preloading_models.erase(it);
    } else {
      ++it;
    }
  }

  advance_active_layer_preloads();
  admit_preload_models();
  schedule_ready_models();
}

void Simulator::advance_active_layer_preloads() {
  for (Model* model : _active_layer_preload_models) {
    if (model == nullptr) continue;
    model->complete_ready_data_movement_stages(_storage_controller.get(),
                                               _core_time);
    if (model->has_data_movement_stage_to_submit()) {
      model->submit_next_data_movement_stage(_storage_controller.get(),
                                             _core_time);
    }
  }
}

void Simulator::admit_preload_models() {
  for (Model* model : _active_layer_preload_models) {
    if (model != nullptr &&
        !model->all_data_movement_stages_done(_storage_controller.get())) {
      return;
    }
  }
  for (const auto& model : _ready_to_compute_models) {
    if (model->uses_layer_preload() &&
        !model->all_data_movement_stages_done(_storage_controller.get())) {
      return;
    }
  }

  const uint32_t max_preloading = _config.max_preloading_models;
  while (!_waiting_to_preload_models.empty() &&
         (max_preloading == 0 ||
          _preloading_models.size() < max_preloading)) {
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
                 _preloading_models.size(), max_preloading,
                 _waiting_to_preload_models.size());
    const uint32_t npu_id = launch_model->get_npu_id();
    if (npu_id >= _residency_managers.size()) {
      spdlog::error("Model {} targets invalid npu_id {} (npu_count={})",
                    launch_model->get_name(), npu_id,
                    _residency_managers.size());
      std::exit(EXIT_FAILURE);
    }
    launch_model->set_residency_manager(_residency_managers[npu_id].get());
    launch_model->initialize_model(_weight_table[weight_key]);
    launch_model->prefill_ssd_tensors(_ssd.get());
    std::vector<uint64_t> movement_ids;
    if (launch_model->uses_layer_preload()) {
      movement_ids = launch_model->submit_next_data_movement_stage(
          _storage_controller.get(), _core_time);
    } else {
      movement_ids = launch_model->submit_data_movements(
          _storage_controller.get(), _core_time);
    }

    const bool ready =
        launch_model->uses_layer_preload()
            ? launch_model->initial_data_movement_stage_ready(
                  _storage_controller.get())
            : launch_model->data_movements_ready(_storage_controller.get());
    const bool no_movement_ready =
        movement_ids.empty() &&
        (!launch_model->uses_layer_preload() || ready);
    if (no_movement_ready || ready) {
      launch_model->complete_data_movements(_storage_controller.get());
      if (launch_model->uses_layer_preload()) {
        launch_model->complete_ready_data_movement_stages(
            _storage_controller.get(), _core_time);
        if (launch_model->has_data_movement_stage_to_submit())
          launch_model->submit_next_data_movement_stage(
              _storage_controller.get(), _core_time);
      }
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
    if (ready_model->uses_layer_preload() &&
        !ready_model->all_data_movement_stages_done(_storage_controller.get())) {
      _active_layer_preload_models.push_back(ready_model.get());
    }
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
        while (_cores[core_id]->has_phase_event()) {
          _scheduler->record_core_phase(_cores[core_id]->pop_phase_event());
        }
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
        uint32_t source_port = response->npu_id * _config.hbm.channels;
        Dram* response_hbm = hbm_for_npu(response->npu_id);
        if (response->target_medium == MemoryMedium::HBM && response_hbm) {
          source_port += response_hbm->get_channel_id(response);
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
  for (uint32_t npu_id = 0; npu_id < _hbms.size(); ++npu_id) {
    if (_hbms[npu_id]) _hbms[npu_id]->print_stat();
  }
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

void Simulator::finish_model_compute(Model* model) {
  if (model == nullptr) return;
  model->release_residency_pins();
  _active_layer_preload_models.erase(
      std::remove(_active_layer_preload_models.begin(),
                  _active_layer_preload_models.end(), model),
      _active_layer_preload_models.end());
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
    for (auto& hbm : _hbms)
      if (hbm) running = running || hbm->running();
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

Dram* Simulator::hbm_for_npu(uint32_t npu_id) const {
  if (npu_id >= _hbms.size()) return nullptr;
  return _hbms[npu_id].get();
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    uint32_t port = access->npu_id * _config.hbm.channels;
    Dram* hbm = hbm_for_npu(access->npu_id);
    if (hbm)
      port += hbm->get_channel_id(access);
    return _config.num_cores * _config.icnt_injection_ports_per_core + port;
  } else {
    uint32_t source_port = access->npu_id * _config.hbm.channels;
    Dram* hbm = hbm_for_npu(access->npu_id);
    if (access->target_medium == MemoryMedium::HBM && hbm)
      source_port += hbm->get_channel_id(access);
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
  const uint64_t global_time_ps = final_sim_time_ps();

  spdlog::info(
      "simulation time : {:.6f} us | cycles: core={}, icnt={}, mem={}",
      ps_to_us(global_time_ps), 
       _core_cycles, _icnt_cycle, _mem_cycles);
  spdlog::info("wall-clock={:.6f} s",wall_clock_seconds);
}

void Simulator::print_final_summary(double wall_clock_seconds) const {
  print_simulation_time_summary(wall_clock_seconds);
  write_final_hardware_summary_csv(final_sim_time_ps());
  write_final_compute_activity_csv(final_sim_time_ps());
  write_final_compute_activity_detail_csv();
}

uint64_t Simulator::final_sim_time_ps() const {
  const uint64_t core_time_ps = _core_cycles * _core_period;
  const uint64_t icnt_time_ps = _icnt_cycle * _icnt_period;
  const uint64_t dram_time_ps = _mem_cycles * _mem_period;
  return std::max({core_time_ps, icnt_time_ps, dram_time_ps, _last_sim_time_ps});
}

std::string Simulator::hardware_summary_csv_path() const {
  if (!_config.hardware_summary_csv.empty()) return _config.hardware_summary_csv;
  if (!_config.pipeline_breakdown_csv.empty()) {
    fs::path path(_config.pipeline_breakdown_csv);
    return path.parent_path().append("hardware_summary.csv").string();
  }
  return "hardware_summary.csv";
}

std::string Simulator::compute_activity_csv_path() const {
  fs::path hardware_path(hardware_summary_csv_path());
  if (!hardware_path.parent_path().empty()) {
    return hardware_path.parent_path().append("compute_activity.csv").string();
  }
  return "compute_activity.csv";
}

void Simulator::write_final_compute_activity_csv(uint64_t sim_time_ps) const {
  const std::string output_path = compute_activity_csv_path();
  fs::path path(output_path);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());

  std::ofstream out(output_path);
  if (!out.is_open()) {
    spdlog::warn("Failed to write compute activity CSV: {}", output_path);
    return;
  }

  out << "scope,index,op_id,op_name,sim_time_us,total_core_cycles,"
         "cube_active_cycles,vector_active_cycles,"
         "vector_overlap_with_cube_cycles,"
         "cube_overlap_with_vector_cycles,same_op_overlap_cycles,"
         "union_active_cycles,cube_utilization_percent,"
         "vector_utilization_percent,vector_overlap_utilization_percent\n";

  std::map<uint32_t, OpComputeActivity> overall_by_op;
  uint64_t total_core_cycles = 0;
  uint64_t total_cube_cycles = 0;
  uint64_t total_vector_cycles = 0;
  uint64_t total_union_cycles = 0;
  uint64_t total_overlap_cycles = 0;
  uint64_t total_same_op_overlap_cycles = 0;

  const auto write_activity_row = [&](const std::string& scope,
                                      const std::string& index,
                                      const std::string& op_id,
                                      const std::string& op_name,
                                      uint64_t cycles,
                                      uint64_t cube_cycles,
                                      uint64_t vector_cycles,
                                      uint64_t vector_overlap_cycles,
                                      uint64_t cube_overlap_cycles,
                                      uint64_t same_op_overlap_cycles,
                                      uint64_t union_cycles) {
    const auto utilization = [cycles](uint64_t active) {
      return cycles == 0
                 ? 0.0
                 : static_cast<double>(active) * 100.0 /
                       static_cast<double>(cycles);
    };
    write_csv_row(out, {
        scope, index, op_id, op_name, csv_value(ps_to_us(sim_time_ps)),
        csv_value(cycles), csv_value(cube_cycles), csv_value(vector_cycles),
        csv_value(vector_overlap_cycles), csv_value(cube_overlap_cycles),
        csv_value(same_op_overlap_cycles), csv_value(union_cycles),
        csv_value(utilization(cube_cycles)),
        csv_value(utilization(vector_cycles)),
        csv_value(utilization(vector_overlap_cycles)),
    });
  };

  for (size_t core_id = 0; core_id < _cores.size(); ++core_id) {
    const Core* core = _cores[core_id].get();
    if (core == nullptr) continue;
    const uint64_t cycles = core->get_total_cycles();
    const uint64_t cube_cycles = core->get_cube_active_cycles();
    const uint64_t vector_cycles = core->get_vector_active_cycles();
    const uint64_t union_cycles = core->get_compute_cycles();
    const uint64_t overlap_cycles = core->get_cube_vector_overlap_cycles();
    uint64_t same_op_overlap_cycles = 0;
    for (const auto& [op_id, activity] : core->get_op_compute_activity()) {
      (void)op_id;
      same_op_overlap_cycles += activity.same_op_overlap_cycles;
    }
    write_activity_row("core_total", std::to_string(core_id), "all", "all",
                       cycles, cube_cycles, vector_cycles, overlap_cycles,
                       overlap_cycles, same_op_overlap_cycles, union_cycles);

    total_core_cycles += cycles;
    total_cube_cycles += cube_cycles;
    total_vector_cycles += vector_cycles;
    total_union_cycles += union_cycles;
    total_overlap_cycles += overlap_cycles;
    total_same_op_overlap_cycles += same_op_overlap_cycles;

    for (const auto& [op_id, activity] : core->get_op_compute_activity()) {
      const uint64_t op_union = activity.cube_active_cycles +
                                activity.vector_active_cycles -
                                activity.same_op_overlap_cycles;
      write_activity_row(
          "core_op", std::to_string(core_id), std::to_string(op_id),
          activity.op_name, cycles, activity.cube_active_cycles,
          activity.vector_active_cycles,
          activity.vector_overlap_with_cube_cycles,
          activity.cube_overlap_with_vector_cycles,
          activity.same_op_overlap_cycles, op_union);

      auto& aggregate = overall_by_op[op_id];
      aggregate.op_name = activity.op_name;
      aggregate.cube_active_cycles += activity.cube_active_cycles;
      aggregate.vector_active_cycles += activity.vector_active_cycles;
      aggregate.vector_overlap_with_cube_cycles +=
          activity.vector_overlap_with_cube_cycles;
      aggregate.cube_overlap_with_vector_cycles +=
          activity.cube_overlap_with_vector_cycles;
      aggregate.same_op_overlap_cycles += activity.same_op_overlap_cycles;
    }
  }

  write_activity_row("npu_total", "all", "all", "all", total_core_cycles,
                     total_cube_cycles, total_vector_cycles,
                     total_overlap_cycles, total_overlap_cycles,
                     total_same_op_overlap_cycles, total_union_cycles);
  for (const auto& [op_id, activity] : overall_by_op) {
    const uint64_t op_union = activity.cube_active_cycles +
                              activity.vector_active_cycles -
                              activity.same_op_overlap_cycles;
    write_activity_row(
        "npu_op", "all", std::to_string(op_id), activity.op_name,
        total_core_cycles, activity.cube_active_cycles,
        activity.vector_active_cycles,
        activity.vector_overlap_with_cube_cycles,
        activity.cube_overlap_with_vector_cycles,
        activity.same_op_overlap_cycles, op_union);
  }
}

void Simulator::write_final_compute_activity_detail_csv() const {
  fs::path hardware_path(compute_activity_csv_path());
  const fs::path detail_path =
      hardware_path.parent_path().append("compute_activity_detail.csv");
  const fs::path interval_path =
      hardware_path.parent_path().append("compute_activity_intervals.csv");
  if (!detail_path.parent_path().empty())
    fs::create_directories(detail_path.parent_path());

  std::ofstream detail(detail_path);
  if (!detail.is_open()) {
    spdlog::warn("Failed to write compute activity detail CSV: {}",
                 detail_path.string());
    return;
  }
  detail << "scope,core_id,op_id,op_name,compute_region,total_core_cycles,"
            "cube_active_cycles,vector_active_cycles,cube_busy_time_us,"
            "vector_busy_time_us\n";

  // The simulator uses an integer picosecond core period, so use the same
  // period here as final_sim_time_ps() rather than 1/frequency.  This keeps
  // detail timestamps bit-for-bit consistent with layer_breakdown.csv.
  const double period_us = static_cast<double>(_core_period) / 1e6;
  for (size_t core_id = 0; core_id < _cores.size(); ++core_id) {
    const Core* core = _cores[core_id].get();
    if (core == nullptr) continue;
    const uint64_t total_cycles = core->get_total_cycles();
    for (const auto& [key, activity] : core->get_compute_activity_detail()) {
      const double cube_us = activity.cube_active_cycles * period_us;
      const double vector_us = activity.vector_active_cycles * period_us;
      write_csv_row(detail, {
          "core_region", std::to_string(core_id), std::to_string(key.op_id),
          activity.op_name, key.compute_region,
          std::to_string(total_cycles),
          std::to_string(activity.cube_active_cycles),
          std::to_string(activity.vector_active_cycles), csv_value(cube_us),
          csv_value(vector_us),
      });
    }
  }
  detail.close();

  std::ofstream intervals(interval_path);
  if (!intervals.is_open()) {
    spdlog::warn("Failed to write compute activity intervals CSV: {}",
                 interval_path.string());
    return;
  }
  intervals << "core_id,op_id,op_name,compute_region,resource,start_cycle,"
               "end_cycle,start_us,end_us,duration_us\n";
  for (const auto& core : _cores) {
    if (core == nullptr) continue;
    for (const auto& event : core->get_compute_activity_intervals()) {
      const double start_us = event.start_cycle * period_us;
      const double end_us = event.end_cycle * period_us;
      intervals << event.core_id << ',' << event.op_id << ','
                << event.op_name << ',' << event.compute_region << ','
                << event.resource << ',' << event.start_cycle << ','
                << event.end_cycle << ',' << start_us << ',' << end_us << ','
                << (end_us - start_us) << '\n';
    }
  }
}

void Simulator::append_memory_hardware_summary_rows(std::ostream& out,
                                                    const std::string& name,
                                                    const Dram* memory,
                                                    const TieredMemoryConfig& tier,
                                                    uint64_t sim_time_ps) const {
  const MemoryBandwidthStats stats =
      memory == nullptr ? MemoryBandwidthStats{} : memory->get_bandwidth_stats();
  const double seconds = static_cast<double>(sim_time_ps) / 1e12;
  const double channel_peak_GBps =
      tier.freq == 0 || tier.req_size == 0
          ? 0.0
          : static_cast<double>(tier.freq) *
                static_cast<double>(tier.req_size) /
                static_cast<double>(std::max(tier.nbl, 1u)) / 1000.0;

  uint64_t total_reads = 0;
  uint64_t total_writes = 0;
  uint64_t total_read_bytes = 0;
  uint64_t total_write_bytes = 0;
  double total_bandwidth_GBps = 0.0;
  const size_t channel_count =
      std::max(stats.channel_reads.size(), stats.channel_writes.size());
  for (size_t ch = 0; ch < channel_count; ch++) {
    const uint64_t reads =
        ch < stats.channel_reads.size() ? stats.channel_reads[ch] : 0;
    const uint64_t writes =
        ch < stats.channel_writes.size() ? stats.channel_writes[ch] : 0;
    const uint64_t read_bytes = reads * static_cast<uint64_t>(tier.req_size);
    const uint64_t write_bytes = writes * static_cast<uint64_t>(tier.req_size);
    const uint64_t bytes = read_bytes + write_bytes;
    const double bandwidth_GBps =
        seconds <= 0.0 ? 0.0 : static_cast<double>(bytes) / seconds / 1e9;
    const double bandwidth_util =
        channel_peak_GBps <= 0.0 ? 0.0 : bandwidth_GBps * 100.0 / channel_peak_GBps;
    const double command_util =
        ch < stats.channel_utilization_percent.size()
            ? stats.channel_utilization_percent[ch]
            : bandwidth_util;
    write_csv_row(out, {
        name, "channel", std::to_string(ch), csv_value(command_util),
        csv_value(bandwidth_GBps), csv_value(bandwidth_util),
        csv_value(channel_peak_GBps), csv_value(ps_to_us(sim_time_ps)), "", "",
        csv_value(reads), csv_value(writes), csv_value(read_bytes),
        csv_value(write_bytes), csv_value(bytes), "", "", "", "", "", "", "",
        "", "", csv_value(static_cast<uint64_t>(tier.req_size)),
    });
    total_reads += reads;
    total_writes += writes;
    total_read_bytes += read_bytes;
    total_write_bytes += write_bytes;
    total_bandwidth_GBps += bandwidth_GBps;
  }
  const uint64_t total_bytes = total_read_bytes + total_write_bytes;
  const double peak_bandwidth_GBps =
      channel_peak_GBps * static_cast<double>(channel_count);
  const double bandwidth_util =
      peak_bandwidth_GBps <= 0.0
          ? 0.0
          : total_bandwidth_GBps * 100.0 / peak_bandwidth_GBps;
  const double command_util =
      stats.channel_utilization_percent.empty()
          ? bandwidth_util
          : stats.average_utilization_percent;
  write_csv_row(out, {
      name, "overall", "all", csv_value(command_util),
      csv_value(total_bandwidth_GBps), csv_value(bandwidth_util),
      csv_value(peak_bandwidth_GBps), csv_value(ps_to_us(sim_time_ps)), "", "",
      csv_value(total_reads), csv_value(total_writes), csv_value(total_read_bytes),
      csv_value(total_write_bytes), csv_value(total_bytes), "", "", "", "", "", "",
      "", "", "", csv_value(static_cast<uint64_t>(tier.req_size)),
  });
}

void Simulator::write_final_hardware_summary_csv(uint64_t sim_time_ps) const {
  const std::string output_path = hardware_summary_csv_path();
  fs::path path(output_path);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());

  std::ofstream out(output_path);
  if (!out.is_open()) {
    spdlog::warn("Failed to write hardware summary CSV: {}", output_path);
    return;
  }

  out << "component,scope,index,utilization_percent,bandwidth_GBps,"
         "bandwidth_utilization_percent,peak_bandwidth_GBps,sim_time_us,"
         "cycles,active_cycles,reads,writes,read_bytes,write_bytes,bytes,"
         "iops,read_iops,write_iops,read_bandwidth_GBps,write_bandwidth_GBps,"
         "read_peak_bandwidth_GBps,write_peak_bandwidth_GBps,"
         "read_bandwidth_utilization_percent,"
         "write_bandwidth_utilization_percent,request_size_bytes\n";

  double weighted_core_util = 0.0;
  uint64_t total_core_cycles = 0;
  uint64_t total_core_active_cycles = 0;
  for (size_t core_id = 0; core_id < _cores.size(); core_id++) {
    const Core* core = _cores[core_id].get();
    const double util = core ? core->get_core_utilization_percent() : 0.0;
    const uint64_t cycles = core ? core->get_total_cycles() : 0;
    const uint64_t active_cycles = core ? core->get_compute_cycles() : 0;
    write_csv_row(out, {
        "NPU", "core", std::to_string(core_id), csv_value(util), "", "", "",
        csv_value(ps_to_us(sim_time_ps)), csv_value(cycles),
        csv_value(active_cycles), "", "", "", "", "", "", "", "", "", "", "",
        "", "", "", "",
    });
    weighted_core_util += util * static_cast<double>(cycles);
    total_core_cycles += cycles;
    total_core_active_cycles += active_cycles;
  }
  const double npu_util =
      total_core_cycles == 0
          ? 0.0
          : weighted_core_util / static_cast<double>(total_core_cycles);
  write_csv_row(out, {
      "NPU", "overall", "all", csv_value(npu_util), "", "", "",
      csv_value(ps_to_us(sim_time_ps)), csv_value(total_core_cycles),
      csv_value(total_core_active_cycles), "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "",
  });

  if (_hbms.size() <= 1) {
    append_memory_hardware_summary_rows(
        out, "HBM", _hbms.empty() ? nullptr : _hbms.front().get(),
        _config.hbm, sim_time_ps);
  } else {
    for (uint32_t npu_id = 0; npu_id < _hbms.size(); ++npu_id) {
      append_memory_hardware_summary_rows(
          out, "HBM.npu" + std::to_string(npu_id), _hbms[npu_id].get(),
          _config.hbm, sim_time_ps);
    }
  }
  append_memory_hardware_summary_rows(out, "DDR", _ddr.get(), _config.ddr,
                                      sim_time_ps);

  if (_ssd) {
    const double read_peak = _ssd->read_peak_bandwidth_GBps();
    const double write_peak = _ssd->write_peak_bandwidth_GBps();
    write_csv_row(out, {
        "SSD", "overall", "all",
        csv_value(_ssd->bandwidth_utilization_percent(sim_time_ps)),
        csv_value(_ssd->bandwidth_GBps(sim_time_ps)),
        csv_value(_ssd->bandwidth_utilization_percent(sim_time_ps)), "",
        csv_value(ps_to_us(sim_time_ps)), "", "",
        csv_value(_ssd->total_reads()), csv_value(_ssd->total_writes()),
        csv_value(_ssd->read_bytes()), csv_value(_ssd->write_bytes()),
        csv_value(_ssd->total_bytes()), csv_value(_ssd->iops(sim_time_ps)),
        csv_value(_ssd->read_iops(sim_time_ps)),
        csv_value(_ssd->write_iops(sim_time_ps)),
        csv_value(_ssd->read_bandwidth_GBps(sim_time_ps)),
        csv_value(_ssd->write_bandwidth_GBps(sim_time_ps)),
        csv_value(read_peak), csv_value(write_peak),
        csv_value(_ssd->read_bandwidth_utilization_percent(sim_time_ps)),
        csv_value(_ssd->write_bandwidth_utilization_percent(sim_time_ps)), "",
    });
    write_csv_row(out, {
        "SSD", "read", "all",
        csv_value(_ssd->read_bandwidth_utilization_percent(sim_time_ps)),
        csv_value(_ssd->read_bandwidth_GBps(sim_time_ps)),
        csv_value(_ssd->read_bandwidth_utilization_percent(sim_time_ps)),
        csv_value(read_peak), csv_value(ps_to_us(sim_time_ps)), "", "",
        csv_value(_ssd->total_reads()), "0", csv_value(_ssd->read_bytes()),
        "0", csv_value(_ssd->read_bytes()),
        csv_value(_ssd->read_iops(sim_time_ps)),
        csv_value(_ssd->read_iops(sim_time_ps)), "0",
        csv_value(_ssd->read_bandwidth_GBps(sim_time_ps)), "0",
        csv_value(read_peak), "", 
        csv_value(_ssd->read_bandwidth_utilization_percent(sim_time_ps)),
        "", "",
    });
    write_csv_row(out, {
        "SSD", "write", "all",
        csv_value(_ssd->write_bandwidth_utilization_percent(sim_time_ps)),
        csv_value(_ssd->write_bandwidth_GBps(sim_time_ps)),
        csv_value(_ssd->write_bandwidth_utilization_percent(sim_time_ps)),
        csv_value(write_peak), csv_value(ps_to_us(sim_time_ps)), "", "",
        "0", csv_value(_ssd->total_writes()), "0",
        csv_value(_ssd->write_bytes()), csv_value(_ssd->write_bytes()),
        csv_value(_ssd->write_iops(sim_time_ps)), "0",
        csv_value(_ssd->write_iops(sim_time_ps)), "0",
        csv_value(_ssd->write_bandwidth_GBps(sim_time_ps)), "",
        csv_value(write_peak), "",
        csv_value(_ssd->write_bandwidth_utilization_percent(sim_time_ps)), "",
    });
  } else {
    write_csv_row(out, {
        "SSD", "overall", "all", "0", "0", "0", "",
        csv_value(ps_to_us(sim_time_ps)), "", "", "0", "0", "0", "0", "0",
        "0", "0", "0", "0", "0", "", "", "0", "0", "",
    });
  }
}
