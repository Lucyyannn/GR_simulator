#include "Simulator.h"

#include <filesystem>
#include <string>

#include "SystolicOS.h"
#include "SystolicWS.h"

namespace fs = std::filesystem;

namespace {

double ps_to_us(uint64_t ps) {
  return static_cast<double>(ps) / 1e6;
}

}  // namespace

Simulator::Simulator(SimulationConfig config, bool language_mode)
    : _config(config), _core_cycles(0), _language_mode(language_mode) {
  // Create dram object
  spdlog::info("Simulator Configuration:");
  for (int i=0; i<config.num_cores;i++)
    spdlog::info("[Core {}] Systolic Array Throughput: {} GFLOPS, Spad size: {} KB, Accumulator size: {} KB",
      i, config.max_systolic_flops(i), config.core_config[i].spad_size, config.core_config[i].accum_spad_size);
  spdlog::info("DRAM Bandwidth {} GB/s", config.max_dram_bandwidth());
  _core_period = 1000000 / (config.core_freq);
  _icnt_period = 1000000 / (config.icnt_freq);
  _dram_period = 1000000 / (config.dram_freq);
  _core_time = 0;
  _dram_time = 0;
  _icnt_time = 0;
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    _dram = std::make_unique<SimpleDram>(config);
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator>(config);
  } 
  else if (config.dram_type == DramType::RAMULATOR2) 
  {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator2 config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator2>(config);
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
    _ssd = std::make_unique<Ssd>(scfg, config.dram_freq);
  }
  _storage_controller =
      std::make_unique<StorageController>(config, _dram.get(), _ssd.get());

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
  _n_memories = config.dram_channels;
  _noc_node_per_core = config.icnt_injection_ports_per_core;
  _memory_req_size = config.dram_req_size;
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    _cores[core_index] = Core::create(core_index, config);
  }

  //Configure Hardware Scheduler
  _scheduler = Scheduler::create(_config, &_core_cycles, &_core_time, this);
  
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
  while (!_models.empty() && _models.front()->get_request_time() <= _core_time) {
    std::unique_ptr<Model> launch_model = std::move(_models.front());
    std::pop_heap(_models.begin(), _models.end(), CompareModel());
    _models.pop_back();

    launch_model->initialize_model(_weight_table[launch_model->get_name()]);
    launch_model->set_request_time(_core_time);
    spdlog::info("Schedule model: {} at {} us", launch_model->get_name(), _core_time);
    _scheduler->schedule_model(std::move(launch_model), 1);
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
      _dram->cycle();
      _dram_cycles++;
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
        uint32_t source_port =
            response->target_medium == MemoryMedium::SSD ? 0
                                                         : _dram->get_channel_id(response);
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
  }
  print_simulation_time_summary();
  /* Print simulation stats */
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    _cores[core_id]->print_stats();
  }
  _icnt->print_stats();
  _dram->print_stat();
  if (_ssd) _ssd->print_stat();
}

void Simulator::register_model(std::unique_ptr<Model> model) {
  if(_weight_table.find(model->get_name()) == _weight_table.end()) {
    model->initialize_weight(_weight_table[model->get_name()]);
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
  for (auto &core : _cores) {
    running = running || core->running();
  }
  running = running || _icnt->running();
  if (_storage_controller)
    running = running || _storage_controller->has_pending();
  else {
    running = running || _dram->running();
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
  uint64_t minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_dram_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _dram_time += _dram_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
  return minimum_time;
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    return _config.num_cores * _config.icnt_injection_ports_per_core + _dram->get_channel_id(access);
  } else {
    return access->core_id * _config.icnt_injection_ports_per_core + (_dram->get_channel_id(access) % _config.icnt_injection_ports_per_core);
  }
}

const double Simulator::get_tile_ops() {
  std::chrono::duration<double> duration = _tile_timestamp.back() - _tile_timestamp.front();
  if (_tile_timestamp.empty())
    return 0.0;
  else
    return _tile_timestamp.size() / duration.count();
}

void Simulator::print_simulation_time_summary() const {
  const uint64_t core_time_ps = _core_cycles * _core_period;
  const uint64_t icnt_time_ps = _icnt_cycle * _icnt_period;
  const uint64_t dram_time_ps = _dram_cycles * _dram_period;
  const uint64_t global_time_ps =
      std::max({core_time_ps, icnt_time_ps, dram_time_ps, _last_sim_time_ps});

  spdlog::info("Simulation Finished: global_sim_time={:.6f} us",
               ps_to_us(global_time_ps));
  spdlog::info(
      "[SIM-TIME] core: cycles={} freq={} MHz sim_time={:.6f} us",
      _core_cycles, _config.core_freq, ps_to_us(core_time_ps));
  spdlog::info(
      "[SIM-TIME] icnt: cycles={} freq={} MHz sim_time={:.6f} us",
      _icnt_cycle, _config.icnt_freq, ps_to_us(icnt_time_ps));
  spdlog::info(
      "[SIM-TIME] mem: cycles={} freq={} MHz sim_time={:.6f} us",
      _dram_cycles, _config.dram_freq, ps_to_us(dram_time_ps));
}
