#include <fstream>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#include "Simulator.h"
#include "TraceModel.h"
#include "benchmark/MemBenchmark.h"
#include "helper/CommandLineParser.h"
#include "operations/OperationFactory.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

void configure_multi_npu_runtime(SimulationConfig& config,
                                 uint32_t npu_count) {
  if (npu_count == 0) {
    spdlog::error("npu_count must be >= 1");
    std::exit(EXIT_FAILURE);
  }

  config.npu_count = npu_count;
  if (config.cores_per_npu == 0) config.cores_per_npu = config.num_cores;
  if (config.hbm_channels_per_npu == 0)
    config.hbm_channels_per_npu = config.hbm.channels;
  if (npu_count == 1) return;

  const uint32_t cores_per_npu = config.cores_per_npu;
  const uint32_t total_cores = cores_per_npu * npu_count;
  CoreConfig* expanded = new CoreConfig[total_cores];
  for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id) {
    for (uint32_t local_core = 0; local_core < cores_per_npu; ++local_core) {
      expanded[npu_id * cores_per_npu + local_core] =
          config.core_config[local_core];
    }
  }
  config.core_config = expanded;
  config.num_cores = total_cores;
  config.partiton_map.clear();
  for (uint32_t npu_id = 0; npu_id < npu_count; ++npu_id) {
    auto& partition = config.partiton_map[npu_id];
    for (uint32_t local_core = 0; local_core < cores_per_npu; ++local_core) {
      partition.push_back(npu_id * cores_per_npu + local_core);
    }
  }
  spdlog::info("[CONFIG] Multi-NPU runtime: npu_count={}, cores_per_npu={}, "
               "total_cores={}, hbm_channels_per_npu={}",
               config.npu_count, config.cores_per_npu, config.num_cores,
               config.hbm.channels);
}

}  // namespace

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info, warn, error], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose default or language mode, default = default");
  cmd_parser.add_command_line_option<std::string>(
      "trace_file", "input trace file for language mode, default = input.csv");
  cmd_parser.add_command_line_option<std::string>(
      "trace_path", "Path for operator trace JSON file (trace mode)");
  cmd_parser.add_command_line_option<std::string>(
      "bench_config", "Path for mem_benchmark configuration JSON");
  cmd_parser.add_command_line_option<std::string>(
      "bench_output_dir", "Output directory for mem_benchmark CSV/chart inputs");
  cmd_parser.add_command_line_option<uint32_t>(
      "npu_count", "Number of same-config NPUs for trace-mode sharding");

  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env) : std::string("./");

  std::string model_base_path = fs::path(onnxim_path).append("models");
  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);
  else if (level == "warn" || level == "warning")
    spdlog::set_level(spdlog::level::warn);
  else if (level == "error" || level == "err")
    spdlog::set_level(spdlog::level::err);

  std::string config_path;
  cmd_parser.set_if_defined("config", &config_path);

  json config_json;
  std::ifstream config_file(config_path);
  if (!config_file) {
    spdlog::error("Error opening file: {}", config_path);
    exit(EXIT_FAILURE);
  }

  config_file >> config_json;
  config_file.close();
  SimulationConfig config = initialize_config(config_json);
  uint32_t npu_count = 1;
  cmd_parser.set_if_defined("npu_count", &npu_count);
  configure_multi_npu_runtime(config, npu_count);
  OperationFactory::initialize(config);

  configure_tensor_placement_policy(config);
  spdlog::info("[CONFIG] Tensor placement policy configured");

  std::string mode = "default";
  bool language_mode = false;
  cmd_parser.set_if_defined("mode", &mode);
  if (mode == "default") {
    spdlog::info("Running in default mode");
  } else if (mode == "language") {
    spdlog::info("Running in language mode");
    language_mode = true;
  } else if (mode == "trace") {
    spdlog::info("Running in trace mode");
  } else if (mode == "mem_bench") {
    spdlog::info("Running in mem_bench mode");
  } else {
    spdlog::error("Invalid mode: {}", mode);
    return 1;
  }

  if (mode == "mem_bench") {
    std::string bench_config_path;
    cmd_parser.set_if_defined("bench_config", &bench_config_path);
    if (bench_config_path.empty()) {
      spdlog::error("bench_config must be provided in mem_bench mode");
      return 1;
    }

    json bench_json;
    std::ifstream bench_config_file(bench_config_path);
    if (!bench_config_file) {
      spdlog::error("Error opening file: {}", bench_config_path);
      return 1;
    }
    bench_config_file >> bench_json;
    bench_config_file.close();

    std::string bench_output_dir =
        bench_json.value("output_dir", std::string("results/mem_benchmark"));
    cmd_parser.set_if_defined("bench_output_dir", &bench_output_dir);

    MemBenchmarkRunner runner(config, bench_json, bench_output_dir);
    runner.run();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Wall-clock runtime: {:2f} seconds", duration.count());
    return 0;
  }

  std::string models_list_path;
  cmd_parser.set_if_defined("models_list", &models_list_path);
  std::ifstream models_list_file(models_list_path);
  if (!models_list_file) {
    spdlog::error("Error opening file: {}", models_list_path);
    exit(EXIT_FAILURE);
  }

  json models_list;
  models_list_file >> models_list;
  models_list_file.close();
  auto simulator = std::make_unique<Simulator>(config, language_mode);
  for (json model_config : models_list["models"]) {
    if(language_mode) {
      std::string model_name = model_config["name"];
      std::string model_path =
        fmt::format("{}/{}/{}.json", model_base_path, "language_models", model_name);
      std::ifstream model_file(model_path);
      if (!models_list_file) {
        spdlog::error("Error opening file: {}", model_path);
        exit(EXIT_FAILURE);
      }
      std::string input_trace = "input.csv";
      cmd_parser.set_if_defined("trace_file", &input_trace);
      model_config["trace_file"] = input_trace;

      json model_json = json::parse(model_file);
      auto model = std::make_unique<LanguageModel>(model_json, config, model_name);
      spdlog::info("Register Language Model: {}", model_name);
      simulator->register_language_model(model_config, std::move(model));
    }
    else if (mode == "trace") {
      std::string model_name = model_config["name"];
      std::string trace_path;
      if (model_config.contains("trace_path"))
        trace_path = model_config["trace_path"];
      cmd_parser.set_if_defined("trace_path", &trace_path);
      if (trace_path.empty()) {
        spdlog::error("trace_path not specified for model {}", model_name);
        exit(EXIT_FAILURE);
      }

      MappingTable mapping_table(config);
      if (model_config.contains("mapping_path")) {
        std::string mapping_path = model_config["mapping_path"];
        mapping_table = MappingTable::parse_mapping_file(mapping_path, config);
      }

      if (config.npu_count > 1) {
        std::vector<uint32_t> users;
        if (model_config.contains("user_ids") && model_config["user_ids"].is_array()) {
          for (const auto& user : model_config["user_ids"])
            users.push_back(user.get<uint32_t>());
        }
        const uint32_t logical_batch_size =
            model_config.value("batch_size",
                               users.empty() ? config.npu_count
                                             : static_cast<uint32_t>(users.size()));
        if (logical_batch_size % config.npu_count != 0) {
          spdlog::error("Trace model {} batch_size {} is not divisible by "
                        "npu_count {}",
                        model_name, logical_batch_size, config.npu_count);
          return 1;
        }
        const uint32_t shard_batch_size = logical_batch_size / config.npu_count;
        if (!users.empty() && users.size() != logical_batch_size) {
          spdlog::error("Trace model {} user_ids size {} does not match "
                        "batch_size {}",
                        model_name, users.size(), logical_batch_size);
          return 1;
        }
        for (uint32_t npu_id = 0; npu_id < config.npu_count; ++npu_id) {
          json shard_config = model_config;
          shard_config["npu_id"] = npu_id;
          shard_config["partition_id"] = npu_id;
          shard_config["logical_batch_size"] = logical_batch_size;
          shard_config["batch_size"] = shard_batch_size;
          if (!users.empty()) {
            std::vector<uint32_t> shard_users;
            auto begin = users.begin() + npu_id * shard_batch_size;
            auto end = begin + shard_batch_size;
            shard_users.assign(begin, end);
            shard_config["user_ids"] = shard_users;
            shard_config["user_id"] = shard_users.front();
          }
          if (!shard_config.contains("weight_key"))
            shard_config["weight_key"] = model_name;
          std::string shard_name =
              model_name + ".npu" + std::to_string(npu_id);
          auto model = std::make_unique<TraceModel>(
              trace_path, shard_config, config, shard_name, mapping_table);
          spdlog::info("Register Trace Model: {} npu_id={}", shard_name,
                       npu_id);
          simulator->register_model(std::move(model));
        }
      } else {
        auto model = std::make_unique<TraceModel>(
            trace_path, model_config, config, model_name, mapping_table);
        spdlog::info("Register Trace Model: {}", model_name);
        simulator->register_model(std::move(model));
      }
    }
    else {
      std::string model_name = model_config["name"];
      std::string onnx_path =
          fmt::format("{}/{}/{}.onnx", model_base_path, model_name, model_name);
      std::string mapping_path = fmt::format("{}/{}/{}.mapping", model_base_path,
                                            model_name, model_name);
      MappingTable mapping_table = MappingTable::parse_mapping_file(mapping_path, config);

      auto model = std::make_unique<Model>(onnx_path, model_config, config, model_name, mapping_table);
      spdlog::info("Register model: {}", model_name);
      simulator->register_model(std::move(model));
    }
  }
  simulator->run_simulator();

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  simulator->print_final_summary(duration.count());
  //spdlog::info("Total tile: {}, simulated tile per seconds(TPS): {:3f}",simulator->get_number_tile(), simulator->get_tile_ops());
  return 0;
}
