#include <fstream>
#include <chrono>
#include <filesystem>

#include "Simulator.h"
#include "TraceModel.h"
#include "helper/CommandLineParser.h"
#include "operations/OperationFactory.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose default or language mode, default = default");
  cmd_parser.add_command_line_option<std::string>(
      "trace_file", "input trace file for language mode, default = input.csv");
  cmd_parser.add_command_line_option<std::string>(
      "trace_path", "Path for operator trace JSON file (trace mode)");

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
  OperationFactory::initialize(config);

  /* Apply SSD tensor-placement policy before any Model is loaded. */
  if (config.ssd.enabled && config.ssd.place_threshold_bytes > 0) {
    set_ssd_placement_policy(config.ssd.place_threshold_bytes,
                             config.ssd.address_base);
    spdlog::info(
        "[CONFIG] SSD placement policy: tensors >= {} bytes mapped to 0x{:x}",
        config.ssd.place_threshold_bytes, config.ssd.address_base);
  }

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
  } else {
    spdlog::error("Invalid mode: {}", mode);
    return 1;
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

      auto model = std::make_unique<TraceModel>(
          trace_path, model_config, config, model_name, mapping_table);
      spdlog::info("Register Trace Model: {}", model_name);
      simulator->register_model(std::move(model));
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
  spdlog::info("Simulation time: {:2f} seconds", duration.count());
  spdlog::info("Total tile: {}, simulated tile per seconds(TPS): {:3f}",
    simulator->get_number_tile(), simulator->get_tile_ops());
  return 0;
}
