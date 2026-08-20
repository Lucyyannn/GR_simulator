#include "LayerNorm.h"

#include "../Model.h"

#include <numeric>
#include <set>

LayerNorm::LayerNorm(SimulationConfig config, Model* model, std::string name,
                     std::map<std::string, std::string>& attributes,
                     uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LayerNorm";
  _input_shape = parse_dims(get_attribute("input_shape"));
  _output_shape = _attributes.count("output_shape")
                      ? parse_dims(get_attribute("output_shape"))
                      : _input_shape;
  _hidden = _input_shape.empty() ? 0 : _input_shape.back();
  _tokens = 1;
  for (size_t i = 0; i + 1 < _input_shape.size(); i++) _tokens *= _input_shape[i];

  std::string output_name = _attributes.count("output_name")
                                ? get_attribute("output_name")
                                : name_gen(_name, "output");
  auto output_tensor = std::make_unique<Tensor>(
      _id, output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));
  calculate_loops();
}

void LayerNorm::calculate_loops() {
  uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;
  uint32_t bytes_per_token = std::max(1u, _hidden * _config.precision);
  // Reserve room for optional affine weight and bias. Trace inputs are bound
  // after construction, so the conservative reservation also keeps the
  // single-input LayerNorm path safe.
  uint32_t affine_bytes = 2 * bytes_per_token;
  uint32_t activation_capacity =
      sram_capacity > affine_bytes ? sram_capacity - affine_bytes : bytes_per_token;
  _tokens_per_tile = std::max(1u, activation_capacity / bytes_per_token);
  _tokens_per_tile = std::min(_tokens_per_tile, std::max(1u, _tokens));
}

void LayerNorm::initialize_tiles(MappingTable& /*mapping_table*/) {
  for (uint32_t offset = 0; offset < _tokens; offset += _tokens_per_tile) {
    uint32_t tokens = std::min(_tokens - offset, _tokens_per_tile);
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _name,
        .layer_id = _id,
        .accum = false,
        .skip = false,
    });
    _tiles.push_back(std::move(tile));
    initialize_instructions(_tiles.back().get(), offset, tokens);
  }
}

void LayerNorm::initialize_instructions(Tile* tile, uint32_t token_offset,
                                        uint32_t tokens) {
  addr_type input_addr = get_operand_addr(_INPUT_OPERAND);
  addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);
  uint32_t bytes = tokens * _hidden * _config.precision;

  std::set<addr_type> input_addrs;
  std::set<addr_type> weight_addrs;
  std::set<addr_type> bias_addrs;
  std::set<addr_type> output_addrs;
  for (uint32_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    input_addrs.insert(_config.align_address(
        input_addr + token_offset * _hidden * _config.precision + offset));
    output_addrs.insert(_config.align_address(
        output_addr + token_offset * _hidden * _config.precision + offset));
  }
  if (_inputs.size() >= 2) {
    addr_type weight_addr = get_operand_addr(_INPUT_OPERAND + 1);
    for (uint32_t offset = 0; offset < _hidden * _config.precision;
         offset += _config.dram_req_size) {
      weight_addrs.insert(_config.align_address(weight_addr + offset));
    }
  }
  if (_inputs.size() >= 3) {
    addr_type bias_addr = get_operand_addr(_INPUT_OPERAND + 2);
    for (uint32_t offset = 0; offset < _hidden * _config.precision;
         offset += _config.dram_req_size) {
      bias_addrs.insert(_config.align_address(bias_addr + offset));
    }
  }

  std::vector<addr_type> layernorm_sources{SPAD_BASE};
  if (!weight_addrs.empty()) layernorm_sources.push_back(SPAD_BASE + bytes);
  if (!bias_addrs.empty()) {
    layernorm_sources.push_back(
        SPAD_BASE + bytes + _hidden * _config.precision);
  }
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
      .operand_id = _INPUT_OPERAND,
  }));
  if (!weight_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = SPAD_BASE + bytes,
        .size = static_cast<uint32_t>(weight_addrs.size()),
        .src_addrs = std::vector<addr_type>(weight_addrs.begin(), weight_addrs.end()),
        .operand_id = _INPUT_OPERAND + 1,
    }));
  }
  if (!bias_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = SPAD_BASE + bytes + _hidden * _config.precision,
        .size = static_cast<uint32_t>(bias_addrs.size()),
        .src_addrs = std::vector<addr_type>(bias_addrs.begin(), bias_addrs.end()),
        .operand_id = _INPUT_OPERAND + 2,
    }));
  }
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::LAYERNORM,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .compute_size = bytes,
      .src_addrs = std::move(layernorm_sources),
      .tile_m = tokens,
      .vector_rows = tokens,
      .vector_bytes_per_row = _hidden * _config.precision,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
  }));
}
