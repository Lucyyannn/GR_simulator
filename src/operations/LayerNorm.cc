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
  _tokens_per_tile = std::max(1u, sram_capacity / bytes_per_token);
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
  std::set<addr_type> output_addrs;
  for (uint32_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    input_addrs.insert(_config.align_address(
        input_addr + token_offset * _hidden * _config.precision + offset));
    output_addrs.insert(_config.align_address(
        output_addr + token_offset * _hidden * _config.precision + offset));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
      .operand_id = _INPUT_OPERAND,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::LAYERNORM,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .compute_size = bytes,
      .src_addrs = std::vector<addr_type>{SPAD_BASE},
      .tile_m = tokens,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
  }));
}
