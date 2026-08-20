#include "HSTUInputPrep.h"

#include "../Model.h"

#include <algorithm>
#include <set>

HSTUInputPrep::HSTUInputPrep(
    SimulationConfig config, Model* model, std::string name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "HSTUInputPrep";
  _input_shape = parse_dims(get_attribute("input_shape"));
  _output_shape = parse_dims(get_attribute("output_shape"));
  _hidden = _input_shape.back();
  _tokens = 1;
  for (size_t i = 0; i + 1 < _input_shape.size(); ++i) {
    _tokens *= _input_shape[i];
  }
  const uint32_t spad_bytes = _config.core_config[target_core].spad_size KB / 2;
  const uint32_t affine_bytes = 2 * _hidden * _config.precision;
  _tokens_per_tile = std::max<uint32_t>(
      1, (spad_bytes - std::min(spad_bytes, affine_bytes)) /
             std::max<uint32_t>(1, _hidden * _config.precision));
  _tokens_per_tile = std::min(_tokens_per_tile, std::max<uint32_t>(1, _tokens));

  std::string output_name = get_attribute("output_name");
  auto output = std::make_unique<Tensor>(
      _id, output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output->get_id());
  _model->add_tensor(std::move(output));
}

void HSTUInputPrep::initialize_tiles(MappingTable& /*mapping_table*/) {
  for (uint32_t offset = 0; offset < _tokens; offset += _tokens_per_tile) {
    const uint32_t tokens = std::min(_tokens - offset, _tokens_per_tile);
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _name,
        .layer_id = _id,
        .accum = false,
        .skip = false,
    });
    initialize_instructions(tile.get(), offset, tokens);
    _tiles.push_back(std::move(tile));
  }
}

void HSTUInputPrep::initialize_instructions(Tile* tile, uint32_t token_offset,
                                            uint32_t tokens) {
  const uint32_t bytes = tokens * _hidden * _config.precision;
  std::set<addr_type> input_addrs;
  std::set<addr_type> weight_addrs;
  std::set<addr_type> bias_addrs;
  const addr_type input = get_operand_addr(_INPUT_OPERAND);
  const addr_type weight = get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type bias = get_operand_addr(_INPUT_OPERAND + 2);
  for (uint32_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    input_addrs.insert(_config.align_address(
        input + token_offset * _hidden * _config.precision + offset));
  }
  for (uint32_t offset = 0; offset < _hidden * _config.precision;
       offset += _config.dram_req_size) {
    weight_addrs.insert(_config.align_address(weight + offset));
    bias_addrs.insert(_config.align_address(bias + offset));
  }

  const addr_type weight_spad = SPAD_BASE + bytes;
  const addr_type bias_spad = weight_spad + _hidden * _config.precision;
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN, .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
      .operand_id = _INPUT_OPERAND,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN, .dest_addr = weight_spad,
      .size = static_cast<uint32_t>(weight_addrs.size()),
      .src_addrs = std::vector<addr_type>(weight_addrs.begin(), weight_addrs.end()),
      .operand_id = _INPUT_OPERAND + 1,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN, .dest_addr = bias_spad,
      .size = static_cast<uint32_t>(bias_addrs.size()),
      .src_addrs = std::vector<addr_type>(bias_addrs.begin(), bias_addrs.end()),
      .operand_id = _INPUT_OPERAND + 2,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::LAYERNORM, .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .compute_size = bytes,
      .src_addrs = {SPAD_BASE, weight_spad, bias_spad},
      .tile_m = tokens, .vector_rows = tokens,
      .vector_bytes_per_row = _hidden * _config.precision,
  }));
  // The normalized activation remains in UB and is consumed by the following
  // projection GEMM. Deliberately no MOVOUT here.
}
