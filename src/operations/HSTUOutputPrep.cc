#include "HSTUOutputPrep.h"

#include "../Model.h"

#include <algorithm>
#include <set>

HSTUOutputPrep::HSTUOutputPrep(
    SimulationConfig config, Model* model, std::string name,
    std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "HSTUOutputPrep";
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
             std::max<uint32_t>(1, 2 * _hidden * _config.precision));
  _tokens_per_tile = std::min(_tokens_per_tile, std::max<uint32_t>(1, _tokens));

  auto output = std::make_unique<Tensor>(
      _id, get_attribute("output_name"), _output_shape, _config.precision, false);
  _outputs.push_back(output->get_id());
  _model->add_tensor(std::move(output));
}

void HSTUOutputPrep::initialize_tiles(MappingTable& /*mapping_table*/) {
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

void HSTUOutputPrep::initialize_instructions(Tile* tile, uint32_t token_offset,
                                             uint32_t tokens) {
  const uint32_t bytes = tokens * _hidden * _config.precision;
  std::set<addr_type> attn_addrs;
  std::set<addr_type> u_addrs;
  std::set<addr_type> weight_addrs;
  std::set<addr_type> bias_addrs;
  const addr_type attn = get_operand_addr(_INPUT_OPERAND);
  const addr_type u = get_operand_addr(_INPUT_OPERAND + 1);
  const addr_type weight = get_operand_addr(_INPUT_OPERAND + 2);
  const addr_type bias = get_operand_addr(_INPUT_OPERAND + 3);
  for (uint32_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    const uint64_t base = token_offset * _hidden * _config.precision + offset;
    attn_addrs.insert(_config.align_address(attn + base));
    u_addrs.insert(_config.align_address(u + base));
  }
  for (uint32_t offset = 0; offset < _hidden * _config.precision;
       offset += _config.dram_req_size) {
    weight_addrs.insert(_config.align_address(weight + offset));
    bias_addrs.insert(_config.align_address(bias + offset));
  }

  const addr_type u_spad = SPAD_BASE + bytes;
  const addr_type weight_spad = u_spad + bytes;
  const addr_type bias_spad = weight_spad + _hidden * _config.precision;
  const auto movin = [&](uint32_t operand, addr_type dest,
                         const std::set<addr_type>& addrs) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN, .dest_addr = dest,
        .size = static_cast<uint32_t>(addrs.size()),
        .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
        .operand_id = operand,
    }));
  };
  movin(_INPUT_OPERAND, SPAD_BASE, attn_addrs);
  movin(_INPUT_OPERAND + 1, u_spad, u_addrs);
  movin(_INPUT_OPERAND + 2, weight_spad, weight_addrs);
  movin(_INPUT_OPERAND + 3, bias_spad, bias_addrs);
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::LAYERNORM, .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(attn_addrs.size()), .compute_size = bytes,
      .src_addrs = {SPAD_BASE, weight_spad, bias_spad},
      .tile_m = tokens, .vector_rows = tokens,
      .vector_bytes_per_row = _hidden * _config.precision,
      .compute_region = "hstu.output_layernorm",
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MUL, .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(attn_addrs.size()), .compute_size = bytes,
      .src_addrs = {SPAD_BASE, u_spad},
      .compute_region = "hstu.output_mul",
  }));
  // [u, attention, u * norm(attention)] is a resident logical view consumed
  // by the 3H -> H output GEMM. No concat copy and no MOVOUT are charged.
}
