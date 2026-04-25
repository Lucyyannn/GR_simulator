#include "Elementwise.h"

#include "../Model.h"

#include <numeric>
#include <set>

namespace {

uint32_t num_elements(const std::vector<uint32_t>& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1u,
                         std::multiplies<uint32_t>());
}

Opcode parse_elementwise_opcode(const std::string& op) {
  if (op == "add") return Opcode::ADD;
  if (op == "mul") return Opcode::MUL;
  return Opcode::MUL;
}

}  // namespace

Elementwise::Elementwise(SimulationConfig config, Model* model,
                         std::string name,
                         std::map<std::string, std::string>& attributes,
                         uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "Elementwise";
  _input_shape = parse_dims(get_attribute("input_shape"));
  _output_shape = _attributes.count("output_shape")
                      ? parse_dims(get_attribute("output_shape"))
                      : _input_shape;
  _opcode = parse_elementwise_opcode(
      _attributes.count("elementwise_op") ? get_attribute("elementwise_op")
                                          : "mul");

  std::string output_name = _attributes.count("output_name")
                                ? get_attribute("output_name")
                                : name_gen(_name, "output");
  auto output_tensor = std::make_unique<Tensor>(
      _id, output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));
  calculate_loops();
}

void Elementwise::calculate_loops() {
  _num_elements = num_elements(_output_shape);
  uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;
  uint32_t bytes_per_element_pair = _config.precision * 3;
  _elements_per_tile =
      std::max(1u, sram_capacity / std::max(1u, bytes_per_element_pair));
  _elements_per_tile = std::min(_elements_per_tile, std::max(1u, _num_elements));
}

void Elementwise::initialize_tiles(MappingTable& /*mapping_table*/) {
  for (uint32_t offset = 0; offset < _num_elements; offset += _elements_per_tile) {
    uint32_t elements = std::min(_num_elements - offset, _elements_per_tile);
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _name,
        .layer_id = _id,
        .accum = false,
        .skip = false,
    });
    _tiles.push_back(std::move(tile));
    initialize_instructions(_tiles.back().get(), offset, elements);
  }
}

void Elementwise::initialize_instructions(Tile* tile, uint32_t element_offset,
                                          uint32_t elements) {
  addr_type lhs_addr = get_operand_addr(_INPUT_OPERAND);
  addr_type rhs_addr = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type out_addr = get_operand_addr(_OUTPUT_OPERAND);

  uint32_t bytes = elements * _config.precision;
  addr_type lhs_spad = SPAD_BASE;
  addr_type rhs_spad = SPAD_BASE + _config.align_address(bytes);

  std::set<addr_type> lhs_addrs;
  std::set<addr_type> rhs_addrs;
  std::set<addr_type> out_addrs;
  for (uint32_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    lhs_addrs.insert(_config.align_address(
        lhs_addr + element_offset * _config.precision + offset));
    rhs_addrs.insert(_config.align_address(
        rhs_addr + element_offset * _config.precision + offset));
    out_addrs.insert(_config.align_address(
        out_addr + element_offset * _config.precision + offset));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = lhs_spad,
      .size = static_cast<uint32_t>(lhs_addrs.size()),
      .src_addrs = std::vector<addr_type>(lhs_addrs.begin(), lhs_addrs.end()),
      .operand_id = _INPUT_OPERAND,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = rhs_spad,
      .size = static_cast<uint32_t>(rhs_addrs.size()),
      .src_addrs = std::vector<addr_type>(rhs_addrs.begin(), rhs_addrs.end()),
      .operand_id = _INPUT_OPERAND + 1,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = _opcode,
      .dest_addr = lhs_spad,
      .size = static_cast<uint32_t>(lhs_addrs.size()),
      .compute_size = bytes,
      .src_addrs = std::vector<addr_type>{lhs_spad, rhs_spad},
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = lhs_spad,
      .size = static_cast<uint32_t>(out_addrs.size()),
      .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
  }));
}
