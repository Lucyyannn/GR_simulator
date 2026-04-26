#include "View.h"

#include "../Model.h"

#include <algorithm>
#include <numeric>
#include <set>

namespace {

uint64_t product(const std::vector<uint32_t>& dims) {
  if (dims.empty()) return 0;
  return std::accumulate(dims.begin(), dims.end(), uint64_t{1},
                         std::multiplies<uint64_t>());
}

std::vector<uint32_t> unflatten(uint64_t index,
                                const std::vector<uint32_t>& dims) {
  std::vector<uint32_t> coords(dims.size(), 0);
  for (int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim) {
    uint32_t extent = std::max<uint32_t>(dims[dim], 1);
    coords[dim] = index % extent;
    index /= extent;
  }
  return coords;
}

uint64_t flatten(const std::vector<uint32_t>& coords,
                 const std::vector<uint32_t>& dims) {
  uint64_t index = 0;
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    index = index * dims[dim] + coords[dim];
  }
  return index;
}

}  // namespace

View::View(SimulationConfig config, Model* model, std::string name,
           std::map<std::string, std::string>& attributes,
           uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "View";
  _input_shape = parse_dims(get_attribute("input_shape"));
  _output_shape = parse_dims(get_attribute("output_shape"));
  if (_attributes.count("dims")) _dims = parse_dims(get_attribute("dims"));
  _output_name = _attributes.count("output_name") ? get_attribute("output_name")
                                                  : name_gen(_name, "output");
  auto output_tensor = std::make_unique<Tensor>(
      _id, _output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));
}

void View::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_attributes.count("modeling_mode") &&
      get_attribute("modeling_mode") == "skip") {
    _model->get_tensor(_outputs[0])->set_address(get_operand_addr(_INPUT_OPERAND));
    _tiles.push_back(std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "View",
        .layer_id = _id,
        .accum = false,
        .skip = true,
    }));
    return;
  }

  uint32_t spad_bytes = _config.core_config[target_core].spad_size KB / 2;
  uint64_t elements_per_tile =
      std::max<uint64_t>(1, spad_bytes / std::max<uint32_t>(1, 2 * _config.precision));
  uint64_t output_elements = product(_output_shape);
  for (uint64_t offset = 0; offset < output_elements;
       offset += elements_per_tile) {
    uint64_t elements = std::min(output_elements - offset, elements_per_tile);
    initialize_copy_tile(offset, elements);
  }
}

uint64_t View::input_element_for_output(uint64_t output_element) const {
  if (_dims.empty()) return output_element;

  std::vector<uint32_t> output_coords = unflatten(output_element, _output_shape);
  std::vector<uint32_t> input_coords(_input_shape.size(), 0);
  for (size_t output_dim = 0; output_dim < _dims.size(); ++output_dim) {
    uint32_t input_dim = _dims[output_dim];
    if (input_dim < input_coords.size() && output_dim < output_coords.size())
      input_coords[input_dim] = output_coords[output_dim];
  }
  return flatten(input_coords, _input_shape);
}

void View::initialize_copy_tile(uint64_t element_offset, uint64_t elements) {
  if (elements == 0) return;

  addr_type input_addr = get_operand_addr(_INPUT_OPERAND);
  addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

  std::set<addr_type> input_addrs;
  std::set<addr_type> output_addrs;
  for (uint64_t i = 0; i < elements; ++i) {
    uint64_t output_element = element_offset + i;
    uint64_t input_element = input_element_for_output(output_element);
    input_addrs.insert(_config.align_address(
        input_addr + input_element * static_cast<uint64_t>(_config.precision)));
    output_addrs.insert(_config.align_address(
        output_addr + output_element * static_cast<uint64_t>(_config.precision)));
  }

  auto tile = std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED,
      .optype = "View",
      .layer_id = _id,
      .accum = false,
      .skip = false,
  });

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
      .operand_id = _INPUT_OPERAND,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::COMP,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(input_addrs.size()),
      .compute_size = static_cast<uint32_t>(elements * _config.precision),
      .src_addrs = std::vector<addr_type>{SPAD_BASE},
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
  }));
  _tiles.push_back(std::move(tile));
}
