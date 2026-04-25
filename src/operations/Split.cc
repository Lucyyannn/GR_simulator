#include "Split.h"

#include "../Model.h"

#include <sstream>

Split::Split(SimulationConfig config, Model* model, std::string name,
             std::map<std::string, std::string>& attributes,
             uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "Split";
  _input_shape = parse_dims(get_attribute("input_shape"));
  _axis = static_cast<uint32_t>(std::stoul(get_attribute("axis")));
  _output_names = split_csv(get_attribute("output_names"));

  auto shape_specs = split_csv(get_attribute("output_shapes"));
  for (const auto& shape : shape_specs) {
    _output_shapes.push_back(parse_dims(shape));
  }

  for (size_t i = 0; i < _output_shapes.size(); i++) {
    const std::string output_name =
        i < _output_names.size() ? _output_names[i]
                                 : name_gen(_name, "out", std::to_string(i));
    auto output_tensor = std::make_unique<Tensor>(
        _id, output_name, _output_shapes[i], _config.precision, false);
    _outputs.push_back(output_tensor->get_id());
    _model->add_tensor(std::move(output_tensor));
  }
}

std::vector<std::string> Split::split_csv(const std::string& value) const {
  std::vector<std::string> result;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ';')) {
    if (!token.empty()) result.push_back(token);
  }
  return result;
}

void Split::initialize_tiles(MappingTable& /*mapping_table*/) {
  addr_type input_addr = get_operand_addr(_INPUT_OPERAND);
  uint64_t axis_offset_elems = 0;
  for (size_t i = 0; i < _outputs.size(); i++) {
    addr_type output_addr =
        input_addr + axis_offset_elems * static_cast<uint64_t>(_config.precision);
    _model->get_tensor(_outputs[i])->set_address(output_addr);

    if (_axis < _output_shapes[i].size()) axis_offset_elems += _output_shapes[i][_axis];
  }

  _tiles.push_back(std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED,
      .optype = "Split",
      .layer_id = _id,
      .accum = false,
      .skip = true,
  }));
}
