#include "View.h"

#include "../Model.h"

View::View(SimulationConfig config, Model* model, std::string name,
           std::map<std::string, std::string>& attributes,
           uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "View";
  _output_shape = parse_dims(get_attribute("output_shape"));
  _output_name = _attributes.count("output_name") ? get_attribute("output_name")
                                                  : name_gen(_name, "output");
  auto output_tensor = std::make_unique<Tensor>(
      _id, _output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));
}

void View::initialize_tiles(MappingTable& /*mapping_table*/) {
  auto output_addr = get_operand_addr(_INPUT_OPERAND);
  _model->get_tensor(_outputs[0])->set_address(output_addr);

  _tiles.push_back(std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED,
      .optype = "View",
      .layer_id = _id,
      .accum = false,
      .skip = true,
  }));
}
