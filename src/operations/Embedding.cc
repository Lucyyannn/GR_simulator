#include "Embedding.h"

#include "../Model.h"
#include "../Tensor.h"

#include <numeric>
#include <set>

namespace {

uint32_t product_or_one(const std::vector<uint32_t>& dims) {
  if (dims.empty()) return 1;
  return std::accumulate(dims.begin(), dims.end(), 1u, std::multiplies<uint32_t>());
}

}  // namespace

Embedding::Embedding(SimulationConfig config, Model* model,
                     onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "Embedding";
  _weight_input_idx = 0;
  _indices_input_idx = 1;
  _weight_shape = get_input(_weight_input_idx)->get_dims();
  _indices_shape = get_input(_indices_input_idx)->get_dims();
  infer_output_shape();

  if (!node_proto.output().empty()) {
    Tensor* predefined_tensor = _model->find_tensor(node_proto.output(0));
    if (predefined_tensor == nullptr) {
      auto output_tensor = std::make_unique<Tensor>(
          _id, node_proto.output(0), _output_shape, _config.precision, false);
      _outputs.push_back(output_tensor->get_id());
      _model->add_tensor(std::move(output_tensor));
    } else {
      predefined_tensor->redefine_tensor(_id, _output_shape);
      _outputs.push_back(predefined_tensor->get_id());
    }
  }

  calculate_loops();
}

Embedding::Embedding(SimulationConfig config, Model* model, std::string name,
                     std::map<std::string, std::string>& attrs,
                     uint32_t target_core)
    : Operation(config, model, name, attrs, target_core) {
  _optype = "Embedding";
  _weight_input_idx = 0;
  _indices_input_idx = 1;
  _weight_shape = parse_dims(get_attribute("weight_shape"));
  _indices_shape = parse_dims(get_attribute("indices_shape"));
  if (_attributes.find("indices_values") != _attributes.end())
    _indices_values = parse_dims(get_attribute("indices_values"));
  if (_attributes.find("output_shape") != _attributes.end())
    _output_shape = parse_dims(get_attribute("output_shape"));
  infer_output_shape();
  if (_attributes.find("indices_dtype") != _attributes.end())
    _index_element_bytes = get_dtype_bytes(get_attribute("indices_dtype"));
  if ((_attributes.count("modeling_mode") &&
       _attributes.at("modeling_mode") == "preloaded_rows") ||
      (_attributes.count("preloaded_rows") &&
       _attributes.at("preloaded_rows") == "1")) {
    _preloaded_rows = true;
  }

  std::string output_name = _attributes.count("output_name")
                                ? _attributes["output_name"]
                                : name_gen(_name, "out");
  auto output_tensor = std::make_unique<Tensor>(
      _id, output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));

  if (!_preloaded_rows)
    calculate_loops();
}

void Embedding::infer_output_shape() {
  if (!_output_shape.empty()) return;

  if (_weight_shape.size() != 2) {
    spdlog::error("[Embedding] weight tensor must be 2D, got {}", _weight_shape);
    throw std::runtime_error("Embedding weight tensor must be 2D");
  }

  _output_shape = _indices_shape;
  _output_shape.push_back(_weight_shape.back());
}

void Embedding::calculate_loops() {
  if (_weight_shape.size() != 2) {
    spdlog::error("[Embedding] unsupported weight shape {}", _weight_shape);
    throw std::runtime_error("Embedding weight shape must be 2D");
  }

  _vocab_size = _weight_shape[0];
  _embedding_dim = _weight_shape[1];
  _num_lookups = product_or_one(_indices_shape);
  if (!_indices_values.empty() && _indices_values.size() != _num_lookups) {
    spdlog::error(
        "[Embedding] indices_values size {} does not match flattened indices shape {}",
        _indices_values.size(), _num_lookups);
    throw std::runtime_error("Embedding indices_values size mismatch");
  }

  uint32_t row_bytes = _embedding_dim * _config.precision;
  uint32_t bytes_per_lookup = row_bytes * 2 + _index_element_bytes;
  uint32_t sram_capacity =
      _config.core_config[target_core].spad_size KB / 2;

  _rows_per_tile = std::max(1u, sram_capacity / std::max(1u, bytes_per_lookup));
  _rows_per_tile = std::min(_rows_per_tile, std::max(1u, _num_lookups));

  spdlog::info(
      "[Embedding] vocab_size={}, embedding_dim={}, num_lookups={}, rows_per_tile={}",
      _vocab_size, _embedding_dim, _num_lookups, _rows_per_tile);
}

void Embedding::initialize_tiles(MappingTable& mapping_table) {
  (void)mapping_table;
  if (_preloaded_rows) {
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "Embedding",
        .layer_id = _id,
        .accum = false,
        .skip = true,
    });
    _tiles.push_back(std::move(tile));
    return;
  }
  for (uint32_t lookup_offset = 0; lookup_offset < _num_lookups;
       lookup_offset += _rows_per_tile) {
    uint32_t lookups = std::min(_num_lookups - lookup_offset, _rows_per_tile);
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "Embedding",
        .layer_id = _id,
        .accum = false,
        .skip = false,
    });
    _tiles.push_back(std::move(tile));
    initialize_instructions(_tiles.back().get(), Mapping{}, lookup_offset, lookups);
  }
}

void Embedding::initialize_instructions(Tile* tile, Mapping mapping,
                                        uint32_t lookup_offset,
                                        uint32_t lookups) {
  (void)mapping;
  addr_type sram_base = SPAD_BASE;
  addr_type index_addr = get_operand_addr(_INPUT_OPERAND + _indices_input_idx);
  addr_type weight_addr = get_operand_addr(_INPUT_OPERAND + _weight_input_idx);
  addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

  uint32_t row_bytes = _embedding_dim * _config.precision;
  uint32_t index_bytes = lookups * _index_element_bytes;
  addr_type weight_sram_base =
      sram_base + _config.align_address(std::max(index_bytes, _config.dram_req_size));

  std::set<addr_type> index_addrs;
  for (uint32_t offset = 0; offset < index_bytes; offset += _config.dram_req_size) {
    index_addrs.insert(_config.align_address(
        index_addr + lookup_offset * _index_element_bytes + offset));
  }

  std::set<addr_type> weight_addrs;
  std::set<addr_type> output_addrs;
  for (uint32_t i = 0; i < lookups; ++i) {
    uint32_t row = get_lookup_row(lookup_offset + i);
    addr_type row_base = weight_addr + static_cast<addr_type>(row) * row_bytes;
    addr_type output_row_base =
        output_addr + static_cast<addr_type>(lookup_offset + i) * row_bytes;

    for (uint32_t offset = 0; offset < row_bytes; offset += _config.dram_req_size) {
      weight_addrs.insert(_config.align_address(row_base + offset));
      output_addrs.insert(_config.align_address(output_row_base + offset));
    }
  }

  if (!index_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_base,
        .size = static_cast<uint32_t>(index_addrs.size()),
        .src_addrs = std::vector<addr_type>(index_addrs.begin(), index_addrs.end()),
        .operand_id = _INPUT_OPERAND + _indices_input_idx,
    }));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = weight_sram_base,
      .size = static_cast<uint32_t>(weight_addrs.size()),
      .src_addrs = std::vector<addr_type>(weight_addrs.begin(), weight_addrs.end()),
      .operand_id = _INPUT_OPERAND + _weight_input_idx,
  }));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = weight_sram_base,
      .size = static_cast<uint32_t>(output_addrs.size()),
      .src_addrs = std::vector<addr_type>(output_addrs.begin(), output_addrs.end()),
      .operand_id = _OUTPUT_OPERAND,
  }));
}

uint32_t Embedding::get_dtype_bytes(const std::string& dtype) const {
  if (dtype.find("64") != std::string::npos) return 8;
  if (dtype.find("32") != std::string::npos) return 4;
  if (dtype.find("16") != std::string::npos) return 2;
  if (dtype.find("8") != std::string::npos) return 1;
  return 4;
}

uint32_t Embedding::get_lookup_row(uint32_t lookup_idx) const {
  if (_vocab_size == 0) return 0;
  if (_indices_values.empty()) return lookup_idx % _vocab_size;
  return _indices_values.at(lookup_idx) % _vocab_size;
}
