/*TODO: implement this */
#include "Concat.h"

#include "../Model.h"
#include "../Tensor.h"

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

Concat::Concat(SimulationConfig config, Model* model,
              	onnx::NodeProto& node_proto, uint32_t target_core) 
    : Operation(config, model, node_proto, target_core) {
	for (auto attribute : node_proto.attribute()) {
		if (attribute.name() == "axis") {
			spdlog::trace("concat axis {}", attribute.ints(0));
      _axis = attribute.ints(0);
		} 
	}

	assert(_axis>=0 && _axis<4);
	std::vector<uint32_t> input0_shape = get_input(0)->get_dims();
	std::vector<uint32_t> input1_shape = get_input(1)->get_dims();
	_output_shape.resize(input0_shape.size());
	for (int i = 0; i < input0_shape.size(); i++) {
		if (i == _axis)
			continue;
		assert(input0_shape[i] == input1_shape[i]);
		_output_shape[i] = input0_shape[i];
	}
	_output_shape[_axis] = input0_shape[_axis] + input1_shape[_axis];

	spdlog::trace("output name : {} {}", node_proto.output(0).c_str(), 
									_output_shape);
	Tensor* predefined_tensor = _model->find_tensor(node_proto.output(0));
	if (predefined_tensor == nullptr) {
		std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
				_id, node_proto.output(0), _output_shape, _config.precision, false);
		_outputs.push_back(output_tensor.get()->get_id());
		_model->add_tensor(std::move(output_tensor));
	} else {
		predefined_tensor->redefine_tensor(_id, _output_shape);
	}
}

Concat::Concat(const Concat& src) : Operation(src) {
	_axis = src._axis;
	_output_shape = src._output_shape;
}

	Concat::Concat(SimulationConfig config, Model* model,
								 std::string name, std::map<std::string, std::string> &attributes, uint32_t target_core)
			: Operation(config, model, name, attributes, target_core) {
			_axis = std::stoi(get_attribute("axis"));
			if (_attributes.count("output_shape")) {
				_output_shape = parse_dims(get_attribute("output_shape"));
				std::string output_name = _attributes.count("output_name")
				                              ? get_attribute("output_name")
				                              : name_gen(_name, "output");
				auto output_tensor = std::make_unique<Tensor>(
				    _id, output_name, _output_shape, _config.precision, false);
				_outputs.push_back(output_tensor->get_id());
				_model->add_tensor(std::move(output_tensor));
			}
	}

void Concat::initialize_tiles(MappingTable& mapping_table) {
	if(_outputs.size() == 0) {
		_output_shape = _model->get_tensor(_inputs[0])->get_dims();
		_output_shape[_axis] = 0;
		for(uint32_t input : _inputs) {
			Tensor* tensor = _model->get_tensor(input);
			_output_shape[_axis] += tensor->get_dims()[_axis];
		}
			std::string output_name = _attributes.count("output_name")
			                              ? get_attribute("output_name")
			                              : name_gen(_name, "output");
			auto output_tensor = std::make_unique<Tensor>(_id, output_name, _output_shape, _config.precision, false);
		_outputs.push_back(output_tensor->get_id());
		_model->add_tensor(std::move(output_tensor));
	}

	if (_attributes.count("modeling_mode") &&
			get_attribute("modeling_mode") == "skip") {
		_tiles.push_back(std::make_unique<Tile>(Tile{
				.status = Tile::Status::INITIALIZED,
				.optype = "Concat",
				.layer_id = _id,
				.accum = false,
				.skip = true,
		}));
		return;
	}

	uint32_t spad_bytes = _config.core_config[target_core].spad_size KB / 2;
	uint64_t elements_per_tile =
			std::max<uint64_t>(1, spad_bytes / std::max<uint32_t>(1, 2 * _config.precision));
	uint64_t axis_base = 0;
	for (uint32_t input_idx = 0; input_idx < _inputs.size(); ++input_idx) {
		auto input_shape = get_input(input_idx)->get_dims();
		uint64_t input_elements = product(input_shape);
		for (uint64_t offset = 0; offset < input_elements;
				 offset += elements_per_tile) {
			uint64_t elements = std::min(input_elements - offset, elements_per_tile);
			initialize_copy_tile(input_idx, offset, elements, axis_base);
		}
		axis_base += input_shape[_axis];
	}
}

void Concat::initialize_instructions(Tile* tile, Mapping mapping) {
}

void Concat::initialize_copy_tile(uint32_t input_idx, uint64_t element_offset,
																	uint64_t elements, uint64_t axis_base) {
	if (elements == 0) return;

	addr_type input_addr = get_operand_addr(_INPUT_OPERAND + input_idx);
	addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);
	auto input_shape = get_input(input_idx)->get_dims();

	std::set<addr_type> input_addrs;
	std::set<addr_type> output_addrs;
	for (uint64_t i = 0; i < elements; ++i) {
		uint64_t input_element = element_offset + i;
		std::vector<uint32_t> coords = unflatten(input_element, input_shape);
		std::vector<uint32_t> output_coords = coords;
		output_coords[_axis] += axis_base;
		uint64_t output_element = flatten(output_coords, _output_shape);

		input_addrs.insert(_config.align_address(
				input_addr + input_element * static_cast<uint64_t>(_config.precision)));
		output_addrs.insert(_config.align_address(
				output_addr + output_element * static_cast<uint64_t>(_config.precision)));
	}

	auto tile = std::make_unique<Tile>(Tile{
			.status = Tile::Status::INITIALIZED,
			.optype = "Concat",
			.layer_id = _id,
			.accum = false,
			.skip = false,
	});
	tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
			.opcode = Opcode::MOVIN,
			.dest_addr = SPAD_BASE,
			.size = static_cast<uint32_t>(input_addrs.size()),
			.src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
			.operand_id = _INPUT_OPERAND + input_idx,
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
