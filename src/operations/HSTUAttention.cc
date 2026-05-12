#include "HSTUAttention.h"

#include "../Model.h"
#include "../Tensor.h"

#include <algorithm>
#include <limits>
#include <numeric>

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

uint64_t dense_offset(const std::vector<uint32_t>& coords,
                      const std::vector<uint32_t>& dims) {
  uint64_t index = 0;
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    index = index * std::max<uint32_t>(dims[dim], 1) + coords[dim];
  }
  return index;
}

uint32_t capped_u32(uint64_t value) {
  return static_cast<uint32_t>(
      std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
  if (divisor == 0) return 0;
  return (value + divisor - 1) / divisor;
}

}  // namespace

HSTUAttention::HSTUAttention(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes,
                             uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "HSTUAttention";
  _q_shape = parse_dims(get_attribute("q_shape"));
  _k_shape = parse_dims(get_attribute("k_shape"));
  _v_shape = parse_dims(get_attribute("v_shape"));
  _k_cache_shape = parse_dims(get_attribute("k_cache_shape"));
  _v_cache_shape = parse_dims(get_attribute("v_cache_shape"));
  _output_shape = parse_dims(get_attribute("output_shape"));
  if (_attributes.count("kv_axis")) _kv_axis = std::stoul(get_attribute("kv_axis"));
  if (_attributes.count("logical_kv_len"))
    _logical_kv_len = std::stoul(get_attribute("logical_kv_len"));
  if (_attributes.count("current_tokens"))
    _current_tokens = std::stoul(get_attribute("current_tokens"));
  if (_attributes.count("hidden")) _hidden = std::stoul(get_attribute("hidden"));
  if (_attributes.count("attention_score_elements"))
    _attention_score_elements = std::stoull(get_attribute("attention_score_elements"));

  if (_logical_kv_len == 0 && _kv_axis < _k_cache_shape.size())
    _logical_kv_len = _k_cache_shape[_kv_axis] + _current_tokens;
  if (_hidden == 0 && !_output_shape.empty()) _hidden = _output_shape.back();

  std::string output_name = _attributes.count("output_name")
                                ? get_attribute("output_name")
                                : name_gen(_name, "output");
  auto output_tensor = std::make_unique<Tensor>(
      _id, output_name, _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor->get_id());
  _model->add_tensor(std::move(output_tensor));
  calculate_tiles();
}

HSTUAttention::HSTUAttention(const HSTUAttention& src) : Operation(src) {
  _q_shape = src._q_shape;
  _k_shape = src._k_shape;
  _v_shape = src._v_shape;
  _k_cache_shape = src._k_cache_shape;
  _v_cache_shape = src._v_cache_shape;
  _output_shape = src._output_shape;
  _kv_axis = src._kv_axis;
  _logical_kv_len = src._logical_kv_len;
  _current_tokens = src._current_tokens;
  _hidden = src._hidden;
  _attention_score_elements = src._attention_score_elements;
  _dense_elements_per_tile = src._dense_elements_per_tile;
  _kv_rows_per_tile = src._kv_rows_per_tile;
  _output_elements_per_tile = src._output_elements_per_tile;
}

void HSTUAttention::calculate_tiles() {
  const uint32_t spad_bytes = _config.core_config[target_core].spad_size KB / 2;
  _dense_elements_per_tile = std::max<uint64_t>(
      1, spad_bytes / std::max<uint32_t>(1, 2 * _config.precision));
  _output_elements_per_tile = std::max<uint64_t>(
      1, spad_bytes / std::max<uint32_t>(1, 3 * _config.precision));

  uint64_t kv_row_elems = 1;
  if (!_k_cache_shape.empty() && _kv_axis < _k_cache_shape.size()) {
    for (size_t dim = 0; dim < _k_cache_shape.size(); ++dim) {
      if (dim == _kv_axis) continue;
      kv_row_elems *= std::max<uint32_t>(_k_cache_shape[dim], 1);
    }
  } else {
    kv_row_elems = std::max<uint32_t>(_hidden, 1);
  }
  const uint64_t kv_row_bytes = kv_row_elems * _config.precision;
  _kv_rows_per_tile = std::max<uint32_t>(
      1, spad_bytes / std::max<uint64_t>(1, 2 * kv_row_bytes));
}

void HSTUAttention::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_attributes.count("modeling_mode") &&
      get_attribute("modeling_mode") == "skip") {
    _tiles.push_back(std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _name,
        .layer_id = _id,
        .accum = false,
        .skip = true,
    }));
    return;
  }

  initialize_dense_input_tiles(1);
  initialize_dense_input_tiles(2);
  initialize_kv_cache_tiles(3);
  initialize_kv_cache_tiles(4);
  initialize_output_compute_tiles();
}

void HSTUAttention::initialize_dense_input_tiles(uint32_t input_idx) {
  if (input_idx >= _inputs.size()) return;
  Tensor* input = get_input(input_idx);
  const uint64_t elements_total = product(input->get_dims());
  for (uint64_t offset = 0; offset < elements_total;
       offset += _dense_elements_per_tile) {
    const uint64_t elements =
        std::min(elements_total - offset, _dense_elements_per_tile);
    std::set<addr_type> input_addrs;
    collect_dense_addresses(input, offset, elements, input_addrs);
    initialize_movin_compute_tile(_INPUT_OPERAND + input_idx, input_addrs,
                                  static_cast<uint32_t>(input_addrs.size()),
                                  capped_u32(elements * _config.precision));
  }
}

void HSTUAttention::initialize_kv_cache_tiles(uint32_t input_idx) {
  if (input_idx >= _inputs.size()) return;
  Tensor* cache = get_input(input_idx);
  const auto shape = cache->get_dims();
  if (_kv_axis >= shape.size()) return;
  const uint32_t logical_rows = shape[_kv_axis];
  for (uint32_t row = 0; row < logical_rows; row += _kv_rows_per_tile) {
    const uint32_t rows = std::min<uint32_t>(logical_rows - row, _kv_rows_per_tile);
    std::set<addr_type> physical_addrs;
    std::set<addr_type> logical_addrs;
    collect_row_addresses(cache, row, rows, physical_addrs, logical_addrs);
    const bool reuse_aware = cache->has_group_layout() ||
        (cache->has_reuse_layout() && cache->reuse_axis() == _kv_axis);
    const uint32_t logical_request_count = static_cast<uint32_t>(
        reuse_aware ? physical_addrs.size() : logical_addrs.size());
    initialize_movin_compute_tile(_INPUT_OPERAND + input_idx, physical_addrs,
                                  logical_request_count,
                                  capped_u32((reuse_aware ? physical_addrs.size()
                                                          : logical_addrs.size()) *
                                             _config.dram_req_size));
  }
}

void HSTUAttention::initialize_output_compute_tiles() {
  Tensor* q = get_input(0);
  Tensor* output = get_output(0);
  const uint64_t elements_total = product(output->get_dims());
  const uint32_t hidden = std::max<uint32_t>(_hidden, 1);
  const uint64_t batch_tokens = std::max<uint64_t>(1, elements_total / hidden);
  uint64_t score_elements = _attention_score_elements;
  if (score_elements == 0)
    score_elements = batch_tokens * std::max<uint32_t>(_logical_kv_len, 1);
  const uint64_t tokens_per_tile =
      std::max<uint64_t>(1, _output_elements_per_tile / hidden);

  for (uint64_t token_offset = 0; token_offset < batch_tokens;
       token_offset += tokens_per_tile) {
    const uint64_t tokens =
        std::min(batch_tokens - token_offset, tokens_per_tile);
    const uint64_t offset = token_offset * hidden;
    const uint64_t elements = tokens * hidden;
    std::set<addr_type> q_addrs;
    std::set<addr_type> out_addrs;
    collect_dense_addresses(q, offset, elements, q_addrs);
    collect_output_addresses(offset, elements, out_addrs);
    if (q_addrs.empty() || out_addrs.empty()) continue;

    const uint64_t tile_score_elements = std::max<uint64_t>(
        1, ceil_div_u64(score_elements * tokens, batch_tokens));
    const uint32_t effective_kv_len =
        capped_u32(std::max<uint64_t>(1, ceil_div_u64(tile_score_elements,
                                                      tokens)));
    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _name,
        .layer_id = _id,
        .accum = false,
        .skip = false,
    });
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = SPAD_BASE,
        .size = static_cast<uint32_t>(q_addrs.size()),
        .src_addrs = std::vector<addr_type>(q_addrs.begin(), q_addrs.end()),
        .operand_id = _INPUT_OPERAND,
    }));
    append_gemm_compute(tile.get(), capped_u32(tokens), hidden,
                        effective_kv_len, ACCUM_SPAD_BASE);
    append_silu_compute(tile.get(), tile_score_elements);
    append_gemm_compute(tile.get(), capped_u32(tokens), effective_kv_len,
                        hidden, ACCUM_SPAD_BASE);
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = static_cast<uint32_t>(out_addrs.size()),
        .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
    _tiles.push_back(std::move(tile));
  }
}

void HSTUAttention::append_gemm_compute(Tile* tile, uint32_t n, uint32_t c,
                                        uint32_t m, addr_type dest_addr) {
  const uint32_t loop_size =
      std::max<uint32_t>(1, _config.core_config[target_core].core_height);
  for (uint32_t m_offset = 0; m_offset < m; m_offset += loop_size) {
    const uint32_t m_loop = std::min<uint32_t>(loop_size, m - m_offset);
    for (uint32_t c_offset = 0; c_offset < c; c_offset += loop_size) {
      const uint32_t c_loop = std::min<uint32_t>(loop_size, c - c_offset);
      for (uint32_t n_offset = 0; n_offset < n; n_offset += loop_size) {
        const uint32_t n_loop = std::min<uint32_t>(loop_size, n - n_offset);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM_PRELOAD,
            .dest_addr = dest_addr,
            .size = std::max<uint32_t>(n_loop, 1),
            .compute_size = std::max<uint32_t>(n_loop, 1),
            .src_addrs = std::vector<addr_type>{SPAD_BASE, SPAD_BASE},
            .tile_m = m_loop,
            .tile_k = c_loop,
            .tile_n = n_loop,
        }));
      }
    }
  }
}

void HSTUAttention::append_silu_compute(Tile* tile, uint64_t score_elements) {
  const uint64_t bytes =
      std::max<uint64_t>(1, score_elements * _config.precision);
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::SWISH,
      .dest_addr = SPAD_BASE,
      .size = std::max<uint32_t>(
          1, capped_u32(ceil_div_u64(bytes, _config.dram_req_size))),
      .compute_size = capped_u32(bytes),
      .src_addrs = std::vector<addr_type>{ACCUM_SPAD_BASE},
      .src_from_accum = true,
  }));
}

void HSTUAttention::initialize_movin_compute_tile(
    uint32_t operand_id, const std::set<addr_type>& input_addrs,
    uint32_t logical_request_count, uint32_t compute_size) {
  if (input_addrs.empty() || logical_request_count == 0) return;
  auto tile = std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED,
      .optype = _name,
      .layer_id = _id,
      .accum = false,
      .skip = false,
  });
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = SPAD_BASE,
      .size = logical_request_count,
      .src_addrs = std::vector<addr_type>(input_addrs.begin(), input_addrs.end()),
      .operand_id = operand_id,
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::COMP,
      .dest_addr = SPAD_BASE,
      .size = logical_request_count,
      .compute_size = std::max<uint32_t>(compute_size, 1),
      .src_addrs = std::vector<addr_type>{SPAD_BASE},
  }));
  _tiles.push_back(std::move(tile));
}

void HSTUAttention::collect_dense_addresses(
    Tensor* tensor, uint64_t element_offset, uint64_t elements,
    std::set<addr_type>& addrs) {
  const addr_type base = tensor->get_address();
  const uint64_t bytes = elements * _config.precision;
  for (uint64_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    addrs.insert(_config.align_address(
        base + element_offset * _config.precision + offset));
  }
}

void HSTUAttention::collect_row_addresses(
    Tensor* tensor, uint32_t row_start, uint32_t rows,
    std::set<addr_type>& physical_addrs,
    std::set<addr_type>& logical_addrs) {
  const auto shape = tensor->get_dims();
  if (_kv_axis >= shape.size() || rows == 0) return;
  const addr_type base = tensor->get_address();
  const bool reuse_aware = tensor->has_reuse_layout() &&
      tensor->reuse_axis() == _kv_axis &&
      tensor->reuse_logical_to_physical().size() == shape[_kv_axis];
  const bool group_aware = tensor->has_group_layout();

  const auto insert_request_range = [&](addr_type start, uint64_t bytes,
                                        std::set<addr_type>& addrs) {
    for (uint64_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
      addrs.insert(_config.align_address(start + offset));
    }
  };

  if (group_aware && tensor->group_axis() < shape.size() &&
      tensor->group_row_axis() == _kv_axis) {
    uint64_t row_bytes = _config.precision;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (dim == tensor->group_axis() || dim == _kv_axis) continue;
      row_bytes *= std::max<uint32_t>(shape[dim], 1);
    }
    for (uint32_t group = 0; group < shape[tensor->group_axis()]; ++group) {
      for (uint32_t row = row_start; row < row_start + rows; ++row) {
        std::vector<uint32_t> coords(shape.size(), 0);
        coords[tensor->group_axis()] = group;
        coords[_kv_axis] = row;
        const addr_type logical_start =
            base + dense_offset(coords, shape) * static_cast<addr_type>(_config.precision);
        const addr_type physical_start =
            tensor->physical_address(coords, _config.precision);
        insert_request_range(logical_start, row_bytes, logical_addrs);
        insert_request_range(physical_start, row_bytes, physical_addrs);
      }
    }
    return;
  }

  if (shape.size() == 2 && _kv_axis == 0) {
    const uint64_t row_bytes =
        static_cast<uint64_t>(shape[1]) * _config.precision;
    for (uint32_t row = row_start; row < row_start + rows; ++row) {
      const addr_type logical_start =
          base + static_cast<addr_type>(row) * row_bytes;
      addr_type physical_start = logical_start;
      if (reuse_aware) {
        const uint32_t physical_row =
            tensor->reuse_logical_to_physical()[row];
        physical_start =
            base + static_cast<addr_type>(physical_row) *
                       tensor->reuse_row_stride_bytes();
      }
      insert_request_range(logical_start, row_bytes, logical_addrs);
      insert_request_range(physical_start, row_bytes, physical_addrs);
    }
    return;
  }

  const uint64_t total_elements = product(shape);

  for (uint64_t idx = 0; idx < total_elements; ++idx) {
    std::vector<uint32_t> coords = unflatten(idx, shape);
    if (coords[_kv_axis] < row_start ||
        coords[_kv_axis] >= row_start + rows) {
      continue;
    }

    logical_addrs.insert(_config.align_address(
        base + idx * static_cast<uint64_t>(_config.precision)));

    if (group_aware) {
      physical_addrs.insert(_config.align_address(
          tensor->physical_address(coords, _config.precision)));
    } else if (reuse_aware) {
      const auto& logical_to_physical = tensor->reuse_logical_to_physical();
      const uint32_t physical_row = logical_to_physical[coords[_kv_axis]];
      uint64_t inner_index = 0;
      for (size_t dim = 0; dim < coords.size(); ++dim) {
        if (dim == _kv_axis) continue;
        inner_index = inner_index * shape[dim] + coords[dim];
      }
      physical_addrs.insert(_config.align_address(
          base + static_cast<addr_type>(physical_row) *
                     tensor->reuse_row_stride_bytes() +
          inner_index * static_cast<uint64_t>(_config.precision)));
    } else {
      physical_addrs.insert(_config.align_address(
          base + idx * static_cast<uint64_t>(_config.precision)));
    }
  }
}

void HSTUAttention::collect_output_addresses(
    uint64_t element_offset, uint64_t elements,
    std::set<addr_type>& addrs) {
  Tensor* output = get_output(0);
  const addr_type base = output->get_address();
  const uint64_t bytes = elements * _config.precision;
  for (uint64_t offset = 0; offset < bytes; offset += _config.dram_req_size) {
    addrs.insert(_config.align_address(
        base + element_offset * _config.precision + offset));
  }
}
