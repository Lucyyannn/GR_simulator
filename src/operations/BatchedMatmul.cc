#include "BatchedMatmul.h"

#include "../Model.h"

#include <set>

BatchedMatmul::BatchedMatmul(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes,
                             uint32_t target_core)
    : Gemm(config, model, name, attributes, target_core) {
  _optype = "BatchedMatmul";
  validate_shapes();
  Cdim_w = 1;
  Mdim = 2;
  _batch_dim = _input_shape[0];
  _rows_per_batch = _input_shape[1];
}

void BatchedMatmul::validate_shapes() const {
  if (_input_shape.size() != 3 || _weight_shape.size() != 3 ||
      _output_shape.size() != 3) {
    spdlog::error(
        "[BatchedMatmul] expected rank-3 tensors, got input={}, weight={}, output={}",
        _input_shape, _weight_shape, _output_shape);
    throw std::runtime_error("BatchedMatmul expects rank-3 tensors");
  }
  if (_input_shape[0] != _weight_shape[0] || _input_shape[0] != _output_shape[0]) {
    spdlog::error(
        "[BatchedMatmul] batch dimension mismatch input={}, weight={}, output={}",
        _input_shape, _weight_shape, _output_shape);
    throw std::runtime_error("BatchedMatmul batch dimension mismatch");
  }
  if (_input_shape[1] != _output_shape[1] || _input_shape[2] != _weight_shape[1] ||
      _weight_shape[2] != _output_shape[2]) {
    spdlog::error(
        "[BatchedMatmul] incompatible matmul shapes input={}, weight={}, output={}",
        _input_shape, _weight_shape, _output_shape);
    throw std::runtime_error("BatchedMatmul shape mismatch");
  }
}

void BatchedMatmul::initialize_tiles(MappingTable& mapping_table) {
  Mapping::LoopCounts key{.N = _rows_per_batch * _batch_dim,
                          .C = _weight_shape[Cdim_w],
                          .M = _weight_shape[Mdim],
                          .S = 1,
                          .R = 1,
                          .Q = 1,
                          .P = 1,
                          .target_core = target_core};

  Mapping mapping;
  try {
    mapping = mapping_table.at(key);
  } catch (const std::out_of_range& e) {
    spdlog::error(
        "[BatchedMatmul] key not found: N: {} C: {} M: {} P: {} Q: {} S: {} R: {}",
        key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }

  const uint32_t npu_cores =
      _config.cores_per_npu == 0 ? _config.num_cores : _config.cores_per_npu;
  int core_id = -1;
  uint32_t n_tiles_per_batch =
      (_rows_per_batch + mapping.tile_in_loop.N - 1) / mapping.tile_in_loop.N;
  for (uint32_t batch = 0; batch < _batch_dim; ++batch) {
    for (uint32_t n_tile = 0; n_tile < n_tiles_per_batch; ++n_tile) {
      for (uint32_t M = 0; M < mapping.tile_out_loop.M; ++M) {
        for (uint32_t C = 0; C < mapping.tile_out_loop.C; ++C) {
          if (C == 0) {
            core_id = (core_id + 1) % npu_cores;
          }
          auto tile = std::make_unique<Tile>(Tile{
              .status = Tile::Status::INITIALIZED,
              .optype = _optype,
              .layer_id = _id,
              .batch = batch,
              .Q = n_tile,
              .P = 1,
              .M = M,
              .C = C,
              .S = 1,
              .R = 1,
              .accum = C != 0,
              .core_id = core_id,
          });
          _tiles.push_back(std::move(tile));
          initialize_instructions(_tiles.back().get(), mapping);
          if (_tiles.back()->instructions.empty())
            _tiles.pop_back();
        }
      }
    }
  }
}

void BatchedMatmul::initialize_instructions(Tile* tile, Mapping mapping) {
  int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  int tout_c_offset = tile->C * mapping.tile_in_loop.C;
  int tout_n_offset = tile->Q * mapping.tile_in_loop.N;
  int elems_per_access = _config.dram_req_size / _config.precision;
  uint32_t batch_idx = tile->batch;

  addr_type act_sp_base_addr = SPAD_BASE;
  addr_type weight_sp_base_addr =
      SPAD_BASE + mapping.tile_in_loop.N * mapping.tile_in_loop.C * _config.precision;

  addr_type first_addr = get_operand_addr(_INPUT_OPERAND);
  addr_type second_addr = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type output_addr = get_operand_addr(_OUTPUT_OPERAND);

  int loop_size = _config.core_config[target_core].core_height;
  int cloop_size = mapping.tile_in_loop.C;
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if (m_loop <= 0) break;

    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += cloop_size) {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                          ? mapping.total_loop.C - C_offset
                          : cloop_size;
      addr_type weight_sp_addr =
          weight_sp_base_addr +
          (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      std::set<addr_type> weight_set;
      for (int iter_m = 0; iter_m < m_loop; ++iter_m) {
        for (int iter_c = 0; iter_c < c_in_loop; iter_c += elems_per_access) {
          int C = C_offset + iter_c;
          int M = M_offset + iter_m;
          weight_set.insert(second_addr + make_address({batch_idx, static_cast<uint32_t>(C),
                                                        static_cast<uint32_t>(M)},
                                                       _weight_shape));
        }
      }
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = weight_sp_addr,
          .size = static_cast<uint32_t>(weight_set.size()),
          .src_addrs = std::vector<addr_type>(weight_set.begin(), weight_set.end()),
          .operand_id = _INPUT_OPERAND + 1,
          .tile_m = mapping.tile_in_loop.M,
          .tile_k = mapping.tile_in_loop.C}));

      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > static_cast<int>(_rows_per_batch)
                         ? static_cast<int>(_rows_per_batch) - N_offset
                         : loop_size;
        if (n_loop <= 0) break;
        addr_type act_sp_addr =
            act_sp_base_addr + (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE + (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;

        if (Ms == 0) {
          std::set<addr_type> input_set;
          for (int iter_n = 0; iter_n < n_loop; ++iter_n) {
            for (int iter_c = 0; iter_c < c_in_loop; iter_c += elems_per_access) {
              uint32_t N = N_offset + iter_n;
              uint32_t C = C_offset + iter_c;
              input_set.insert(
                  first_addr + make_address({batch_idx, N, C}, _input_shape));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .size = static_cast<uint32_t>(input_set.size()),
              .src_addrs = std::vector<addr_type>(input_set.begin(), input_set.end()),
              .operand_id = _INPUT_OPERAND,
              .tile_k = mapping.tile_in_loop.C,
              .tile_n = mapping.tile_in_loop.N}));
        }
      }
    }
  }

  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if (m_loop <= 0) break;
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += cloop_size) {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                          ? mapping.total_loop.C - C_offset
                          : cloop_size;
      addr_type weight_sp_addr =
          weight_sp_base_addr +
          (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > static_cast<int>(_rows_per_batch)
                         ? static_cast<int>(_rows_per_batch) - N_offset
                         : loop_size;
        if (n_loop <= 0) break;
        addr_type act_sp_addr =
            act_sp_base_addr + (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE + (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        for (int c_iter = 0; c_iter < c_in_loop;
             c_iter += _config.core_config[target_core].core_height) {
          int c_iter_size =
              c_in_loop - c_iter > _config.core_config[target_core].core_height
                  ? _config.core_config[target_core].core_height
                  : c_in_loop - c_iter;
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::GEMM_PRELOAD,
              .dest_addr = out_sp_addr,
              .size = static_cast<uint32_t>(n_loop),
              .compute_size = static_cast<uint32_t>(n_loop),
              .src_addrs = std::vector<addr_type>{act_sp_addr, weight_sp_addr},
              .tile_m = static_cast<unsigned int>(m_loop),
              .tile_k = static_cast<unsigned int>(c_iter_size),
              .tile_n = static_cast<unsigned int>(n_loop)}));
        }
      }
    }
  }

  if (tout_c_offset + mapping.tile_in_loop.C >= mapping.total_loop.C) {
    for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
      int M_offset = tout_m_offset + Ms;
      int m_loop = M_offset + loop_size > mapping.total_loop.M
                       ? mapping.total_loop.M - M_offset
                       : loop_size;
      if (m_loop <= 0) break;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > static_cast<int>(_rows_per_batch)
                         ? static_cast<int>(_rows_per_batch) - N_offset
                         : loop_size;
        if (n_loop <= 0) break;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE + (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        std::set<addr_type> output_set;
        for (int iter_n = 0; iter_n < n_loop; ++iter_n) {
          for (int iter_m = 0; iter_m < m_loop; iter_m += elems_per_access) {
            uint32_t N = N_offset + iter_n;
            uint32_t M = M_offset + iter_m;
            output_set.insert(
                output_addr + make_address({batch_idx, N, M}, _output_shape));
          }
        }
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = out_sp_addr,
            .size = static_cast<uint32_t>(output_set.size()),
            .src_addrs = std::vector<addr_type>(output_set.begin(), output_set.end()),
            .operand_id = _OUTPUT_OPERAND}));
      }
    }
  }
}
