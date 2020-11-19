/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/transform_output_iterator.h>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#include <iterator>
#include <type_traits>

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

// ===== Agent =====================================================================================

template <typename Policy>
struct agent {
  typename Policy::InputIterator d_input;
  typename Policy::InOutStateIterator d_inout_state;
  typename Policy::InOutAggregatesIterator d_inout_aggregates;
  typename Policy::InOutOutputsIterator d_inout_outputs;
  uint32_t num_items;
  typename Policy::ScanStateOp scan_state_op;
  typename Policy::ScanAggregateOp scan_aggregate_op;
  typename Policy::OutputOp output_op;
  typename Policy::StateTileState& state_tile_state;
  typename Policy::AggregateTileState& aggregates_tile_state;
  typename Policy::OutputTileState& output_tile_state;
  bool is_first_pass;

  inline __device__ void consume_range(uint32_t const num_tiles, uint32_t const start_tile)
  {
    uint32_t const tile_idx            = start_tile + blockIdx.x;
    uint32_t const tile_offset         = Policy::ITEMS_PER_TILE * tile_idx;
    uint32_t const num_items_remaining = num_items - tile_offset;

    if (tile_idx < num_tiles - 1) {
      consume_tile<false>(tile_idx, tile_offset, num_items_remaining);
    } else {
      auto state = consume_tile<true>(tile_idx, tile_offset, num_items_remaining);

      if (threadIdx.x == 0) {
        // printf("bid(%i) tid(%i): ===== final outputs =====\n", blockIdx.x, threadIdx.x);
        *d_inout_state   = state.first;
        *d_inout_outputs = state.second;
        // printf("bid(%i) tid(%i): ===== final outputs - end =====\n", blockIdx.x, threadIdx.x);
      }
    }
  }

  template <bool IS_LAST_TILE>
  inline __device__ thrust::pair<typename Policy::State, typename Policy::Output>  //
  consume_tile(uint32_t const tile_idx,
               uint32_t const tile_offset,
               uint32_t const num_items_remaining)
  {
    __shared__ union {
      typename Policy::ItemsBlockLoad::TempStorage item_load;
      struct {
        typename Policy::StatePrefixCallback::TempStorage state_prefix;
        typename Policy::StateBlockScan::TempStorage state_scan;
      };
      struct {
        typename Policy::AggregatePrefixCallback::TempStorage aggregate_prefix;
        typename Policy::AggregateBlockScan::TempStorage aggregate_scan;
      };
      struct {
        typename Policy::OutputPrefixCallback::TempStorage output_prefix;
        typename Policy::OutputBlockScan::TempStorage output_scan;
      };
    } temp_storage;

    uint32_t const thread_offset = threadIdx.x * Policy::ITEMS_PER_THREAD;

    // 1: Load Inputs
    // if (threadIdx.x == 0) { printf("bid(%2i) tid(%2i) Stage 1\n", blockIdx.x, threadIdx.x); }

    typename Policy::Input items[Policy::ITEMS_PER_THREAD];

    if (IS_LAST_TILE) {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items, num_items_remaining);
    } else {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items);
    }

    __syncthreads();

    // 2: Scan State
    // if (threadIdx.x == 0) { printf("bid(%2i) tid(%2i) Stage 2\n", blockIdx.x, threadIdx.x); }

    auto const thread_state_seed     = *d_inout_state;
    auto const thread_aggregate_seed = *d_inout_aggregates;
    auto const thread_output_seed    = *d_inout_outputs;
    auto thread_state                = thread_state_seed;
    auto thread_aggregate            = thread_aggregate_seed;
    auto thread_output               = thread_output_seed;

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        thread_state = scan_state_op(thread_state, items[i]);
      }
    }

    __syncthreads();

    typename Policy::State block_state;

    scan_intermediates<IS_LAST_TILE,
                       typename Policy::StateBlockScan,
                       typename Policy::StatePrefixCallback>(  //
      temp_storage.state_prefix,
      temp_storage.state_scan,
      state_tile_state,
      tile_idx,
      thread_state_seed,
      thread_state,
      block_state);

    __syncthreads();

    auto const thread_state_prefix = thread_state;

    // 3: Scan Aggregate
    // if (threadIdx.x == 0) { printf("bid(%2i) tid(%2i) Stage 3\n", blockIdx.x, threadIdx.x); }

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        thread_state     = scan_state_op(thread_state, items[i]);
        thread_aggregate = scan_aggregate_op(thread_aggregate, thread_state);
      }
    }

    typename Policy::Aggregate block_aggregates;

    scan_intermediates<IS_LAST_TILE,
                       typename Policy::AggregateBlockScan,
                       typename Policy::AggregatePrefixCallback>(  //
      temp_storage.aggregate_prefix,
      temp_storage.aggregate_scan,
      aggregates_tile_state,
      tile_idx,
      thread_aggregate_seed,
      thread_aggregate,
      block_aggregates);

    __syncthreads();

    auto const thread_aggregate_prefix = thread_aggregate;

    // 4: Scan Output Count
    // if (threadIdx.x == 0) { printf("bid(%2i) tid(%2i) Stage 4\n", blockIdx.x, threadIdx.x); }
    thread_state = thread_state_prefix;

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        thread_state     = scan_state_op(thread_state, items[i]);
        thread_aggregate = scan_aggregate_op(thread_aggregate, thread_state);
        thread_output    = output_op.operator()<false>(thread_output, thread_aggregate);
      }
    }

    typename Policy::Output block_output;

    scan_intermediates<IS_LAST_TILE,
                       typename Policy::OutputBlockScan,
                       typename Policy::OutputPrefixCallback>(  //
      temp_storage.output_prefix,
      temp_storage.output_scan,
      output_tile_state,
      tile_idx,
      thread_output_seed,
      thread_output,
      block_output);

    __syncthreads();

    auto const thread_output_prefix = thread_output;

    // 5: Scan Output Mutation
    // if (threadIdx.x == 0) { printf("bid(%2i) tid(%2i) Stage 5\n", blockIdx.x, threadIdx.x); }
    thread_state     = thread_state_prefix;
    thread_aggregate = thread_aggregate_prefix;

    if (not is_first_pass) {
      for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
        if (thread_offset + i < num_items_remaining) {
          thread_state     = scan_state_op(thread_state, items[i]);
          thread_aggregate = scan_aggregate_op(thread_aggregate, thread_state);
          thread_output    = output_op.operator()<true>(thread_output, thread_aggregate);
        }
      }
    }

    return {block_state, block_output};
  }

  template <bool IS_LAST_TILE,
            typename BlockScan,
            typename PrefixCallback,
            typename PrefixStorage,
            typename ScanStorage,
            typename TileState,
            typename Intermediate>
  static inline __device__ void scan_intermediates(  //
    PrefixStorage& prefix_storage,
    ScanStorage& scan_storage,
    TileState& tile_storage,
    uint32_t tile_idx,
    Intermediate const& thread_state_seed,
    Intermediate& thread_state,
    Intermediate& block_state)
  {
    if (tile_idx == 0) {
      BlockScan(scan_storage)  //
        .ExclusiveScan(        //
          thread_state,
          thread_state,
          thread_state_seed,
          cub::Sum(),
          block_state);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {  //
        tile_storage.SetInclusive(0, block_state);
      }

    } else {
      auto prefix_op = PrefixCallback(  //
        tile_storage,
        prefix_storage,
        cub::Sum(),
        tile_idx);

      BlockScan(scan_storage)  //
        .ExclusiveScan(        //
          thread_state,
          thread_state,
          cub::Sum(),
          prefix_op);

      block_state = prefix_op.GetInclusivePrefix();
    }
  }
};

// ===== Kernels ===================================================================================

template <typename Policy>
__global__ void initialization_pass_kernel(  //
  typename Policy::StateTileState state_tile_state,
  typename Policy::AggregateTileState aggregates_tile_state,
  typename Policy::OutputTileState output_tile_state,
  uint32_t num_tiles)
{
  state_tile_state.InitializeStatus(num_tiles);
  aggregates_tile_state.InitializeStatus(num_tiles);
  output_tile_state.InitializeStatus(num_tiles);
}

template <typename Policy>
__global__ void execution_pass_kernel(  //
  typename Policy::InputIterator d_input,
  typename Policy::InOutStateIterator d_inout_state,
  typename Policy::InOutAggregatesIterator d_inout_aggregates,
  typename Policy::InOutOutputsIterator d_inout_outputs,
  uint32_t num_items,
  typename Policy::ScanStateOp scan_state_op,
  typename Policy::ScanAggregateOp scan_aggregate_op,
  typename Policy::OutputOp output_op,
  typename Policy::StateTileState state_tile_state,
  typename Policy::AggregateTileState aggregates_tile_state,
  typename Policy::OutputTileState output_tile_state,
  bool is_first_pass,
  uint32_t num_tiles,
  uint32_t start_tile)
{
  auto agent_instance = agent<Policy>{
    d_input,
    d_inout_state,
    d_inout_aggregates,
    d_inout_outputs,
    num_items,
    scan_state_op,
    scan_aggregate_op,
    output_op,
    state_tile_state,
    aggregates_tile_state,
    output_tile_state,
    is_first_pass,
  };

  agent_instance.consume_range(num_tiles, start_tile);
}

// ===== Policy ====================================================================================

template <typename InputIterator_,
          typename InOutStateIterator_,
          typename InOutAggregatesIterator_,
          typename InOutOutputsIterator_,
          typename StateScanStateOp_,
          typename ScanAggregateOp_,
          typename OutputOp_>
struct policy {
  static constexpr uint32_t THREADS_PER_INIT_BLOCK = 128;
  static constexpr uint32_t THREADS_PER_BLOCK      = 32;
  static constexpr uint32_t ITEMS_PER_THREAD       = 32;
  static constexpr uint32_t ITEMS_PER_TILE         = ITEMS_PER_THREAD * THREADS_PER_BLOCK;

  using InputIterator           = InputIterator_;
  using InOutStateIterator      = InOutStateIterator_;
  using InOutAggregatesIterator = InOutAggregatesIterator_;
  using InOutOutputsIterator    = InOutOutputsIterator_;

  using ScanStateOp     = StateScanStateOp_;
  using ScanAggregateOp = ScanAggregateOp_;
  using OutputOp        = OutputOp_;

  using Input     = typename std::iterator_traits<InputIterator>::value_type;
  using Offset    = typename std::iterator_traits<InputIterator>::difference_type;
  using State     = typename std::iterator_traits<InOutStateIterator>::value_type;
  using Aggregate = typename std::iterator_traits<InOutAggregatesIterator>::value_type;
  using Output    = typename std::iterator_traits<InOutOutputsIterator>::value_type;

  // Items Load

  using ItemsBlockLoad = cub::BlockLoad<  //
    Input,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  // Scan State

  using StateTileState      = cub::ScanTileState<State>;
  using StatePrefixCallback = cub::TilePrefixCallbackOp<State, cub::Sum, StateTileState>;

  using StateBlockScan = cub::BlockScan<  //
    State,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  // Scan Aggregate

  using AggregateTileState = cub::ScanTileState<Aggregate>;
  using AggregatePrefixCallback =
    cub::TilePrefixCallbackOp<Aggregate, cub::Sum, AggregateTileState>;

  using AggregateBlockScan = cub::BlockScan<  //
    Aggregate,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  // Scan Output

  using OutputTileState      = cub::ScanTileState<Output>;
  using OutputPrefixCallback = cub::TilePrefixCallbackOp<Output, cub::Sum, OutputTileState>;

  using OutputBlockScan = cub::BlockScan<  //
    Output,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;
};

// ===== Entry =====================================================================================

template <typename InputIterator,
          typename InOutStateIterator,
          typename InOutAggregatesIterator,
          typename InOutOutputsIterator,
          typename ScanStateOp,
          typename ScanAggregateOp,
          typename OutputOp>
void scan_state_machine(  //
  rmm::device_buffer& temp_storage,
  InputIterator d_in_begin,
  InputIterator d_in_end,
  InOutStateIterator d_inout_state,
  InOutAggregatesIterator d_inout_aggregates,
  InOutOutputsIterator d_inout_outputs,
  ScanStateOp scan_state_op,
  ScanAggregateOp scan_aggregate_op,
  OutputOp output_op,
  cudaStream_t stream = 0)
{
  CUDF_FUNC_RANGE();

  using Policy = policy<InputIterator,
                        InOutStateIterator,
                        InOutAggregatesIterator,
                        InOutOutputsIterator,
                        ScanStateOp,
                        ScanAggregateOp,
                        OutputOp>;

  uint32_t num_tiles = ceil_div(d_in_end - d_in_begin, Policy::ITEMS_PER_TILE);

  // calculate temp storage requirements

  void* allocations[3]         = {};
  uint64_t allocation_sizes[3] = {};

  CUDA_TRY(Policy::StateTileState::AllocationSize(num_tiles, allocation_sizes[0]));
  CUDA_TRY(Policy::AggregateTileState::AllocationSize(num_tiles, allocation_sizes[1]));
  CUDA_TRY(Policy::OutputTileState::AllocationSize(num_tiles, allocation_sizes[2]));

  uint64_t temp_storage_bytes;

  CUDA_TRY(cub::AliasTemporaries(nullptr,  //
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes));

  auto const is_first_pass = temp_storage.size() != temp_storage_bytes;

  if (is_first_pass) { temp_storage = rmm::device_buffer(temp_storage_bytes, stream); }

  CUDA_TRY(cub::AliasTemporaries(temp_storage.data(),  //
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes));

  // initialize

  typename Policy::StateTileState state_tile_state;
  typename Policy::AggregateTileState aggregates_tile_state;
  typename Policy::OutputTileState output_tile_state;

  CUDA_TRY(state_tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
  CUDA_TRY(aggregates_tile_state.Init(num_tiles, allocations[1], allocation_sizes[1]));
  CUDA_TRY(output_tile_state.Init(num_tiles, allocations[2], allocation_sizes[2]));

  // // ideal we could avoid the upsweep by relying on prior results
  // if (is_first_pass) {
  uint32_t num_init_blocks = ceil_div(num_tiles, Policy::THREADS_PER_INIT_BLOCK);

  auto init_kernel = initialization_pass_kernel<Policy>;
  init_kernel<<<num_init_blocks, Policy::THREADS_PER_INIT_BLOCK, 0, stream>>>(  //
    state_tile_state,
    aggregates_tile_state,
    output_tile_state,
    num_tiles);

  CHECK_CUDA(stream);
  // }

  auto exec_kernel = execution_pass_kernel<Policy>;

  uint32_t tiles_per_pass = 1 << 10;

  for (uint32_t start_tile = 0; start_tile < num_tiles; start_tile += tiles_per_pass) {
    tiles_per_pass = std::min(tiles_per_pass, num_tiles - start_tile);

    exec_kernel<<<tiles_per_pass, Policy::THREADS_PER_BLOCK, 0, stream>>>(  //
      d_in_begin,
      d_inout_state,
      d_inout_aggregates,
      d_inout_outputs,
      d_in_end - d_in_begin,
      scan_state_op,
      scan_aggregate_op,
      output_op,
      state_tile_state,
      aggregates_tile_state,
      output_tile_state,
      is_first_pass,
      num_tiles,
      start_tile);

    CHECK_CUDA(stream);
  }
}
