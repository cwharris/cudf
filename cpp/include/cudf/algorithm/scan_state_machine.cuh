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

template <typename T>
struct fsm_output {
  T* output_buffer;
  uint32_t output_count;

  template <bool output_enabled>
  inline constexpr void emit(T value)
  {
    if (output_enabled) {
      // printf("bid(%i) tid(%i): output %i = %i\n",  //
      //        blockIdx.x,
      //        threadIdx.x,
      //        output_count,
      //        value);
      output_buffer[output_count++] = value;
    } else {
      output_count++;
    }
  }

  inline constexpr fsm_output operator+(fsm_output other) const
  {
    return {output_buffer, output_count + other.output_count};
  }
};

// ===== Agent =====================================================================================

template <typename Policy>
struct agent {
  typename Policy::InputIterator d_input;
  typename Policy::OutputStateIterator d_output_state;
  typename Policy::OutputIterator d_output;
  uint32_t num_items;
  typename Policy::SeedOp seed_op;
  typename Policy::StepOp step_op;
  typename Policy::JoinOp join_op;
  typename Policy::OutputOp output_op;
  typename Policy::StateTileState& state_tile_state;
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
        *d_output_state = state.first;
        *d_output       = state.second;
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
        typename Policy::OutputPrefixCallback::TempStorage output_prefix;
        typename Policy::OutputBlockScan::TempStorage output_scan;
      };
    } temp_storage;

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 0 =====\n", blockIdx.x, threadIdx.x);
    // }

    uint32_t const thread_offset = threadIdx.x * Policy::ITEMS_PER_THREAD;

    // Load Inputs

    typename Policy::Input items[Policy::ITEMS_PER_THREAD];

    if (IS_LAST_TILE) {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items, num_items_remaining);
    } else {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items);
    }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 1 =====\n", blockIdx.x, threadIdx.x);
    // }

    __syncthreads();

    // Scan Inputs per Thread

    auto thread_state_seed        = seed_op(tile_offset + thread_offset);
    auto const thread_output_seed = *d_output;
    auto thread_state             = thread_state_seed;
    auto thread_output            = thread_output_seed;

    // if (thread_offset < num_items_remaining) {
    //   printf("bid(%i) tid(%i): thread: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          thread_state.sum,
    //          thread_output.a.output_count,
    //          thread_output.b.output_count);
    // }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 1 and halvfe =====\n", blockIdx.x, threadIdx.x);
    // }

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {  // remove the if
      if (thread_offset + i < num_items_remaining) {
        thread_state = step_op(thread_state, items[i]);
      }
    }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 2 =====\n", blockIdx.x, threadIdx.x);
    // }

    __syncthreads();

    // if (thread_offset < num_items_remaining) {
    //   printf("bid(%i) tid(%i): thread: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          thread_state.sum,
    //          thread_output.a.output_count,
    //          thread_output.b.output_count);
    // }

    // Scan Inputs

    typename Policy::State block_state;

    if (tile_idx == 0) {
      // if (threadIdx.x == 0) { printf("bid(%i) tid(%i): ===== 3 =====\n", blockIdx.x,
      // threadIdx.x); }

      Policy::StateBlockScan(temp_storage.state_scan)  //
        .ExclusiveScan(                                //
          thread_state,
          thread_state,
          thread_state_seed,
          join_op,
          block_state);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {  //
        state_tile_state.SetInclusive(0, block_state);
      }

    } else {
      auto prefix_op = Policy::StatePrefixCallback(  //
        state_tile_state,
        temp_storage.state_prefix,
        join_op,
        tile_idx);

      Policy::StateBlockScan(temp_storage.state_scan)  //
        .ExclusiveScan(                                //
          thread_state,
          thread_state,
          join_op,
          prefix_op);

      block_state = prefix_op.GetInclusivePrefix();
    }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 3 =====\n", blockIdx.x, threadIdx.x);
    // }

    __syncthreads();
    // if (thread_offset < num_items_remaining) {
    //   printf("bid(%i) tid(%i): thread: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          thread_state.sum,
    //          thread_output.a.output_count,
    //          thread_output.b.output_count);
    // }

    // if (threadIdx.x == 0) {                                     //
    //   printf("bid(%i) tid(%i): block: state (%i) out (x x)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          block_state.sum);
    // }

    // Count Outputs

    thread_output     = *d_output;  // reset state gathering.
    thread_state_seed = thread_state;

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        auto next_state = step_op(thread_state, items[i]);
        thread_output   = output_op.operator()<false>(  //
          thread_output,
          thread_state,
          next_state,
          items[i]);
        thread_state    = next_state;
      }
    }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 4 =====\n", blockIdx.x, threadIdx.x);
    // }

    // __syncthreads();
    // if (thread_offset < num_items_remaining) {
    //   printf("bid(%i) tid(%i): thread: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          thread_state.sum,
    //          thread_output.a.output_count,
    //          thread_output.b.output_count);
    // }

    // if (threadIdx.x == 0) {                                     //
    //   printf("bid(%i) tid(%i): block: state (%i) out (x x)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          block_state.sum);
    // }

    typename Policy::Output block_output;

    if (tile_idx == 0) {
      Policy::OutputBlockScan(temp_storage.output_scan)  //
        .ExclusiveScan(                                  //
          thread_output,
          thread_output,
          *d_output,
          cub::Sum(),
          block_output);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        output_tile_state.SetInclusive(0, block_output);
      }
    } else {
      auto prefix_op = Policy::OutputPrefixCallback(  //
        output_tile_state,
        temp_storage.output_prefix,
        cub::Sum(),
        tile_idx);

      Policy::OutputBlockScan(temp_storage.output_scan)  //
        .ExclusiveScan(                                  //
          thread_output,
          thread_output,
          cub::Sum(),
          prefix_op);

      block_output = prefix_op.GetInclusivePrefix();
    }

    __syncthreads();

    thread_state = thread_state_seed;

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 5 =====\n", blockIdx.x, threadIdx.x);
    // }

    // __syncthreads();
    // if (thread_offset < num_items_remaining) {
    //   printf("bid(%i) tid(%i): thread: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          thread_state.sum,
    //          thread_output.a.output_count,
    //          thread_output.b.output_count);
    // }

    // if (threadIdx.x == 0) {                                       //
    //   printf("bid(%i) tid(%i): block: state (%i) out (%i %i)\n",  //
    //          blockIdx.x,
    //          threadIdx.x,
    //          block_state.sum,
    //          block_output.a.output_count,
    //          block_output.b.output_count);
    // }

    // Collect Outputs

    // thread_state = thread_state_seed;

    if (not is_first_pass) {
      for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
        if (thread_offset + i < num_items_remaining) {
          auto next_state = step_op(thread_state, items[i]);
          thread_output   = output_op.operator()<true>(  //
            thread_output,
            thread_state,
            next_state,
            items[i]);
          thread_state    = next_state;
        }
      }
    }

    // if (threadIdx.x == 0) {  //
    //   printf("bid(%i) tid(%i): ===== 6 =====\n", blockIdx.x, threadIdx.x);
    // }

    return {block_state, block_output};  // also need to output block_output
  }
};

// ===== Kernels ===================================================================================

template <typename Policy>
__global__ void initialization_pass_kernel(  //
  typename Policy::StateTileState state_tile_state,
  typename Policy::OutputTileState output_tile_state,
  uint32_t num_tiles)
{
  state_tile_state.InitializeStatus(num_tiles);
  output_tile_state.InitializeStatus(num_tiles);
}

template <typename Policy>
__global__ void execution_pass_kernel(  //
  typename Policy::InputIterator d_input,
  typename Policy::OutputStateIterator d_output_state,
  typename Policy::OutputIterator d_output,
  uint32_t num_items,
  typename Policy::SeedOp seed_op,
  typename Policy::StepOp step_op,
  typename Policy::JoinOp join_op,
  typename Policy::OutputOp output_op,
  typename Policy::StateTileState state_tile_state,
  typename Policy::OutputTileState output_tile_state,
  bool is_first_pass,
  uint32_t num_tiles,
  uint32_t start_tile)
{
  auto agent_instance = agent<Policy>{
    d_input,
    d_output_state,
    d_output,
    num_items,
    seed_op,
    step_op,
    join_op,
    output_op,
    state_tile_state,
    output_tile_state,
    is_first_pass,
  };

  agent_instance.consume_range(num_tiles, start_tile);
}

// ===== Policy ====================================================================================

template <typename InputIterator_,
          typename OutputStateIterator_,
          typename OutputIterator_,
          typename StateSeedOp_,
          typename StateStepOp_,
          typename StateJoinOp_,
          typename OutputOp_>
struct policy {
  static constexpr uint32_t THREADS_PER_INIT_BLOCK = 128;
  static constexpr uint32_t THREADS_PER_BLOCK      = 32;
  static constexpr uint32_t ITEMS_PER_THREAD       = 2;
  static constexpr uint32_t ITEMS_PER_TILE         = ITEMS_PER_THREAD * THREADS_PER_BLOCK;

  using InputIterator       = InputIterator_;
  using OutputStateIterator = OutputStateIterator_;

  using SeedOp   = StateSeedOp_;
  using StepOp   = StateStepOp_;
  using JoinOp   = StateJoinOp_;
  using OutputOp = OutputOp_;

  using OutputIterator = OutputIterator_;

  using Input  = typename std::iterator_traits<InputIterator>::value_type;
  using Offset = typename std::iterator_traits<InputIterator>::difference_type;
  using State  = typename std::iterator_traits<OutputStateIterator>::value_type;
  using Output = typename std::iterator_traits<OutputIterator>::value_type;

  // Items Load

  using ItemsBlockLoad = cub::BlockLoad<  //
    Input,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  // State Scan

  using StateTileState      = cub::ScanTileState<State>;
  using StatePrefixCallback = cub::TilePrefixCallbackOp<State, JoinOp, StateTileState>;

  using StateBlockScan = cub::BlockScan<  //
    State,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  using OutputTileState      = cub::ScanTileState<Output>;
  using OutputPrefixCallback = cub::TilePrefixCallbackOp<Output, cub::Sum, OutputTileState>;

  using OutputBlockScan = cub::BlockScan<  //
    Output,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;
};

// ===== Entry =====================================================================================

template <typename InputIterator,
          typename OutputStateIterator,
          typename OutputIterator,
          typename SeedOp,
          typename StepOp,
          typename JoinOp,
          typename OutputOp>
void scan_state_machine(  //
  rmm::device_buffer& temp_storage,
  InputIterator d_in_begin,
  InputIterator d_in_end,
  OutputStateIterator d_output_state,
  OutputIterator d_output,
  SeedOp seed_op,
  StepOp step_op,
  JoinOp join_op,
  OutputOp output_op,
  cudaStream_t stream = 0)
{
  CUDF_FUNC_RANGE();

  using Policy =
    policy<InputIterator, OutputStateIterator, OutputIterator, SeedOp, StepOp, JoinOp, OutputOp>;

  uint32_t num_tiles = ceil_div(d_in_end - d_in_begin, Policy::ITEMS_PER_TILE);

  // calculate temp storage requirements

  void* allocations[2]         = {};
  uint64_t allocation_sizes[2] = {};

  CUDA_TRY(Policy::StateTileState::AllocationSize(num_tiles, allocation_sizes[0]));
  CUDA_TRY(Policy::OutputTileState::AllocationSize(num_tiles, allocation_sizes[1]));

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
  typename Policy::OutputTileState output_tile_state;

  CUDA_TRY(state_tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
  CUDA_TRY(output_tile_state.Init(num_tiles, allocations[1], allocation_sizes[1]));

  // // ideal we could avoid the upsweep by relying on prior results
  // if (is_first_pass) {
  uint32_t num_init_blocks = ceil_div(num_tiles, Policy::THREADS_PER_INIT_BLOCK);

  auto init_kernel = initialization_pass_kernel<Policy>;
  init_kernel<<<num_init_blocks, Policy::THREADS_PER_INIT_BLOCK, 0, stream>>>(  //
    state_tile_state,
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
      d_output_state,
      d_output,
      d_in_end - d_in_begin,
      seed_op,
      step_op,
      join_op,
      output_op,
      state_tile_state,
      output_tile_state,
      is_first_pass,
      num_tiles,
      start_tile);

    CHECK_CUDA(stream);
  }
}
