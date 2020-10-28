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
struct output_callback {
  typename Policy::OutputIterator output;
  uint32_t output_count;
  bool output_enabled = false;
  inline __device__ void operator()(typename Policy::Output value)
  {
    if (output_enabled) { output[output_count] = value; }
    output_count++;
  }
};

template <typename Policy>
struct agent {
  typename Policy::InputIterator d_input;
  typename Policy::OutputCountIterator d_output_count;
  typename Policy::OutputIterator d_output;
  uint32_t num_items;
  typename Policy::SeedOperator seed_op;
  typename Policy::ScanOperator scan_op;
  typename Policy::IntersectionOperator intersection_op;
  typename Policy::StateTileState& state_tile_state;
  typename Policy::IndexTileState& index_tile_state;

  inline __device__ void consume_range(bool const do_output,
                                       uint32_t const num_tiles,
                                       uint32_t const start_tile)
  {
    uint32_t const tile_idx            = start_tile + blockIdx.x;
    uint32_t const tile_offset         = Policy::ITEMS_PER_TILE * tile_idx;
    uint32_t const num_items_remaining = num_items - tile_offset;

    typename Policy::OutputCount num_selected;

    if (tile_idx < num_tiles - 1) {
      num_selected = consume_tile<false>(do_output, tile_idx, tile_offset, Policy::ITEMS_PER_TILE);
    } else {
      num_selected = consume_tile<true>(do_output, tile_idx, tile_offset, num_items_remaining);
      if (threadIdx.x == 0) { *d_output_count = num_selected; }
    }
  }

  template <bool IS_LAST_TILE>
  inline __device__ uint32_t consume_tile(bool const do_output,
                                          uint32_t const tile_idx,
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
        typename Policy::IndexPrefixCallback::TempStorage index_prefix;
        typename Policy::IndexBlockScan::TempStorage index_scan;
      };
    } temp_storage;

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

    __syncthreads();

    // Scan Inputs per Thread
    auto thread_seed  = seed_op(tile_offset + thread_offset, items[0]);
    auto thread_state = thread_seed;
    auto callback     = output_callback<Policy>{d_output};

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        thread_state = scan_op(thread_state, items[i], callback);
      };
    };

    // Intersect Block States and Get Exclusive Thread State
    if (tile_idx == 0) {
      typename Policy::Input block_state;

      Policy::StateBlockScan(temp_storage.state_scan)  //
        .ExclusiveScan(                                //
          thread_state,
          thread_state,
          thread_seed,
          intersection_op,
          block_state);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {  //
        state_tile_state.SetInclusive(0, block_state);
      }

    } else {
      auto prefix_op = Policy::StatePrefixCallback(  //
        state_tile_state,
        temp_storage.state_prefix,
        intersection_op,
        tile_idx);

      Policy::StateBlockScan(temp_storage.state_scan)  //
        .ExclusiveScan(                                //
          thread_state,
          thread_state,
          intersection_op,
          prefix_op);

      // thread_state = prefix_op.GetExclusivePrefix();
    }

    __syncthreads();

    // Count Thread Outputs
    callback.output_count = 0;

    thread_seed = thread_state;

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      if (thread_offset + i < num_items_remaining) {
        thread_state = scan_op(thread_state, items[i], callback);
      }
    }

    // count block outputs and initialize output offsets
    uint32_t tile_output_count;

    if (tile_idx == 0) {
      Policy::IndexBlockScan(temp_storage.index_scan)  //
        .ExclusiveScan(                                //
          callback.output_count,
          callback.output_count,
          cub::Sum(),
          tile_output_count);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        index_tile_state.SetInclusive(0, tile_output_count);
      }
    } else {
      auto prefix_op = Policy::IndexPrefixCallback(  //
        index_tile_state,
        temp_storage.index_prefix,
        cub::Sum(),
        tile_idx);

      Policy::IndexBlockScan(temp_storage.index_scan)  //
        .ExclusiveScan(                                //
          callback.output_count,
          callback.output_count,
          cub::Sum(),
          prefix_op);

      tile_output_count = prefix_op.GetInclusivePrefix();
    }

    // Output

    thread_state = thread_seed;

    if (do_output) {
      callback.output_enabled = true;
      for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
        if (thread_offset + i < num_items_remaining) {
          thread_state = scan_op(thread_state, items[i], callback);
        }
      }
    }

    return tile_output_count;
  }
};

// ===== Kernels ===================================================================================

template <typename Policy>
__global__ void initialization_pass_kernel(  //
  typename Policy::StateTileState items_state,
  typename Policy::IndexTileState index_state,
  uint32_t num_tiles)
{
  items_state.InitializeStatus(num_tiles);
  index_state.InitializeStatus(num_tiles);
}

template <typename Policy>
__global__ void execution_pass_kernel(  //
  typename Policy::InputIterator d_input,
  typename Policy::OutputCountIterator d_output_count,
  typename Policy::OutputIterator d_output,
  uint32_t num_items,
  typename Policy::SeedOperator seed_op,
  typename Policy::ScanOperator scan_op,
  typename Policy::IntersectionOperator intersection_op,
  typename Policy::StateTileState state_tile_state,
  typename Policy::IndexTileState index_tile_state,
  bool do_output,
  uint32_t num_tiles,
  uint32_t start_tile)
{
  auto agent_instance = agent<Policy>{
    d_input,
    d_output_count,
    d_output,
    num_items,
    seed_op,
    scan_op,
    intersection_op,
    state_tile_state,
    index_tile_state  //
  };

  agent_instance.consume_range(do_output, num_tiles, start_tile);
}

// ===== Policy ====================================================================================

template <typename InputIterator_,
          typename OutputCountIterator_,
          typename OutputIterator_,
          typename SeedOperator_,
          typename ScanOperator_,
          typename IntersectionOperator_>
struct policy {
  static constexpr uint32_t THREADS_PER_INIT_BLOCK = 128;
  static constexpr uint32_t THREADS_PER_BLOCK      = 32;
  static constexpr uint32_t ITEMS_PER_THREAD       = 32;
  static constexpr uint32_t ITEMS_PER_TILE         = ITEMS_PER_THREAD * THREADS_PER_BLOCK;

  using InputIterator       = InputIterator_;
  using OutputIterator      = OutputIterator_;
  using OutputCountIterator = OutputCountIterator_;

  using SeedOperator         = SeedOperator_;
  using ScanOperator         = ScanOperator_;
  using IntersectionOperator = IntersectionOperator_;

  using State       = typename std::result_of<SeedOperator>;
  using Offset      = typename std::iterator_traits<InputIterator>::difference_type;
  using Input       = typename std::iterator_traits<InputIterator>::value_type;
  using OutputCount = typename std::iterator_traits<OutputCountIterator>::value_type;
  using Output      = typename std::iterator_traits<OutputIterator>::value_type;

  // Items Load

  using ItemsBlockLoad = cub::BlockLoad<  //
    Input,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  // Items Scan

  using StateTileState = cub::ScanTileState<Input>;
  using StatePrefixCallback =
    cub::TilePrefixCallbackOp<Input, IntersectionOperator, StateTileState>;

  using StateBlockScan = cub::BlockScan<  //
    Input,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;

  // Index Scan

  using IndexTileState      = cub::ScanTileState<uint32_t>;
  using IndexPrefixCallback = cub::TilePrefixCallbackOp<uint32_t, cub::Sum, IndexTileState>;
  using IndexBlockScan      = cub::BlockScan<  //
    uint32_t,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;
};

// ===== Entry =====================================================================================

template <typename InputIterator,
          typename OutputCountIterator,
          typename OutputIterator,
          typename SeedOperator,
          typename ScanOperator,
          typename IntersectionOperator>
void scan_artifacts(  //
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIterator d_in,
  OutputCountIterator d_count_out,
  OutputIterator d_out,
  uint32_t num_items,
  SeedOperator seed_op,
  ScanOperator scan_op,
  IntersectionOperator intersection_op,
  bool do_initialize,
  bool do_output,
  cudaStream_t stream = 0)
{
  CUDF_FUNC_RANGE();

  using Policy = policy<InputIterator,
                        OutputCountIterator,
                        OutputIterator,
                        SeedOperator,
                        ScanOperator,
                        IntersectionOperator>;

  uint32_t num_tiles = ceil_div(num_items, Policy::ITEMS_PER_TILE);

  // calculate temp storage requirements

  void* allocations[2];
  size_t allocation_sizes[2];

  CUDA_TRY(Policy::StateTileState::AllocationSize(num_tiles, allocation_sizes[0]));
  CUDA_TRY(Policy::IndexTileState::AllocationSize(num_tiles, allocation_sizes[1]));

  CUDA_TRY(
    cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));

  if (d_temp_storage == nullptr) { return; }

  // initialize

  typename Policy::StateTileState state_tile_state;
  typename Policy::IndexTileState index_tile_state;

  CUDA_TRY(state_tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
  CUDA_TRY(index_tile_state.Init(num_tiles, allocations[1], allocation_sizes[1]));

  if (do_initialize) {
    uint32_t num_init_blocks = ceil_div(num_tiles, Policy::THREADS_PER_INIT_BLOCK);

    auto init_kernel = initialization_pass_kernel<Policy>;
    init_kernel<<<num_init_blocks, Policy::THREADS_PER_INIT_BLOCK, 0, stream>>>(  //
      state_tile_state,
      index_tile_state,
      num_tiles);

    CHECK_CUDA(stream);
  }

  // execute

  auto exec_kernel = execution_pass_kernel<Policy>;

  uint32_t tiles_per_pass = 1 << 10;

  for (uint32_t start_tile = 0; start_tile < num_tiles; start_tile += tiles_per_pass) {
    tiles_per_pass = std::min(tiles_per_pass, num_tiles - start_tile);

    exec_kernel<<<tiles_per_pass, Policy::THREADS_PER_BLOCK, 0, stream>>>(  //
      d_in,
      d_count_out,
      d_out,
      num_items,
      seed_op,
      scan_op,
      intersection_op,
      state_tile_state,
      index_tile_state,
      do_output,
      num_tiles,
      start_tile);

    CHECK_CUDA(stream);
  }
}

template <typename Result,
          typename InputIterator,
          typename SeedOperator,
          typename ScanOperator,
          typename IntersectionOperator>
rmm::device_uvector<Result>  //
scan_artifacts(              //
  InputIterator d_in_begin,
  InputIterator d_in_end,
  SeedOperator seed_op,
  ScanOperator scan_op,
  IntersectionOperator intersection_op,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_FUNC_RANGE();

  // using Input          = typename std::iterator_traits<InputIterator>::value_type;
  // using State          = std::result_of<SeedOperator>();
  using OutputIterator = Result*;

  // static_assert(scan_op(std::declval<State>, std::declval<Input>, std::declval<OutputIterator>));

  auto d_num_selections = rmm::device_scalar<uint32_t>(0, stream);

  uint64_t temp_storage_bytes;

  // query required temp storage (does not launch kernel)

  scan_artifacts(static_cast<void*>(nullptr),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 static_cast<OutputIterator>(nullptr),
                 static_cast<uint32_t>(d_in_end - d_in_begin),
                 seed_op,
                 scan_op,
                 intersection_op,
                 false,  // do_initialize
                 false,  // do_output
                 stream);

  auto d_temp_storage = rmm::device_buffer(temp_storage_bytes, stream);

  // phase 1 - determine number of results

  scan_artifacts(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 static_cast<OutputIterator>(nullptr),
                 static_cast<uint32_t>(d_in_end - d_in_begin),
                 seed_op,
                 scan_op,
                 intersection_op,
                 true,   // do_initialize
                 false,  // do_output
                 stream);

  auto d_output = rmm::device_uvector<Result>(d_num_selections.value(stream), stream, mr);

  // phase 2 - gather results

  scan_artifacts(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 d_output.data(),
                 static_cast<uint32_t>(d_in_end - d_in_begin),
                 seed_op,
                 scan_op,
                 intersection_op,
                 false,  // do_initialize
                 true,   // do_output
                 stream);

  cudaStreamSynchronize(stream);

  return d_output;
}
