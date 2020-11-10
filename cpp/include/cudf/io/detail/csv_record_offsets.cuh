#pragma once

#include <cstdint>
#include <limits>

#include <cudf/algorithm/scan_state_machine.cuh>
#include <cudf/io/detail/csv_state.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

using cudf::detail::device_span;

namespace cudf {
namespace io {
namespace detail {

// ===== PARSING =====

struct csv_token_options {
  char escape     = '\\';
  char terminator = '\n';
  char delimiter  = ',';
  char quote      = '"';  // 0x100 is current default
  char comment    = '#';  // 0x100 is current default
};

inline constexpr csv_token get_token(csv_token_options const& options, char prev, char current)
{
  // what happens if we see the escape char multiple times?
  // i.e: "\\\\" <- this should be two backslashes
  // not sure if fsm can handle this since it'd make state non-deterministic.
  // that said, it could be a flag on the state which gets carried over.
  if (prev == options.escape) { return csv_token::other; }

  if (current == options.delimiter) { return csv_token::delimiter; }
  if (current == options.comment) { return csv_token::comment; }
  if (current == options.quote) { return csv_token::quote; }
  if (current == options.terminator) { return csv_token::newline; }

  return csv_token::other;
}

// ===== parallel state machine for CSV parsing =====

struct csv_fsm_outputs {
  fsm_output<uint32_t> record_offsets;

  inline constexpr csv_fsm_outputs operator+(csv_fsm_outputs const& other) const
  {
    return {record_offsets + other.record_offsets};
  }
};

struct csv_state_segments {
  static constexpr uint8_t N = 7;
  csv_state segments[N]      = {};

  inline constexpr csv_state_segments()
  {
    for (auto i = 0; i < N; i++) {  //
      segments[i] = static_cast<csv_state>(i);
    }
  }

  inline constexpr csv_state_segments(csv_state state)
  {
    for (auto i = 0; i < N; i++) {  //
      segments[i] = state;
    }
  }

  inline constexpr csv_state_segments operator+(csv_token const token)
  {
    csv_state_segments result;
    for (auto i = 0; i < N; i++) {  //
      result.segments[i] = get_next_state(segments[i], token);
    }
    return result;
  }

  inline constexpr csv_state_segments operator+(csv_state_segments rhs)
  {
    csv_state_segments result;
    for (auto i = 0; i < N; i++) {  //
      result.segments[i] = rhs.segments[static_cast<uint8_t>(segments[i])];
    }
    return result;
  }

  inline constexpr bool operator==(csv_state state) { return segments[0] == state; }
};

inline constexpr bool operator==(csv_state state, csv_state_segments segments)
{
  return segments == state;
}

struct csv_machine_state {
  uint32_t position = 0;
  csv_state_segments states;

  inline constexpr csv_machine_state() {}

  inline constexpr csv_machine_state(uint32_t position)  //
    : position(position)
  {
  }

  inline constexpr csv_machine_state(uint32_t position, csv_state state)  //
    : position(position), states(state)
  {
  }

  inline constexpr csv_machine_state(uint32_t position, csv_state_segments states)  //
    : position(position), states(states)
  {
  }

  inline constexpr csv_machine_state operator+(csv_token rhs)
  {
    return {position + 1, states + rhs};
  }
  inline constexpr csv_machine_state operator+(csv_machine_state rhs)
  {
    return {position + rhs.position, states + rhs.states};
  }
};

struct csv_fsm_seed_op {
  inline constexpr csv_machine_state operator()(uint32_t position) { return csv_machine_state{0}; }
};

struct csv_fsm_scan_op {
  csv_token_options tokens_options;
  inline constexpr csv_machine_state operator()(csv_machine_state prev, char current_char)
  {
    return prev + get_token(tokens_options, 0, current_char);
  }
};

struct csv_fsm_join_op {
  inline constexpr csv_machine_state operator()(csv_machine_state lhs, csv_machine_state rhs)
  {
    return lhs + rhs;
  }
};

struct csv_output_ranges {
  uint32_t bytes_begin = 0;
  uint32_t bytes_end   = std::numeric_limits<uint32_t>::max();
  uint32_t rows_begin  = 0;
  uint32_t rows_end    = std::numeric_limits<uint32_t>::max();
};

struct csv_fsm_output_op {
  csv_output_ranges range;

  template <bool output_enabled>
  inline __device__ csv_fsm_outputs
  operator()(csv_fsm_outputs out, csv_machine_state prev, csv_machine_state next, char current_char)
  {
    if (prev.position < range.bytes_begin) { return out; }
    if (prev.position >= range.bytes_end) { return out; }
    if (not(prev.states == csv_state::record_end)) { return out; }
    if (not(next.states == csv_state::field)) { return out; }

    out.record_offsets.emit<output_enabled>(prev.position);

    // if (output_enabled) {
    //   printf("bid(%2i) tid(%2i): pos(%4i -> %-4i) char(%2c) out(%2i)\n",
    //          blockIdx.x,
    //          threadIdx.x,
    //          prev.position,
    //          next.position,
    //          current_char,
    //          out.record_offsets.output_count);
    // }

    return out;
  }

  // TODO: add finalizer
};

rmm::device_uvector<uint32_t> csv_gather_row_offsets(
  device_span<char> input,
  csv_token_options token_options     = {},
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto seed_op   = csv_fsm_seed_op{};
  auto scan_op   = csv_fsm_scan_op{token_options};
  auto join_op   = csv_fsm_join_op{};
  auto output_op = csv_fsm_output_op{};

  auto d_output_state = rmm::device_scalar<csv_machine_state>(csv_machine_state{0}, stream, mr);
  auto d_output       = rmm::device_scalar<csv_fsm_outputs>(csv_fsm_outputs{}, stream, mr);

  rmm::device_buffer temp_memory;

  scan_state_machine(temp_memory,
                     input.begin(),
                     input.end(),
                     d_output_state.data(),
                     d_output.data(),
                     seed_op,
                     scan_op,
                     join_op,
                     output_op,
                     stream);

  auto h_output = d_output.value(stream);
  // auto h_output_state = d_output.value(stream);

  auto d_record_offsets =
    rmm::device_uvector<uint32_t>(h_output.record_offsets.output_count, stream, mr);

  h_output                              = {};
  h_output.record_offsets.output_buffer = d_record_offsets.data();

  d_output.set_value(h_output, stream);

  scan_state_machine(temp_memory,
                     input.begin(),
                     input.end(),
                     d_output_state.data(),
                     d_output.data(),
                     seed_op,
                     scan_op,
                     join_op,
                     output_op,
                     stream);

  return d_record_offsets;
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
