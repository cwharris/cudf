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

struct csv_outputs {
  dfa_output<uint32_t> record_offsets;

  inline constexpr csv_outputs operator+(csv_outputs const& other) const
  {
    return {record_offsets + other.record_offsets};
  }
};

using csv_superstate = dfa_superposition<csv_state, csv_token, 7>;

struct csv_machine_state {
  csv_superstate state;
  uint32_t byte_count  = 0;
  uint32_t row_count   = 0;
  bool is_record_start = false;

  inline constexpr csv_machine_state operator+(csv_machine_state rhs)
  {
    return {
      state + rhs.state,
      byte_count + rhs.byte_count,
      row_count + rhs.row_count,
      rhs.is_record_start,
    };
  }
};

struct csv_fsm_seed_op {
  inline constexpr csv_machine_state operator()(uint32_t position) { return csv_machine_state(); }
};

struct csv_fsm_scan_op {
  csv_token_options tokens_options;
  inline constexpr csv_machine_state operator()(csv_machine_state prev, char current_char)
  {
    auto token = get_token(tokens_options, 0, current_char);

    auto is_new_record = prev.state == csv_state::record_end and token == csv_token::other;

    return {
      prev.state + token,
      prev.byte_count + 1,
      prev.row_count + is_new_record,
      is_new_record,
    };
  }
};

struct csv_fsm_join_op {
  inline constexpr csv_machine_state operator()(csv_machine_state lhs, csv_machine_state rhs)
  {
    return lhs + rhs;
  }
};

struct csv_range_options {
  uint32_t bytes_begin = 0;
  uint32_t bytes_end   = std::numeric_limits<uint32_t>::max();
  uint32_t rows_begin  = 0;
  uint32_t rows_end    = std::numeric_limits<uint32_t>::max();
};

struct csv_fsm_output_op {
  csv_range_options range;

  template <bool output_enabled>
  inline __device__ csv_outputs
  operator()(csv_outputs out, csv_machine_state prev, csv_machine_state next, char current_char)
  {
    if (output_enabled) {
      printf(
        "bid(%2i) tid(%2i): byte(%-4i) char(%2c) state(%i - %i) range(%i, %i) count(%2i) "
        "out(%2i)\n ",
        blockIdx.x,
        threadIdx.x,
        next.byte_count,
        current_char,
        prev.state.states[0],
        next.state.states[0],
        range.bytes_begin,
        range.bytes_end,
        next.row_count,
        out.record_offsets.output_count);
    }

    if (next.byte_count - 1 < range.bytes_begin) { return out; }
    if (next.byte_count - 1 >= range.bytes_end) { return out; }

    if (not next.is_record_start) { return out; }

    if (next.row_count < range.rows_begin) { return out; }
    if (next.row_count >= range.rows_end) { return out; }

    out.record_offsets.emit<output_enabled>(next.byte_count - 1);

    return out;
  }

  // TODO: add finalizer
};

rmm::device_uvector<uint32_t> csv_gather_row_offsets(
  device_span<char> input,
  csv_token_options token_options     = {},
  csv_range_options range_options     = {},
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto seed_op   = csv_fsm_seed_op{};
  auto scan_op   = csv_fsm_scan_op{token_options};
  auto join_op   = csv_fsm_join_op{};
  auto output_op = csv_fsm_output_op{range_options};

  auto d_output_state = rmm::device_scalar<csv_machine_state>(csv_machine_state(), stream, mr);
  auto d_output       = rmm::device_scalar<csv_outputs>(csv_outputs(), stream, mr);

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
