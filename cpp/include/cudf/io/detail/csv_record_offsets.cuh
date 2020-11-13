#pragma once

#include <cstdint>
#include <exception>
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

struct csv_range_options {
  uint64_t bytes_begin = 0;
  uint64_t bytes_end   = std::numeric_limits<uint64_t>::max();
  uint64_t rows_begin  = 0;
  uint64_t rows_end    = std::numeric_limits<uint64_t>::max();
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
  dfa_output<uint64_t> record_offsets;

  inline constexpr csv_outputs operator+(csv_outputs const& rhs) const
  {
    return {record_offsets + rhs.record_offsets};
  }
};

using csv_superstate = dfa_superstate<csv_state, csv_token, 7>;

struct csv_machine_state {
  csv_superstate superstate;
  csv_state state_prev;
  uint64_t offset;
  inline constexpr csv_machine_state operator+(csv_machine_state const rhs) const
  {
    return {
      superstate + rhs.superstate,
      rhs.state_prev,
      offset + rhs.offset,
    };
  }
};

struct csv_fsm_state_scan_op {
  csv_token_options const tokens_options;
  inline constexpr csv_machine_state operator()(csv_machine_state machine_state,
                                                char current_char) const
  {
    return {
      machine_state.superstate + get_token(tokens_options, 0, current_char),
      static_cast<csv_state>(machine_state.superstate),
      machine_state.offset + 1,
    };
  }
};

struct csv_aggregates {
  csv_state state;
  csv_state state_prev;
  uint64_t offset;
  uint64_t record_begin;
  uint64_t record_count;
  inline constexpr csv_aggregates operator+(csv_aggregates const rhs) const
  {
    auto const is_new_record = state == csv_state::record_end || offset != rhs.record_begin;
    return {
      rhs.state,
      rhs.state_prev,
      rhs.offset,
      is_new_record ? rhs.record_begin : record_begin,
      record_count + rhs.record_count,
    };
  }
};

struct csv_aggregates_scan_op {
  inline constexpr csv_aggregates operator()(csv_aggregates prev,
                                             csv_machine_state machine_state) const
  {
    auto const state         = static_cast<csv_state>(machine_state.superstate);
    auto const is_new_record = machine_state.state_prev == csv_state::record_end;

    return {
      state,
      machine_state.state_prev,
      machine_state.offset,
      is_new_record ? machine_state.offset - 1 : prev.record_begin,
      is_new_record + prev.record_count,
    };
  }
};

struct csv_fsm_output_op {
  csv_range_options range;
  template <bool output_enabled>
  inline constexpr csv_outputs operator()(csv_outputs out, csv_aggregates agg)
  {
    if (agg.state != csv_state::record_end) {
      // this is not the end of a record
      return out;
    }

    if (agg.state_prev == csv_state::comment) {
      // ignore comments
      return out;
    }

    if (agg.record_begin < range.bytes_begin ||  //
        agg.record_begin >= range.bytes_end) {
      return out;
    }

    if (agg.record_count < range.rows_begin ||  //
        agg.record_count >= range.rows_end) {
      return out;
    }

    out.record_offsets.emit<output_enabled>(agg.record_begin);

    return out;
  }
};

rmm::device_uvector<uint64_t> csv_gather_row_offsets(
  device_span<char> input,
  csv_token_options token_options     = {},
  csv_range_options range_options     = {},
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto state_scan_op      = csv_fsm_state_scan_op{token_options};
  auto aggregates_scan_op = csv_aggregates_scan_op{};
  auto output_op          = csv_fsm_output_op{range_options};

  auto d_state      = rmm::device_scalar<csv_machine_state>(csv_machine_state(), stream, mr);
  auto d_aggregates = rmm::device_scalar<csv_aggregates>(csv_aggregates(), stream, mr);
  auto d_outputs    = rmm::device_scalar<csv_outputs>(csv_outputs(), stream, mr);

  rmm::device_buffer temp_memory = {};

  scan_state_machine(temp_memory,
                     input.begin(),
                     input.end(),
                     d_state.data(),
                     d_aggregates.data(),
                     d_outputs.data(),
                     state_scan_op,
                     aggregates_scan_op,
                     output_op,
                     stream);

  auto h_output = d_outputs.value(stream);
  // auto h_output_state = d_outputs.value(stream);

  auto d_record_offsets =
    rmm::device_uvector<uint64_t>(h_output.record_offsets.output_count, stream, mr);

  h_output                              = {};
  h_output.record_offsets.output_buffer = d_record_offsets.data();

  // reset outputs before second call.
  d_state.set_value({}, stream);
  d_aggregates.set_value({}, stream);
  d_outputs.set_value(h_output, stream);

  scan_state_machine(temp_memory,
                     input.begin(),
                     input.end(),
                     d_state.data(),
                     d_aggregates.data(),
                     d_outputs.data(),
                     state_scan_op,
                     aggregates_scan_op,
                     output_op,
                     stream);

  return d_record_offsets;
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
