#pragma once

#include <cstdint>
#include <type_traits>

#include <cudf/algorithm/scan_state_machine.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

using cudf::detail::device_span;

namespace cudf {
namespace io {
namespace detail {

// ===== STATES AND TOKENS =====

enum class csv_token : uint8_t {
  comment = 1,
  delimiter,
  quote,
  newline,
  other,
};

enum class csv_state : uint8_t {
  none = 0,
  record_end,
  comment,
  field,
  field_quoted,
  field_quoted_quote,
  field_end,
};

constexpr csv_state operator|(csv_state lhs, csv_state rhs)
{
  using underlying = std::underlying_type_t<csv_state>;
  return static_cast<csv_state>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}

constexpr csv_state operator&(csv_state lhs, csv_state rhs)
{
  using underlying = std::underlying_type_t<csv_state>;
  return static_cast<csv_state>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}

// ===== TRANSITIONS =====

template <csv_state, csv_token>
struct transition {
  static constexpr csv_state destination = csv_state::none;
};

#ifndef TRANSITION_DEFAULT
#define TRANSITION_DEFAULT(from, to)             \
  template <csv_token via>                       \
  struct transition<csv_state::from, via> {      \
    static constexpr csv_state destination = to; \
  }
#endif

#ifndef TRANSITION
#define TRANSITION(from, via, to)                      \
  template <>                                          \
  struct transition<csv_state::from, csv_token::via> { \
    static constexpr csv_state destination = to;       \
  }
#endif

TRANSITION_DEFAULT(comment, csv_state::comment);
TRANSITION(comment, newline, csv_state::record_end);
TRANSITION(record_end, newline, csv_state::record_end);
TRANSITION(record_end, comment, csv_state::comment);
TRANSITION(record_end, other, csv_state::field);
TRANSITION(field, other, csv_state::field);
TRANSITION(field, delimiter, csv_state::field_end);
TRANSITION(field, newline, csv_state::record_end);
TRANSITION(field_end, other, csv_state::field);
TRANSITION(field_end, newline, csv_state::record_end);
TRANSITION(field_end, quote, csv_state::field_quoted);
TRANSITION(record_end, quote, csv_state::field_quoted);
TRANSITION(field_quoted, quote, csv_state::field_quoted_quote);
TRANSITION(field_quoted, other, csv_state::field_quoted);
TRANSITION(field_quoted, newline, csv_state::field_quoted);
TRANSITION(field_quoted, delimiter, csv_state::field_quoted);
TRANSITION(field_quoted, comment, csv_state::field_quoted);
TRANSITION(field_quoted_quote, quote, csv_state::field_quoted);
TRANSITION(field_quoted_quote, delimiter, csv_state::field_end);
TRANSITION(field_quoted_quote, newline, csv_state::record_end);

// ===== DISPATCH =====

template <csv_state state, csv_token token>
inline constexpr csv_state get_next_state()
{
  return transition<state, token>::destination;
}

template <csv_state state>
inline constexpr csv_state get_next_state(csv_token token)
{
  switch (token) {
    case (csv_token::comment):  //
      return get_next_state<state, csv_token::comment>();
    case (csv_token::other):  //
      return get_next_state<state, csv_token::other>();
    case (csv_token::newline):  //
      return get_next_state<state, csv_token::newline>();
    case (csv_token::delimiter):  //
      return get_next_state<state, csv_token::delimiter>();
    case (csv_token::quote):  //
      return get_next_state<state, csv_token::quote>();
  }

  return csv_state::none;
}

inline constexpr csv_state get_next_state(csv_state state, csv_token token)
{
  switch (state) {
    case (csv_state::field_quoted_quote):  //
      return get_next_state<csv_state::field_quoted_quote>(token);
    case (csv_state::field_quoted):  //
      return get_next_state<csv_state::field_quoted>(token);
    case (csv_state::field_end):  //
      return get_next_state<csv_state::field_end>(token);
    case (csv_state::record_end):  //
      return get_next_state<csv_state::record_end>(token);
    case (csv_state::comment):  //
      return get_next_state<csv_state::comment>(token);
    case (csv_state::field):  //
      return get_next_state<csv_state::field>(token);
    case (csv_state::none):  //
      return csv_state::none;
  }

  return csv_state::none;
}

// ===== PARSING =====

inline constexpr csv_token get_token(char prev, char current)
{
  if (prev != '\\') {
    // what happens if we see `\` twice? i.e: "\\\\" <- this should be two backslashes
    // maybe take care of this in the fsm.
    switch (current) {
      case '\n': return csv_token::newline;
      case '#': return csv_token::comment;
      case ',': return csv_token::delimiter;
      case '"': return csv_token::quote;
    }
  }

  return csv_token::other;
}

// ===== parallel state machine for CSV parsing =====W

struct csv_fsm_outputs {
  fsm_output<uint32_t> record_offsets;

  inline constexpr csv_fsm_outputs operator+(csv_fsm_outputs const& other) const
  {
    return {record_offsets + other.record_offsets};
  }
};

struct csv_state_segment {
  inline constexpr csv_state_segment() : csv_state_segment(csv_state::none) {}
  inline constexpr csv_state_segment(csv_state state) : head(state), tail(state) {}
  inline constexpr csv_state_segment(csv_state head, csv_state tail) : head(head), tail(tail) {}
  csv_state head;
  csv_state tail;
};

struct csv_machine_state {
  uint32_t position  = 0;
  uint8_t num_states = 0;
  csv_state_segment states[8];

  constexpr csv_machine_state() {}
  constexpr csv_machine_state(uint32_t position, csv_state_segment state)
    : position(position), num_states(1)
  {
    states[0] = state;
  }

  inline constexpr csv_machine_state operator+(csv_token const& token)
  {
    csv_machine_state result;

    result.position = position + 1;

    for (auto i = 0; i < num_states; i++) {
      auto next_state = get_next_state(states[i].tail, token);
      if (next_state != csv_state::none) {
        result |= csv_state_segment(states[i].head, next_state);
      }
    }

    return result;
  }

  inline constexpr bool operator==(csv_state needle)
  {
    for (auto i = 0; i < num_states; i++) {
      if (states[i].tail == needle) { return true; }
    }
    return false;
  }

  inline constexpr void operator|=(csv_state_segment state) { states[num_states++] = state; }

  inline constexpr csv_machine_state operator&(csv_machine_state const& rhs) const
  {
    // the result of this function is the "real" rhs state.
    csv_machine_state result;

    result.position = position + rhs.position;

    for (auto i = 0; i < num_states; i++) {
      for (auto j = 0; j < rhs.num_states; j++) {
        if (states[i].tail == rhs.states[j].head) {
          result |= csv_state_segment(states[i].head, rhs.states[j].tail);
        }
      }
    }

    return result;
  }
};

struct csv_fsm_seed_op {
  inline constexpr csv_machine_state operator()(uint32_t position)
  {
    if (position == 0) {  //
      return csv_machine_state(position, csv_state_segment(csv_state::record_end));
    }

    auto result = csv_machine_state();

    result.position   = 0;
    result.num_states = 6;
    result.states[0]  = csv_state_segment(csv_state::record_end);
    result.states[1]  = csv_state_segment(csv_state::comment);
    result.states[2]  = csv_state_segment(csv_state::field);
    result.states[3]  = csv_state_segment(csv_state::field_quoted);
    result.states[4]  = csv_state_segment(csv_state::field_quoted_quote);
    result.states[5]  = csv_state_segment(csv_state::field_end);

    return result;
  }
};

struct csv_fsm_scan_op {
  inline constexpr csv_machine_state operator()(csv_machine_state prev, char current_char)
  {
    return prev + get_token(0, current_char);
  }
};

struct csv_fsm_join_op {
  inline constexpr csv_machine_state operator()(csv_machine_state lhs, csv_machine_state rhs)
  {
    return lhs & rhs;
  }
};

struct csv_fsm_output_op {
  template <bool output_enabled>
  inline constexpr csv_fsm_outputs operator()(csv_fsm_outputs out,
                                              csv_machine_state prev,
                                              csv_machine_state next,
                                              char current_char)
  {
    if (not(prev == csv_state::record_end)) { return out; }
    if (not(next == csv_state::field)) { return out; }

    out.record_offsets.emit<output_enabled>(prev.position);

    return out;
  }
};

rmm::device_uvector<uint32_t> csv_gather_row_offsets(
  device_span<char> csv_input,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto seed_op   = csv_fsm_seed_op{};
  auto scan_op   = csv_fsm_scan_op{};
  auto join_op   = csv_fsm_join_op{};
  auto output_op = csv_fsm_output_op{};

  auto d_output_state = rmm::device_scalar<csv_machine_state>(csv_machine_state{}, stream, mr);
  auto d_output       = rmm::device_scalar<csv_fsm_outputs>(csv_fsm_outputs{}, stream, mr);

  rmm::device_buffer temp_memory;

  scan_state_machine(temp_memory,
                     csv_input.begin(),
                     csv_input.end(),
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
                     csv_input.begin(),
                     csv_input.end(),
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
